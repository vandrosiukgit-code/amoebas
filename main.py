import math
import multiprocessing
import random
import json
import unittest
import copy
import time
from collections import deque
from functools import lru_cache
from pathlib import Path

import pygame
import tkinter as tk
from tkinter import messagebox, ttk


WORLD_WIDTH, WORLD_HEIGHT = 800, 600
CELL_SIZE = 42
MAZE_COLS, MAZE_ROWS = WORLD_WIDTH // CELL_SIZE, WORLD_HEIGHT // CELL_SIZE
MAZE_OFFSET_X = (WORLD_WIDTH - MAZE_COLS * CELL_SIZE) // 2
MAZE_OFFSET_Y = (WORLD_HEIGHT - MAZE_ROWS * CELL_SIZE) // 2
GRID_WIDTH, GRID_HEIGHT = WORLD_WIDTH // 16, WORLD_HEIGHT // 16
FIELD_CELL = 16
FIELD_WIDTH, FIELD_HEIGHT = WORLD_WIDTH // FIELD_CELL, WORLD_HEIGHT // FIELD_CELL
ARENA = pygame.Rect(0, 0, WORLD_WIDTH, WORLD_HEIGHT)
FPS = 60
ALPHA = 0.1
GAMMA = 0.9
START_EPSILON = 1.0
MIN_EPSILON = 0.05
EPSILON_DECAY = 0.0005
ACTION_NAMES = ("прямо", "налево", "направо")
GRADIENT_NAMES = ("слева", "впереди", "справа", "сзади", "не выражен")
LIGHT_FIELD_REACH = 450.0
HEAT_FIELD_REACH = 720.0
FOOD_LIGHT_THRESHOLD = 0.68
FOOD_CELLS = ((2, 2), (16, 2), (2, 11), (16, 11), (9, 4), (9, 9))
FOOD_RADIUS = round(LIGHT_FIELD_REACH * (1.0 - math.sqrt(FOOD_LIGHT_THRESHOLD)))
MAX_SATIETY = 130.0
EGG_SATIETY_THRESHOLD = 130.0
EGG_REWARD = 100.0
SESSION_VALUE_GROWTH = 2.0
Q_TABLE_FORMAT_VERSION = 3
PREDATOR_BASE_SPEED_RATIO = 0.7
PREDATOR_HUNGER_DRAIN_RATIO = 0.4
AMOEBA_SCENT_REACH = 1050.0
PREDATOR_EAT_SECONDS = 3.0
GENE_LEVELS = 11
MOVE_ENERGY_PER_PIXEL = 0.025
MIN_AMOEBA_FOR_PREDATOR = 4
MIN_REACHABLE_PREY_FOR_PREDATOR = 2
EGG_HATCH_SECONDS = 10.0


class SimulationLogger:
    """Append-only JSONL telemetry for tuning simulation behavior."""

    def __init__(self):
        log_dir = Path(__file__).resolve().parent / "simulation_logs"
        log_dir.mkdir(exist_ok=True)
        run_id = time.strftime("%Y%m%d-%H%M%S") + f"-{time.time_ns() % 1_000_000:06d}"
        self.path = log_dir / f"run-{run_id}.jsonl"
        self.started_at = time.perf_counter()
        self.sim_time = 0.0
        self.next_snapshot = 10.0
        self.births = {"amoeba": 0, "predator": 0}
        self.deaths = {"amoeba": 0, "predator": 0}
        self.kills = 0
        self.last_settings = None
        self.file = self.path.open("a", encoding="utf-8")

    def write(self, event, **fields):
        record = {
            "event": event,
            "wall_s": round(time.perf_counter() - self.started_at, 3),
            "sim_s": round(self.sim_time, 3),
            **fields,
        }
        self.file.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        self.file.flush()

    def record_settings(self, settings):
        settings = tuple(settings)
        if settings != self.last_settings:
            self.last_settings = settings
            self.write("settings_changed", settings=settings)

    @staticmethod
    def _population_summary(agents):
        by_genome = {}
        for agent in agents:
            key = f"{agent.genome[0]}:{agent.genome[1]}"
            item = by_genome.setdefault(key, {
                "count": 0, "satiety": 0.0, "speed": 0.0,
                "speed_gene": agent.genome[0], "sense_gene": agent.genome[1],
                "q_entries": len(agent.q_table),
            })
            item["count"] += 1
            item["satiety"] += agent.satiety
            item["speed"] += agent.speed
        for item in by_genome.values():
            count = item["count"]
            item["mean_satiety"] = round(item.pop("satiety") / count, 2)
            item["mean_speed"] = round(item.pop("speed") / count, 2)
        return by_genome

    def advance(self, dt, amoebas, predators, eggs, spots, amoeba_steps, predator_steps):
        self.sim_time += dt
        if self.sim_time < self.next_snapshot:
            return
        while self.next_snapshot <= self.sim_time:
            self.next_snapshot += 10.0
        agents = (*amoebas, *predators)
        visit_count = sum(sum(agent.visited_cells.values()) for agent in agents)
        unique_visit_count = sum(len(agent.visited_cells) for agent in agents)
        self.write(
            "snapshot",
            population={"amoeba": len(amoebas), "predator": len(predators)},
            eggs={species: sum(egg.species == species for egg in eggs)
                  for species in ("amoeba", "predator")},
            food_total=round(sum(spot.value for spot in spots), 2),
            route_visits={"total": visit_count, "unique_cells": unique_visit_count,
                          "revisits": max(0, visit_count - unique_visit_count)},
            births=dict(self.births), deaths=dict(self.deaths), kills=self.kills,
            amoeba_genotypes=self._population_summary(amoebas),
            predator_genotypes=self._population_summary(predators),
            q_steps={"amoeba": amoeba_steps, "predator": predator_steps},
        )

    def close(self, status):
        self.write("run_end", status=status, births=self.births,
                   deaths=self.deaths, kills=self.kills)
        self.file.close()

SETTING_NAMES = (
    "Радиус чувствительности к свету",
    "Радиус чувствительности к теплу",
    "Скорость амёбы",
    "Рост сытости на свету",
    "Потеря сытости в темноте",
    "Расход ценности пятна",
    "Яиц амёбы для зачёта обучения",
)
SETTING_DEFAULTS = (100.0, 260.0, 143.0, 14.0, 3.5, 8.0, 2.0)
SETTING_RANGES = ((20, 250, 5), (30, 500, 10), (30, 240, 1), (2, 35, 1), (1, 15, 0.5), (1, 25, 1), (1, 20, 1))


class FoodSpot:
    def __init__(self, position):
        self.position = pygame.Vector2(position)
        self.value = 100.0


@lru_cache(maxsize=32)
def _unoccluded_field_layer(source_x, source_y, reach):
    """Heat fields ignore walls, so reuse their immutable raster across worlds."""
    layer = bytearray(FIELD_WIDTH * FIELD_HEIGHT)
    for gy in range(FIELD_HEIGHT):
        y = gy * FIELD_CELL + FIELD_CELL // 2
        row = gy * FIELD_WIDTH
        for gx in range(FIELD_WIDTH):
            x = gx * FIELD_CELL + FIELD_CELL // 2
            distance = math.hypot(x - source_x, y - source_y)
            if distance < reach:
                layer[row + gx] = round(255 * (1.0 - distance / reach) ** 2)
    return bytes(layer)


class SensorMaps:
    """Precomputed light and heat fields. Light rays stop at walls; heat does not."""

    def __init__(self, walls, food_spots):
        self.walls = walls
        self.light_layers = [self._build_layer(spot.position, LIGHT_FIELD_REACH, True) for spot in food_spots]
        self.heat_layers = [self._build_layer(spot.position, HEAT_FIELD_REACH, False) for spot in food_spots]
        self._render_cache_key = None
        self._render_cache = None

    def _build_layer(self, source, reach, occluded):
        if not occluded:
            return _unoccluded_field_layer(source.x, source.y, reach)
        layer = bytearray(FIELD_WIDTH * FIELD_HEIGHT)
        source_line = (round(source.x), round(source.y))
        for gy in range(FIELD_HEIGHT):
            y = gy * FIELD_CELL + FIELD_CELL // 2
            row = gy * FIELD_WIDTH
            for gx in range(FIELD_WIDTH):
                x = gx * FIELD_CELL + FIELD_CELL // 2
                distance = math.hypot(x - source.x, y - source.y)
                if distance >= reach:
                    continue
                if occluded:
                    target = (x, y)
                    if any(wall.clipline(source_line, target) for wall in self.walls):
                        continue
                strength = (1.0 - distance / reach) ** 2
                layer[row + gx] = round(255 * strength)
        return layer

    @staticmethod
    def _sample(layer, point):
        gx = max(0.0, min(FIELD_WIDTH - 1.0, point.x / FIELD_CELL - 0.5))
        gy = max(0.0, min(FIELD_HEIGHT - 1.0, point.y / FIELD_CELL - 0.5))
        x0, y0 = int(gx), int(gy)
        x1, y1 = min(x0 + 1, FIELD_WIDTH - 1), min(y0 + 1, FIELD_HEIGHT - 1)
        fx, fy = gx - x0, gy - y0
        top_left = layer[y0 * FIELD_WIDTH + x0] / 255.0
        top_right = layer[y0 * FIELD_WIDTH + x1] / 255.0
        bottom_left = layer[y1 * FIELD_WIDTH + x0] / 255.0
        bottom_right = layer[y1 * FIELD_WIDTH + x1] / 255.0
        top = top_left + (top_right - top_left) * fx
        bottom = bottom_left + (bottom_right - bottom_left) * fx
        return top + (bottom - top) * fy

    def light_at(self, point, food_spots):
        return min(1.0, sum(self._sample(layer, point) * spot.value / 100.0
                            for layer, spot in zip(self.light_layers, food_spots)))

    def light_from_spot(self, index, point):
        return self._sample(self.light_layers[index], point)

    def heat_at(self, point, food_spots):
        return max((self._sample(layer, point) * spot.value / 100.0
                    for layer, spot in zip(self.heat_layers, food_spots)), default=0.0)

    def move_source(self, index, position):
        self.light_layers[index] = self._build_layer(position, LIGHT_FIELD_REACH, True)
        self.heat_layers[index] = self._build_layer(position, HEAT_FIELD_REACH, False)
        self._render_cache_key = None

    def render(self, food_spots):
        cache_key = tuple(int(spot.value) for spot in food_spots)
        if cache_key == self._render_cache_key:
            return self._render_cache
        pixels = bytearray(FIELD_WIDTH * FIELD_HEIGHT * 3)
        for index in range(FIELD_WIDTH * FIELD_HEIGHT):
            heat = max((layer[index] * spot.value / 100.0
                        for layer, spot in zip(self.heat_layers, food_spots)), default=0) / 255.0
            light = min(1.0, sum(layer[index] * spot.value / 100.0 / 255.0
                                 for layer, spot in zip(self.light_layers, food_spots)))
            offset = index * 3
            pixels[offset] = min(255, round(18 + 105 * heat + 225 * light))
            pixels[offset + 1] = min(255, round(24 + 35 * heat + 185 * light))
            pixels[offset + 2] = min(255, round(38 + 12 * heat + 65 * light))
        self._render_cache = pygame.transform.scale(
            pygame.image.frombuffer(pixels, (FIELD_WIDTH, FIELD_HEIGHT), "RGB"),
            (WORLD_WIDTH, WORLD_HEIGHT),
        )
        self._render_cache_key = cache_key
        return self._render_cache


class Ameba:
    """Q-learning агент, использующий направления светового и теплового градиентов."""

    def __init__(self):
        self.start = pygame.Vector2(*_cell_center(MAZE_COLS // 2, MAZE_ROWS // 2))
        self.radius = 6
        self.speed = SETTING_DEFAULTS[2]
        self.turn_step = math.pi / 4
        self.q_table = {}
        self.genome = (5, 5)
        self.training_brain_bank = {}
        self.defer_training_save = False
        self.state_size = 8
        self.neighbor_agents = ()
        self.training_state_path = None
        self.shared_training_steps = None
        self.steps = 0
        self.episode = 1
        self.epsilon = START_EPSILON
        self.heading = random.uniform(0, math.tau)
        self.satiety = 100.0
        self.position = self.start.copy()
        self.visited_cells = {}
        self.last_memory_cell = None
        self._record_visit(self.position)
        self.trail = deque([self.position.copy()], maxlen=180)
        self.scent_clock = 0.0
        self.scent_trail = deque(maxlen=180)
        self.scent_trail.append((self.position.copy(), self.scent_clock))
        self._cached_scent_clock = None
        self._cached_scent_sources = ()
        self._wall_grid_source = None
        self._wall_grid = {}
        self.previous_state = None
        self.previous_action = None
        self.pending_reward = 0.0
        self.food_eaten = 0.0
        self.best_food_eaten = 0.0
        self.eggs_laid = 0
        self.total_eggs_laid = 0
        self.best_eggs_laid = 0
        self.training_egg_threshold = int(SETTING_DEFAULTS[6])
        self.accepted_sessions = 0
        self.episode_experiences = []
        self.last_action = None
        self.bumper = False
        self.currently_feeding = False
        self.current_food = None
        self.decision_interval = 0.14
        self.decision_timer = 0.0
        self.dead = False
        self.being_eaten = False
        self.reproduction_allowed = True

    def _wall_ahead(self, walls, point):
        if not ARENA.inflate(-self.radius, -self.radius).collidepoint(point):
            return True
        if self._wall_grid_source is not walls:
            buckets = {}
            max_cell_x = math.ceil(WORLD_WIDTH / CELL_SIZE)
            max_cell_y = math.ceil(WORLD_HEIGHT / CELL_SIZE)
            for wall in walls:
                bounds = wall.inflate(self.radius * 2, self.radius * 2)
                left = max(0, int(bounds.left // CELL_SIZE))
                right = min(max_cell_x, int(bounds.right // CELL_SIZE))
                top = max(0, int(bounds.top // CELL_SIZE))
                bottom = min(max_cell_y, int(bounds.bottom // CELL_SIZE))
                for cell_y in range(top, bottom + 1):
                    for cell_x in range(left, right + 1):
                        buckets.setdefault((cell_x, cell_y), []).append(wall)
            self._wall_grid = buckets
            self._wall_grid_source = walls
        radius_squared = self.radius * self.radius
        cell = (int(point.x // CELL_SIZE), int(point.y // CELL_SIZE))
        for wall in self._wall_grid.get(cell, ()):
            nearest_x = max(wall.left, min(point.x, wall.right))
            nearest_y = max(wall.top, min(point.y, wall.bottom))
            if (point.x - nearest_x) ** 2 + (point.y - nearest_y) ** 2 < radius_squared:
                return True
        return False

    def _gradient_state(self, field_value, sensor_radius):
        forward = pygame.Vector2(math.cos(self.heading), math.sin(self.heading))
        right = pygame.Vector2(-forward.y, forward.x)
        forward_change = field_value(self.position + forward * sensor_radius) - field_value(
            self.position - forward * sensor_radius
        )
        right_change = field_value(self.position + right * sensor_radius) - field_value(
            self.position - right * sensor_radius
        )
        magnitude = math.hypot(forward_change, right_change)
        if magnitude < 0.002:
            direction = 4
        else:
            angle = math.atan2(right_change, forward_change)
            if -math.pi / 4 <= angle <= math.pi / 4:
                direction = 1
            elif math.pi / 4 < angle < 3 * math.pi / 4:
                direction = 2
            elif -3 * math.pi / 4 < angle < -math.pi / 4:
                direction = 0
            else:
                direction = 3
        strength = int(magnitude >= 0.05)
        return direction, strength

    def sensors(self, walls, maps, spots, light_radius, heat_radius):
        vision_scale = self._vision_scale()
        light_radius *= vision_scale
        heat_radius *= vision_scale
        light_field = lambda point: maps.light_at(point, spots)
        heat_field = lambda point: maps.heat_at(point, spots)
        light_direction, light_gradient = self._gradient_state(light_field, light_radius)
        heat_direction, heat_gradient = self._gradient_state(heat_field, heat_radius)
        forward = pygame.Vector2(math.cos(self.heading), math.sin(self.heading))
        bumper = int(self._wall_ahead(walls, self.position + forward * (self.radius * 2 + 8)))
        light_level = min(2, int(light_field(self.position) * 3))
        heat_level = min(2, int(heat_field(self.position) * 3))
        satiety_level = min(3, int(max(0.0, self.satiety) / 33))
        return (light_direction, heat_direction, bumper, light_gradient, heat_gradient,
                light_level, heat_level, satiety_level)

    def _neighbor_scent_at(self, point):
        intensity = 0.0
        smell_scale = self._smell_scale()
        reach = 320.0 * smell_scale
        reach_squared = reach * reach
        for neighbor in self.neighbor_agents:
            if neighbor is self or neighbor.dead:
                continue
            if neighbor._cached_scent_clock != neighbor.scent_clock:
                sources = []
                for index, (trail_position, deposited_at) in enumerate(
                    reversed(neighbor.scent_trail)
                ):
                    if index >= 60:
                        break
                    if index % 3:
                        continue
                    age = neighbor.scent_clock - deposited_at
                    if age > 48.0:
                        break
                    sources.append((trail_position, age))
                neighbor._cached_scent_sources = tuple(sources)
                neighbor._cached_scent_clock = neighbor.scent_clock
            for trail_position, age in neighbor._cached_scent_sources:
                distance_squared = point.distance_squared_to(trail_position)
                if distance_squared > reach_squared:
                    continue
                distance = math.sqrt(distance_squared)
                scent = smell_scale * math.exp(
                    -age / 16.0 - distance / (140.0 * smell_scale)
                )
                intensity = max(intensity, scent)
        return intensity

    def _own_trail_scent_at(self, point):
        intensity = 0.0
        for age_steps, trail_position in enumerate(reversed(self.trail)):
            if age_steps < 3:
                continue
            if age_steps >= 60:
                break
            distance = point.distance_to(trail_position)
            if distance <= 220.0:
                age = age_steps * self.decision_interval
                intensity = max(intensity, math.exp(-age / 14.0 - distance / 90.0))
        return intensity

    def _food_at_position(self, spots, maps):
        choices = []
        for index, spot in enumerate(spots):
            if spot.value <= 0:
                continue
            if maps.light_from_spot(index, self.position) >= FOOD_LIGHT_THRESHOLD:
                choices.append((self.position.distance_squared_to(spot.position), index, spot))
        return min(choices, key=lambda item: item[0])[2] if choices else None

    def _learning_potential(self, maps, spots, light_radius, heat_radius, position=None, heading=None):
        position = self.position if position is None else position
        heading = self.heading if heading is None else heading
        forward = pygame.Vector2(math.cos(heading), math.sin(heading))
        vision_scale = self._vision_scale()
        light_reading = maps.light_at(position + forward * light_radius * vision_scale, spots)
        heat_reading = maps.heat_at(position + forward * heat_radius * vision_scale, spots)
        return 25.0 * light_reading + 30.0 * heat_reading

    def _vision_scale(self):
        return 0.75 + 0.05 * self.genome[1]

    def _smell_scale(self):
        return 1.25 - 0.05 * self.genome[1]

    @staticmethod
    def _genome_key(genome):
        return f"{genome[0]}:{genome[1]}"

    def bind_genome(self, genome, inherited_q_table=None):
        self.genome = tuple(max(0, min(GENE_LEVELS - 1, int(gene))) for gene in genome)
        key = self._genome_key(self.genome)
        if key not in self.training_brain_bank:
            self.training_brain_bank[key] = (
                copy.deepcopy(inherited_q_table) if inherited_q_table is not None else {}
            )
        self.q_table = self.training_brain_bank[key]

    def _path_clear(self, walls, direction, distance):
        substeps = max(1, math.ceil(distance / 4.0))
        step = direction * (distance / substeps)
        point = self.position.copy()
        for _ in range(substeps):
            point += step
            if self._wall_ahead(walls, point):
                return False
        return True

    def _move_forward(self, walls, direction, distance):
        substeps = max(1, math.ceil(distance / 4.0))
        step = direction * (distance / substeps)
        moved = False
        self.bumper = False
        for _ in range(substeps):
            candidate = self.position + step
            if self._wall_ahead(walls, candidate):
                self.bumper = True
                break
            self.position = candidate
            moved = True
        if moved:
            self._record_visit(self.position)
        return moved

    def _cell_for_position(self, position):
        cell_x = max(0, min(MAZE_COLS - 1, int((position.x - MAZE_OFFSET_X) // CELL_SIZE)))
        cell_y = max(0, min(MAZE_ROWS - 1, int((position.y - MAZE_OFFSET_Y) // CELL_SIZE)))
        return cell_x, cell_y

    def _record_visit(self, position):
        cell = self._cell_for_position(position)
        if cell != self.last_memory_cell:
            self.visited_cells[cell] = self.visited_cells.get(cell, 0) + 1
            self.last_memory_cell = cell

    def _lookahead_cell(self, walls, direction, distance):
        substeps = max(1, math.ceil(distance / 4.0))
        step = direction * (distance / substeps)
        point = self.position.copy()
        previous_cell = self._cell_for_position(point)
        furthest_cell = previous_cell
        for _ in range(substeps):
            candidate = point + step
            if self._wall_ahead(walls, candidate):
                break
            point = candidate
            cell = self._cell_for_position(point)
            if cell != previous_cell:
                return cell
        return furthest_cell

    def _escape_wall(self, walls, spots, maps, light_radius, heat_radius, distance):
        options = []
        current_potential = self._learning_potential(maps, spots, light_radius, heat_radius)
        reverse_direction = -(
            self.position - self.trail[-2]
        ) if len(self.trail) > 1 else pygame.Vector2()
        has_reverse = reverse_direction.length_squared() > 1.0
        if has_reverse:
            reverse_direction.normalize_ip()

        for offset in (0, -1, 1, -2, 2, -3, 3, 4):
            heading = self.heading + offset * self.turn_step
            direction = pygame.Vector2(math.cos(heading), math.sin(heading))
            if not self._path_clear(walls, direction, distance):
                continue
            is_backtrack = has_reverse and direction.dot(reverse_direction) >= math.cos(self.turn_step)
            if is_backtrack:
                continue
            destination = self.position + direction * distance
            cell = self._lookahead_cell(walls, direction, max(CELL_SIZE * 5, distance * 3))
            visits = self.visited_cells.get(cell, 0)
            potential = self._learning_potential(
                maps, spots, light_radius, heat_radius, destination, heading,
            )
            field_guidance = max(-1.0, min(1.0, (potential - current_potential) * 1.5))
            revisit_penalty = min(visits, 6) * 0.22
            options.append((field_guidance - revisit_penalty - abs(offset) * 0.04,
                            -visits, -abs(offset), heading))

        # The route back is an emergency exit only when every other safe
        # heading is blocked (or would immediately retrace the blue trail).
        if not options:
            for offset in (4, -3, 3, -2, 2, -1, 1, 0):
                heading = self.heading + offset * self.turn_step
                direction = pygame.Vector2(math.cos(heading), math.sin(heading))
                if self._path_clear(walls, direction, distance):
                    options.append((
                        self._learning_potential(
                            maps, spots, light_radius, heat_radius,
                            self.position + direction * distance, heading,
                        ), -abs(offset), 0, heading,
                    ))
                    break
        if not options:
            self.bumper = True
            return False
        self.heading = max(options)[-1]
        moved = self._move_forward(
            walls, pygame.Vector2(math.cos(self.heading), math.sin(self.heading)), distance,
        )
        self.bumper = True
        return moved

    def _values(self, state):
        return self.q_table.setdefault(state, [0.0, 0.0, 0.0])

    def choose_action(self, state, walls, spots, maps, light_radius, heat_radius):
        values = self.q_table.get(state, (0.0, 0.0, 0.0))
        q_scale = max(1.0, max(abs(value) for value in values))
        current_potential = self._learning_potential(maps, spots, light_radius, heat_radius)
        distance = self.speed * self.decision_interval * 1.5
        action_offsets = (0, -1, 1)
        scores = []
        exploration_scores = []
        options = []
        last_motion = self.position - self.trail[-2] if len(self.trail) > 1 else pygame.Vector2()
        has_momentum = last_motion.length_squared() > 1.0
        if has_momentum:
            last_motion.normalize_ip()
        scent_here = self._neighbor_scent_at(self.position)
        own_trail_here = self._own_trail_scent_at(self.position)
        for action, offset in enumerate(action_offsets):
            candidate_heading = self.heading + offset * self.turn_step
            direction = pygame.Vector2(math.cos(candidate_heading), math.sin(candidate_heading))
            if not self._path_clear(walls, direction, distance):
                continue
            destination = self.position + direction * distance
            potential = self._learning_potential(
                maps, spots, light_radius, heat_radius, destination, candidate_heading
            )
            field_guidance = max(-1.0, min(1.0, (potential - current_potential) * 1.5))
            scent_avoidance = (scent_here - self._neighbor_scent_at(destination)) * 8.0
            scent_avoidance += (
                own_trail_here - self._own_trail_scent_at(destination)
            ) * 4.0
            route_cell = self._lookahead_cell(walls, direction, max(CELL_SIZE * 5, distance * 3))
            visits = self.visited_cells.get(route_cell, 0)
            revisit_penalty = min(visits, 6) * self._exploration_revisit_weight()
            continuation_bonus = direction.dot(last_motion) if has_momentum else 0.0
            turn_penalty = 0.08 * abs(offset)
            reverses_turn = self.last_action in (1, 2) and action in (1, 2) and action != self.last_action
            turn_switch_penalty = 0.35 if reverses_turn else 0.0
            route_score = (field_guidance + scent_avoidance - revisit_penalty + continuation_bonus
                           - turn_penalty - turn_switch_penalty)
            options.append((visits, values[action] / q_scale + route_score, route_score, action))
        if not options:
            return random.randrange(3)
        # Prefer routes that reach an unseen cell; revisit old cells only when
        # every safe heading leads through territory already mapped this life.
        fresh_options = [option for option in options if option[0] == 0]
        if fresh_options:
            best_route_score = max(option[2] for option in options)
            worthwhile_fresh = [option for option in fresh_options
                                if option[2] >= best_route_score - 0.2]
            if worthwhile_fresh:
                options = worthwhile_fresh
        scores = [(score, action) for _, score, _, action in options]
        exploration_scores = [(route_score, action) for _, _, route_score, action in options]
        if random.random() < self.epsilon:
            best_route_score = max(score for score, _ in exploration_scores)
            exploratory_actions = [action for score, action in exploration_scores
                                   if score >= best_route_score - 0.25]
            return random.choice(exploratory_actions)
        best_score = max(score for score, _ in scores)
        return random.choice([action for score, action in scores if score == best_score])

    def learn(self, state, action, reward, next_state, terminal=False, learning_weight=1):
        values = self._values(state)
        next_best = 0.0 if terminal else max(self._values(next_state))
        if learning_weight >= 10_000:
            learning_rate = 1.0
        else:
            learning_rate = -math.expm1(learning_weight * math.log1p(-ALPHA))
        values[action] += learning_rate * (reward + GAMMA * next_best - values[action])
        self.epsilon = max(MIN_EPSILON, START_EPSILON / (1 + EPSILON_DECAY * self.steps))

    def _record_decision(self):
        current_steps = (self.shared_training_steps[0]
                         if self.shared_training_steps is not None else self.steps)
        self.steps = current_steps + 1
        if self.shared_training_steps is not None:
            self.shared_training_steps[0] = self.steps
        self.epsilon = max(MIN_EPSILON, START_EPSILON / (1 + EPSILON_DECAY * self.steps))

    def _sync_epsilon(self):
        if self.shared_training_steps is not None:
            self.steps = self.shared_training_steps[0]
            self.epsilon = max(MIN_EPSILON, START_EPSILON / (1 + EPSILON_DECAY * self.steps))

    def load_training_state(self, path):
        path = Path(path)
        self.training_state_path = path
        self.bind_genome(self.genome)
        if not path.exists():
            return False
        try:
            with path.open("r", encoding="utf-8") as state_file:
                saved = json.load(state_file)
            if saved.get("format") == Q_TABLE_FORMAT_VERSION:
                for key, rows in saved.get("genomes", {}).items():
                    table = {
                        tuple(int(value) for value in state): [float(value) for value in values]
                        for state, values in rows
                        if len(state) == self.state_size and len(values) == 3
                    }
                    self.training_brain_bank[key] = table
            elif saved.get("format") == 2:
                table = {
                    tuple(int(value) for value in state): [float(value) for value in values]
                    for state, values in saved.get("q_table", [])
                    if len(state) == self.state_size and len(values) == 3
                }
                self.training_brain_bank[self._genome_key(self.genome)] = table
            else:
                return False
            self.bind_genome(self.genome)
            self.steps = max(0, int(saved.get("steps", 0)))
            if self.shared_training_steps is not None:
                self.shared_training_steps[0] = self.steps
            self.epsilon = max(MIN_EPSILON, START_EPSILON / (1 + EPSILON_DECAY * self.steps))
            return True
        except (OSError, ValueError, TypeError, KeyError) as error:
            print(f"Не удалось загрузить Q-таблицу из {path}: {error}")
            return False

    def save_training_state(self):
        if self.training_state_path is None:
            return
        path = self.training_state_path
        temporary_path = path.with_suffix(path.suffix + ".tmp")
        saved = {
            "format": Q_TABLE_FORMAT_VERSION,
            "steps": (self.shared_training_steps[0] if self.shared_training_steps is not None
                      else self.steps),
            "genomes": {
                key: [[list(state), values] for state, values in table.items()]
                for key, table in self.training_brain_bank.items()
            },
        }
        try:
            with temporary_path.open("w", encoding="utf-8") as state_file:
                json.dump(saved, state_file, ensure_ascii=False, separators=(",", ":"))
            temporary_path.replace(path)
        except OSError as error:
            print(f"Не удалось сохранить Q-таблицу в {path}: {error}")

    def finish_learning_session(self):
        session_value = SESSION_VALUE_GROWTH ** self.eggs_laid
        minimum_value = SESSION_VALUE_GROWTH ** self.training_egg_threshold
        if self.eggs_laid >= self.training_egg_threshold:
            learning_weight = session_value / minimum_value
            for experience in reversed(self.episode_experiences):
                self.learn(*experience, learning_weight=learning_weight)
            self.accepted_sessions += 1
        self.episode_experiences.clear()
        if not self.defer_training_save:
            self.save_training_state()
        self.best_food_eaten = max(self.best_food_eaten, self.food_eaten)
        self.best_eggs_laid = max(self.best_eggs_laid, self.eggs_laid)

    def mark_dead(self, terminal_reward=-100.0):
        if self.previous_state is not None:
            self.episode_experiences.append((
                self.previous_state, self.previous_action,
                self.pending_reward + terminal_reward, None, True,
            ))
        self.finish_learning_session()
        self.dead = True
        self.being_eaten = False

    def reset_episode(self):
        self.finish_learning_session()
        self.episode += 1
        self.position.update(self.start)
        self.visited_cells.clear()
        self.last_memory_cell = None
        self._record_visit(self.position)
        self.trail.clear()
        self.trail.append(self.position.copy())
        self.heading = random.uniform(0, math.tau)
        self.satiety = 100.0
        self.previous_state = None
        self.previous_action = None
        self.pending_reward = 0.0
        self.food_eaten = 0.0
        self.eggs_laid = 0
        self.last_action = None
        self.bumper = False
        self.currently_feeding = False
        self.current_food = None
        self.decision_timer = 0.0
        self.dead = False

    def _gain_satiety(self, amount):
        self.satiety = min(MAX_SATIETY, self.satiety + amount)
        if self.satiety >= EGG_SATIETY_THRESHOLD and self.reproduction_allowed:
            self.eggs_laid += 1
            self.total_eggs_laid += 1
            self.pending_reward += EGG_REWARD
            self.satiety = 100.0
            return True
        return False

    def set_start(self, position):
        self.start.update(position)
        self.position.update(position)
        self.visited_cells.clear()
        self.last_memory_cell = None
        self._record_visit(self.position)
        self.trail.clear()
        self.trail.append(self.position.copy())
        self.scent_clock = 0.0
        self.scent_trail.clear()
        self.scent_trail.append((self.position.copy(), self.scent_clock))
        self.being_eaten = False

    def _movement_speed(self, base_speed):
        if self.satiety > 70.0:
            return base_speed * self._speed_scale()
        satiety_ratio = max(0.0, self.satiety) / 70.0
        return (base_speed * self._speed_scale() * PREDATOR_BASE_SPEED_RATIO
                * (0.5 + 1.5 * satiety_ratio))

    def _exploration_revisit_weight(self):
        return 0.22

    def _speed_scale(self):
        return 0.8 + 0.04 * self.genome[0]

    def update(self, dt, walls, spots, maps, settings, auto_respawn=True):
        if self.dead or self.being_eaten:
            return
        self.scent_clock += dt
        light_radius, heat_radius, base_speed, satiety_gain, dark_loss, depletion_rate = settings[:6]
        self.speed = self._movement_speed(base_speed)
        self.training_egg_threshold = int(settings[6])
        self.current_food = self._food_at_position(spots, maps)
        self.currently_feeding = self.current_food is not None
        if self.current_food:
            self._gain_satiety(satiety_gain * dt)
            spot_index = spots.index(self.current_food)
            consumed = min(self.current_food.value, depletion_rate * dt)
            self.current_food.value -= consumed
            self.food_eaten += consumed
            self.pending_reward += consumed
            if self.current_food.value <= 0:
                available = [_cell_center(x, y)
                             for y in range(MAZE_ROWS)
                             for x in range(MAZE_COLS)]
                occupied = [spot.position for i, spot in enumerate(spots) if i != spot_index]
                candidates = [pygame.Vector2(point) for point in available
                              if all(pygame.Vector2(point).distance_to(other) >= FOOD_RADIUS * 2
                                     for other in occupied)
                              and pygame.Vector2(point).distance_to(self.position) >= CELL_SIZE * 2]
                if not candidates:
                    candidates = [pygame.Vector2(point) for point in available
                                  if all(pygame.Vector2(point).distance_to(other) >= FOOD_RADIUS * 2
                                         for other in occupied)]
                self.current_food.position.update(random.choice(candidates))
                self.current_food.value = 100.0
                maps.move_source(spot_index, self.current_food.position)
                self.current_food = None
                self.currently_feeding = False
        else:
            self.satiety -= dark_loss * dt
            self.pending_reward -= dark_loss * dt * 0.1

        if self.satiety <= 0:
            if auto_respawn:
                if self.previous_state is not None:
                    self.episode_experiences.append((
                        self.previous_state, self.previous_action,
                        self.pending_reward - 100.0, None, True,
                    ))
                self.reset_episode()
            else:
                self.mark_dead(-100.0)
            return

        # Stay on a food patch long enough to consume it before searching again.
        if self.currently_feeding:
            return

        self.decision_timer += dt
        if self.decision_timer < self.decision_interval:
            return
        self.decision_timer -= self.decision_interval

        state = self.sensors(walls, maps, spots, light_radius, heat_radius)
        self._sync_epsilon()
        self.bumper = bool(state[2])
        old_potential = self._learning_potential(maps, spots, light_radius, heat_radius)
        if self.previous_state is not None:
            self.episode_experiences.append((
                self.previous_state, self.previous_action, self.pending_reward, state, False,
            ))
        self.pending_reward = 0.0
        action = self.choose_action(state, walls, spots, maps, light_radius, heat_radius)
        self._record_decision()
        if action == 1:
            self.heading -= self.turn_step
        elif action == 2:
            self.heading += self.turn_step

        distance = self.speed * self.decision_interval * 1.5
        old_position = self.position.copy()
        direction = pygame.Vector2(math.cos(self.heading), math.sin(self.heading))
        moved = self._move_forward(walls, direction, distance)
        hit_wall = self.bumper
        if hit_wall:
            if moved:
                self.trail.append(self.position.copy())
            escaped = self._escape_wall(
                walls, spots, maps, light_radius, heat_radius, distance,
            )
            if escaped:
                moved = True
                self.trail.append(self.position.copy())
                self.scent_trail.append((self.position.copy(), self.scent_clock))
            self.bumper = hit_wall
            self.pending_reward -= 10.0
        elif moved:
            self.trail.append(self.position.copy())
            self.scent_trail.append((self.position.copy(), self.scent_clock))
        actual_distance = old_position.distance_to(self.position)
        self.satiety -= actual_distance * MOVE_ENERGY_PER_PIXEL * max(
            0.0, self._speed_scale() - 1.0,
        )
        if self.satiety <= 0:
            self.mark_dead(-100.0)
            return
        # Small gradient feedback guides exploration; eating remains the main reward.
        shaping = self._learning_potential(maps, spots, light_radius, heat_radius) - old_potential
        self.pending_reward += max(-2.0, min(2.0, shaping))
        self.previous_state = state
        self.previous_action = action
        self.last_action = action


class Predator(Ameba):
    """Хищник учится на тех же датчиках, дополненных запахом добычи."""

    def __init__(self):
        super().__init__()
        self.state_size = 11
        self.training_egg_threshold = 1
        self.kills_since_egg = 0
        self.total_kills = 0
        self.prey_agents = []
        self.hatch_eggs_pending = 0
        self.eating_prey = None
        self.eating_timer = 0.0
        self.scent_walls = []
        self.scent_blockers = ()
        self._scent_cache = {}

    def _scent_at(self, point):
        cache_key = (point.x, point.y, id(self.scent_blockers))
        if cache_key in self._scent_cache:
            return self._scent_cache[cache_key]
        living_prey = [prey for prey in self.prey_agents
                       if not prey.dead and not prey.being_eaten]
        intensity = 0.0
        for prey in living_prey:
            delta = prey.position - point
            distance = delta.length()
            smell_reach = AMOEBA_SCENT_REACH * self._smell_scale()
            if distance > smell_reach:
                continue
            if distance > 0 and self.scent_walls:
                segment_start = (round(point.x), round(point.y))
                segment_end = (round(prey.position.x), round(prey.position.y))
                segment_bounds = pygame.Rect(
                    min(segment_start[0], segment_end[0]),
                    min(segment_start[1], segment_end[1]),
                    max(1, abs(segment_end[0] - segment_start[0])),
                    max(1, abs(segment_end[1] - segment_start[1])),
                )
                candidate_walls = segment_bounds.collidelistall(self.scent_blockers)
                blocked = any(
                    self.scent_blockers[index].clipline(segment_start, segment_end)
                    for index in candidate_walls
                )
                if blocked:
                    continue
            intensity = max(intensity, self._smell_scale()
                            * math.exp(-3.0 * distance / smell_reach))
        self._scent_cache[cache_key] = intensity
        return intensity

    def _hunger_level(self):
        return max(0.0, min(1.0, 1.0 - self.satiety / 100.0))

    def _aggression_level(self):
        return 1.0 + 0.5 * self._hunger_level()

    def _exploration_revisit_weight(self):
        return 0.22 + 0.45 * self._hunger_level()

    def choose_action(self, state, walls, spots, maps, light_radius, heat_radius):
        scent_here = self._scent_at(self.position)
        own_trail_here = self._own_trail_scent_at(self.position)
        if scent_here > 0.01:
            # Sample the odor field around the predator to estimate a stable
            # bearing. Comparing only the three immediate actions can make it
            # repeatedly turn toward tiny scent fluctuations and trace circles.
            scent_gradient = pygame.Vector2()
            sensor_radius = max(48.0, self.radius * 6.0)
            for sample in range(8):
                angle = sample * math.tau / 8
                direction = pygame.Vector2(math.cos(angle), math.sin(angle))
                scent_gradient += direction * self._scent_at(
                    self.position + direction * sensor_radius
                )
            # Aim ahead of moving prey so the predator intercepts it instead
            # of continually following its previous position.
            for prey in self.prey_agents:
                if prey.dead or prey.being_eaten or len(prey.trail) < 2:
                    continue
                motion = prey.trail[-1] - prey.trail[-2]
                if motion.length_squared() <= 1.0:
                    continue
                distance = self.position.distance_to(prey.position)
                lead = min(120.0, distance * 0.35)
                predicted_position = prey.position + motion.normalize() * lead
                intercept_direction = predicted_position - self.position
                if intercept_direction.length_squared() > 1.0:
                    scent_gradient += (
                        intercept_direction.normalize() * 3.5 * self._aggression_level()
                    )
            if scent_gradient.length_squared() > 1e-8:
                scent_direction = scent_gradient.normalize()
                values = self.q_table.get(state, (0.0, 0.0, 0.0))
                q_scale = max(1.0, max(abs(value) for value in values))
                distance = self.speed * self.decision_interval * 1.5
                options = []
                for action, offset in enumerate((0, -1, 1)):
                    heading = self.heading + offset * self.turn_step
                    direction = pygame.Vector2(math.cos(heading), math.sin(heading))
                    if not self._path_clear(walls, direction, distance):
                        continue
                    route_cell = self._lookahead_cell(
                        walls, direction, max(CELL_SIZE * 5, distance * 3),
                    )
                    visits = self.visited_cells.get(route_cell, 0)
                    scent_alignment = direction.dot(scent_direction)
                    exploration_drive = self._hunger_level()
                    revisit_penalty = min(visits, 6) * (1.0 + 1.5 * exploration_drive)
                    novelty_bonus = 0.75 * exploration_drive if visits == 0 else 0.0
                    own_trail_avoidance = (
                        own_trail_here - self._own_trail_scent_at(
                            self.position + direction * distance,
                        )
                    ) * 4.0
                    score = (3.5 * self._aggression_level() * scent_alignment
                             + own_trail_avoidance + novelty_bonus
                             - revisit_penalty
                             + 0.2 * values[action] / q_scale - 0.08 * abs(offset))
                    options.append((score, action))
                if options:
                    best_score = max(score for score, _ in options)
                    return random.choice([action for score, action in options
                                          if score == best_score])
        return super().choose_action(state, walls, spots, maps, light_radius, heat_radius)

    def sensors(self, walls, maps, spots, light_radius, heat_radius, prey_agents=None):
        if prey_agents is not None:
            self.prey_agents = prey_agents
        base_state = super().sensors(walls, maps, spots, light_radius, heat_radius)
        scent_field = self._scent_at
        scent_direction, scent_gradient = self._gradient_state(
            scent_field, AMOEBA_SCENT_REACH * 0.2,
        )
        scent_level = min(2, int(scent_field(self.position) * 3))
        return (*base_state, scent_direction, scent_gradient, scent_level)

    def _learning_potential(self, maps, spots, light_radius, heat_radius, position=None, heading=None):
        base_potential = super()._learning_potential(
            maps, spots, light_radius, heat_radius, position, heading,
        )
        position = self.position if position is None else position
        return base_potential + 240.0 * self._scent_at(position)

    def update(self, dt, walls, spots, maps, settings, prey_agents):
        light_radius, heat_radius, base_speed, _, dark_loss, _ = settings[:6]
        self.speed = base_speed * PREDATOR_BASE_SPEED_RATIO * self._speed_scale()
        self.prey_agents = prey_agents
        self._scent_cache.clear()
        if self.scent_walls is not walls:
            self.scent_walls = walls
            self.scent_blockers = tuple(
                wall.inflate(self.radius * 2, self.radius * 2) for wall in walls
            )
        self.satiety -= dark_loss * PREDATOR_HUNGER_DRAIN_RATIO * dt
        if self.satiety <= 0:
            if self.eating_prey is not None:
                self.eating_prey.being_eaten = False
                self.eating_prey = None
            self.mark_dead(-100.0)
            return False

        if self.eating_prey is not None:
            if self.eating_prey.dead:
                self.eating_prey.being_eaten = False
                self.eating_prey = None
                self.eating_timer = 0.0
            else:
                self.eating_timer -= dt
                if self.eating_timer > 0.0:
                    return False
                prey = self.eating_prey
                self.eating_prey = None
                self.eating_timer = 0.0
                prey.mark_dead(-100.0)
                self.satiety = 100.0
                self.kills_since_egg += 1
                self.total_kills += 1
                self.pending_reward += EGG_REWARD

        self.decision_timer += dt
        if self.decision_timer < self.decision_interval:
            return False
        self.decision_timer -= self.decision_interval

        state = self.sensors(walls, maps, spots, light_radius, heat_radius)
        if self.previous_state is not None:
            self.episode_experiences.append((
                self.previous_state, self.previous_action, self.pending_reward, state, False,
            ))
        self.pending_reward = 0.0
        self._sync_epsilon()
        action = self.choose_action(state, walls, spots, maps, light_radius, heat_radius)
        self._record_decision()
        if action == 1:
            self.heading -= self.turn_step
        elif action == 2:
            self.heading += self.turn_step

        distance = self.speed * self.decision_interval * 1.5
        old_position = self.position.copy()
        direction = pygame.Vector2(math.cos(self.heading), math.sin(self.heading))
        moved = self._move_forward(walls, direction, distance)
        hit_wall = self.bumper
        if hit_wall:
            if moved:
                self.trail.append(self.position.copy())
            escaped = self._escape_wall(
                walls, spots, maps, light_radius, heat_radius, distance,
            )
            if escaped:
                moved = True
                self.trail.append(self.position.copy())
            self.pending_reward -= 10.0
        elif moved:
            self.trail.append(self.position.copy())

        actual_distance = old_position.distance_to(self.position)
        self.satiety -= actual_distance * MOVE_ENERGY_PER_PIXEL * max(
            0.0, self._speed_scale() - 1.0,
        )
        if self.satiety <= 0:
            self.mark_dead(-100.0)
            return False

        prey = min((agent for agent in prey_agents
                    if not agent.dead and not agent.being_eaten),
                   key=lambda agent: self.position.distance_squared_to(agent.position), default=None)
        if prey is not None:
            movement = self.position - old_position
            movement_length_squared = movement.length_squared()
            projection = 0.0
            if movement_length_squared > 0.0:
                projection = max(0.0, min(1.0,
                    (prey.position - old_position).dot(movement) / movement_length_squared
                ))
            closest_point = old_position + movement * projection
            contact_distance = self.radius + prey.radius
            if closest_point.distance_squared_to(prey.position) <= contact_distance ** 2:
                self.position.update(closest_point)
                self.trail.append(self.position.copy())
                self.eating_prey = prey
                self.eating_prey.being_eaten = True
                self.eating_timer = PREDATOR_EAT_SECONDS

        if self.kills_since_egg >= 3 and self.reproduction_allowed:
            self.eggs_laid += 1
            self.total_eggs_laid += 1
            self.hatch_eggs_pending += 1
            self.pending_reward += EGG_REWARD
            self.episode_experiences.append((state, action, self.pending_reward, None, True))
            self.finish_learning_session()
            self.eggs_laid = 0
            self.kills_since_egg = 0
            self.episode += 1
            self.previous_state = None
            self.previous_action = None
            self.pending_reward = 0.0
        else:
            self.previous_state = state
            self.previous_action = action
            self.last_action = action
        return False


class Egg:
    def __init__(self, species, position, genome=(5, 5),
                 inherited_q_table=None, brain_bank=None, parent_genome=None):
        self.species = species
        self.position = pygame.Vector2(position)
        self.remaining = EGG_HATCH_SECONDS
        self.genome = tuple(genome)
        self.inherited_q_table = inherited_q_table
        self.brain_bank = brain_bank
        self.parent_genome = tuple(parent_genome) if parent_genome is not None else None

    @classmethod
    def from_parent(cls, species, position, parent):
        genome = list(parent.genome)
        locus = random.randrange(len(genome))
        direction = random.choice((-1, 1))
        if genome[locus] == 0:
            direction = 1
        elif genome[locus] == GENE_LEVELS - 1:
            direction = -1
        genome[locus] += direction
        return cls(species, position, genome, copy.deepcopy(parent.q_table),
                   parent.training_brain_bank, parent.genome)


def _hatch_agent(egg, amoeba_brain, predator_brain, amoeba_steps, predator_steps,
                 amoeba_state_path, predator_state_path,
                 amoeba_brain_bank=None, predator_brain_bank=None):
    agent = Ameba() if egg.species == "amoeba" else Predator()
    if egg.species == "amoeba":
        fallback_brain = amoeba_brain
        agent.shared_training_steps = amoeba_steps
        agent.training_state_path = amoeba_state_path
    else:
        fallback_brain = predator_brain
        agent.shared_training_steps = predator_steps
        agent.training_state_path = predator_state_path
    inherited_bank = egg.brain_bank
    if inherited_bank is None:
        inherited_bank = amoeba_brain_bank if egg.species == "amoeba" else predator_brain_bank
    if inherited_bank is not None:
        agent.training_brain_bank = inherited_bank
    else:
        agent.training_brain_bank = {agent._genome_key((5, 5)): fallback_brain}
    agent.bind_genome(egg.genome, egg.inherited_q_table or fallback_brain)
    agent.set_start(egg.position)
    return agent


def _carve_maze(cols=MAZE_COLS, rows=MAZE_ROWS, required_cells=()):
    walls = [[[True, True, True, True] for _ in range(cols)] for _ in range(rows)]
    visited = {(0, 0)}
    stack = [(0, 0)]
    directions = ((0, -1, 0, 2), (1, 0, 1, 3), (0, 1, 2, 0), (-1, 0, 3, 1))
    while stack:
        x, y = stack[-1]
        neighbors = [
            (x + dx, y + dy, wall, opposite)
            for dx, dy, wall, opposite in directions
            if 0 <= x + dx < cols and 0 <= y + dy < rows and (x + dx, y + dy) not in visited
        ]
        if not neighbors:
            stack.pop()
            continue
        nx, ny, wall, opposite = random.choice(neighbors)
        walls[y][x][wall] = False
        walls[ny][nx][opposite] = False
        visited.add((nx, ny))
        stack.append((nx, ny))

    start_cell = (cols // 2, rows // 2)
    while True:
        reachable = {start_cell}
        queue = deque([start_cell])
        while queue:
            x, y = queue.popleft()
            for dx, dy, wall, _ in directions:
                neighbor = (x + dx, y + dy)
                if (0 <= neighbor[0] < cols and 0 <= neighbor[1] < rows
                        and not walls[y][x][wall] and neighbor not in reachable):
                    reachable.add(neighbor)
                    queue.append(neighbor)
        if all(cell in reachable for cell in required_cells):
            break
        walls = [[[True, True, True, True] for _ in range(cols)] for _ in range(rows)]
        visited = {(0, 0)}
        stack = [(0, 0)]
        while stack:
            x, y = stack[-1]
            neighbors = [
                (x + dx, y + dy, wall, opposite)
                for dx, dy, wall, opposite in directions
                if 0 <= x + dx < cols and 0 <= y + dy < rows and (x + dx, y + dy) not in visited
            ]
            if not neighbors:
                stack.pop()
                continue
            nx, ny, wall, opposite = random.choice(neighbors)
            walls[y][x][wall] = False
            walls[ny][nx][opposite] = False
            visited.add((nx, ny))
            stack.append((nx, ny))

    horizontal = [[False] * cols for _ in range(rows + 1)]
    vertical = [[False] * (cols + 1) for _ in range(rows)]
    for y in range(rows):
        for x in range(cols):
            north, east, south, west = walls[y][x]
            horizontal[y][x] |= north
            horizontal[y + 1][x] |= south
            vertical[y][x] |= west
            vertical[y][x + 1] |= east

    thickness = 4
    merged = []
    for y, edge_row in enumerate(horizontal):
        x = 0
        while x < cols:
            if not edge_row[x]:
                x += 1
                continue
            start = x
            while x < cols and edge_row[x]:
                x += 1
            merged.append(pygame.Rect(MAZE_OFFSET_X + start * CELL_SIZE,
                                      MAZE_OFFSET_Y + y * CELL_SIZE - thickness // 2,
                                      (x - start) * CELL_SIZE, thickness))
    for x in range(cols + 1):
        row_index = 0
        while row_index < rows:
            if not vertical[row_index][x]:
                row_index += 1
                continue
            start = row_index
            while row_index < rows and vertical[row_index][x]:
                row_index += 1
            merged.append(pygame.Rect(MAZE_OFFSET_X + x * CELL_SIZE - thickness // 2,
                                      MAZE_OFFSET_Y + start * CELL_SIZE,
                                      thickness, (row_index - start) * CELL_SIZE))
    return merged


def _cell_center(cell_x, cell_y):
    return (MAZE_OFFSET_X + cell_x * CELL_SIZE + CELL_SIZE // 2,
            MAZE_OFFSET_Y + cell_y * CELL_SIZE + CELL_SIZE // 2)


def make_world():
    walls = _carve_maze(required_cells=FOOD_CELLS)
    food_spots = [
        FoodSpot(_cell_center(x, y)) for x, y in FOOD_CELLS
    ]
    return walls, food_spots


def _dark_start_position(maps, spots, avoid=None, avoid_positions=()):
    candidates = [
        (_cell_center(x, y), maps.light_at(pygame.Vector2(_cell_center(x, y)), spots))
        for y in range(MAZE_ROWS)
        for x in range(MAZE_COLS)
    ]
    dark_cells = [position for position, light in candidates if light < FOOD_LIGHT_THRESHOLD]
    if not dark_cells:
        raise RuntimeError("Не удалось найти тёмную стартовую клетку для амёбы")
    avoided = [pygame.Vector2(position) for position in avoid_positions]
    if avoid is not None:
        avoided.append(pygame.Vector2(avoid))
    if not avoided:
        return pygame.Vector2(min(
            dark_cells,
            key=lambda position: maps.light_at(pygame.Vector2(position), spots),
        ))
    return pygame.Vector2(max(
        dark_cells,
        key=lambda position: (min(pygame.Vector2(position).distance_squared_to(other)
                                  for other in avoided),
                              -maps.light_at(pygame.Vector2(position), spots)),
    ))


def _reachable_amoeba_count(walls, start, amoebas):
    start = pygame.Vector2(start)
    agent = Ameba()
    agent.set_start(start)
    target_cells = {}
    for amoeba in amoebas:
        cell = agent._cell_for_position(amoeba.position)
        target_cells.setdefault(cell, []).append(amoeba)

    start_cell = agent._cell_for_position(start)
    reachable = {start_cell}
    queue = deque([start_cell])
    reached_amoebas = set()
    directions = ((1, 0), (-1, 0), (0, 1), (0, -1))

    while queue and len(reached_amoebas) < MIN_REACHABLE_PREY_FOR_PREDATOR:
        cell_x, cell_y = queue.popleft()
        origin = pygame.Vector2(_cell_center(cell_x, cell_y))
        for amoeba in target_cells.get((cell_x, cell_y), ()):
            delta = amoeba.position - origin
            if delta.length_squared() == 0:
                reached_amoebas.add(amoeba)
            else:
                agent.position.update(origin)
                if agent._path_clear(walls, delta.normalize(), delta.length()):
                    reached_amoebas.add(amoeba)
        if len(reached_amoebas) >= MIN_REACHABLE_PREY_FOR_PREDATOR:
            break

        for dx, dy in directions:
            neighbor = (cell_x + dx, cell_y + dy)
            if (neighbor in reachable or not 0 <= neighbor[0] < MAZE_COLS
                    or not 0 <= neighbor[1] < MAZE_ROWS):
                continue
            destination = pygame.Vector2(_cell_center(*neighbor))
            delta = destination - origin
            agent.position.update(origin)
            if agent._path_clear(walls, delta.normalize(), delta.length()):
                reachable.add(neighbor)
                queue.append(neighbor)

    return len(reached_amoebas)


def _predator_spawn_position(walls, maps, spots, amoebas, prefer_near=False):
    candidates = [
        _cell_center(x, y)
        for y in range(MAZE_ROWS)
        for x in range(MAZE_COLS)
    ]
    dark_candidates = [position for position in candidates
                       if maps.light_at(pygame.Vector2(position), spots) < FOOD_LIGHT_THRESHOLD]
    minimum_separation = (CELL_SIZE * 2) ** 2 if prefer_near else 0
    dark_candidates = [position for position in dark_candidates
                       if min(pygame.Vector2(position).distance_squared_to(amoeba.position)
                              for amoeba in amoebas) >= minimum_separation]
    dark_candidates.sort(
        key=lambda position: (
            min(pygame.Vector2(position).distance_squared_to(amoeba.position)
                for amoeba in amoebas),
            -maps.light_at(pygame.Vector2(position), spots),
        ),
        reverse=not prefer_near,
    )
    for position in dark_candidates:
        if _reachable_amoeba_count(walls, position, amoebas) >= MIN_REACHABLE_PREY_FOR_PREDATOR:
            return pygame.Vector2(position)
    return None


def make_wall_overlay(walls):
    overlay = pygame.Surface((WORLD_WIDTH, WORLD_HEIGHT), pygame.SRCALPHA)
    for wall in walls:
        shadow = wall.move(2, 3)
        pygame.draw.rect(overlay, (3, 7, 13, 115), shadow, border_radius=2)
        pygame.draw.rect(overlay, (8, 14, 23, 225), wall, border_radius=2)
        pygame.draw.rect(overlay, (91, 119, 137, 235), wall, 1, border_radius=2)
        highlight = pygame.Rect(wall.left + 2, wall.top + 1, max(1, wall.width - 4), 1)
        pygame.draw.rect(overlay, (143, 176, 188, 190), highlight)
    return overlay


@lru_cache(maxsize=32)
def _radial_glow(color, radius):
    diameter = radius * 2
    pixels = bytearray(diameter * diameter * 4)
    center = radius - 0.5
    for y in range(diameter):
        for x in range(diameter):
            distance = math.hypot(x - center, y - center) / radius
            if distance >= 1:
                continue
            alpha = round(95 * (1 - distance) ** 2)
            offset = (y * diameter + x) * 4
            pixels[offset:offset + 4] = bytes((*color, alpha))
    return pygame.image.frombuffer(pixels, (diameter, diameter), "RGBA").copy()


def _draw_glow(screen, position, color, radius):
    glow = _radial_glow(color, radius)
    screen.blit(glow, (round(position[0] - radius), round(position[1] - radius)),
                special_flags=pygame.BLEND_RGBA_ADD)


def draw(screen, amoebas, predators, eggs, spots, maps, wall_overlay):
    screen.blit(maps.render(spots), (0, 0))
    screen.blit(wall_overlay, (0, 0))

    for spot_index, spot in enumerate(spots):
        center = (round(spot.position.x), round(spot.position.y))
        if spot.value > 0:
            intensity = spot.value / 100.0
            _draw_glow(screen, center, (255, 176, 74), round(32 + 20 * intensity))
            previous_point = None
            for sample_index in range(49):
                angle = math.tau * sample_index / 48
                sample_point = spot.position + pygame.Vector2(math.cos(angle), math.sin(angle)) * FOOD_RADIUS
                if maps.light_from_spot(spot_index, sample_point) >= FOOD_LIGHT_THRESHOLD:
                    pixel = (round(sample_point.x), round(sample_point.y))
                    if previous_point is not None:
                        pygame.draw.line(screen, (205, 164, 76), previous_point, pixel, 1)
                    previous_point = pixel
                else:
                    previous_point = None
            pygame.draw.circle(screen, (255, round(218 * intensity), 100), center, 8)
            pygame.draw.circle(screen, (255, 235, 176), center, 4)
            pygame.draw.circle(screen, (255, 255, 228), center, 2)
            pygame.draw.circle(screen, (255, 191, 94), center, 12, 1)
            pygame.draw.circle(screen, (235, 185, 82, 120), center, 16, 1)
    for agent in (*amoebas, *predators):
        is_predator = isinstance(agent, Predator)
        outer_color = (218, 78, 87) if is_predator else (38, 221, 210)
        body_color = (115, 34, 51) if is_predator else (250, 248, 230)
        trail_color = (129, 43, 62) if is_predator else (30, 175, 165)
        trail_points = [(round(point.x), round(point.y)) for point in list(agent.trail)[-120:]]
        if len(trail_points) > 1:
            pygame.draw.lines(screen, (*trail_color, 75), False, trail_points, 4)
            pygame.draw.lines(screen, trail_color, False, trail_points, 1)
        _draw_glow(screen, agent.position, outer_color, 17 if is_predator else 15)
        pygame.draw.circle(screen, (8, 14, 22), agent.position, agent.radius + 3)
        pygame.draw.circle(screen, outer_color, agent.position, agent.radius + 1)
        pygame.draw.circle(screen, body_color, agent.position, agent.radius - 2)
        pygame.draw.arc(screen, (255, 255, 255, 180),
                        pygame.Rect(round(agent.position.x - agent.radius + 1),
                                    round(agent.position.y - agent.radius + 1),
                                    (agent.radius - 1) * 2, (agent.radius - 1) * 2),
                        math.radians(205), math.radians(320), 2)
        facing_direction = pygame.Vector2(math.cos(agent.heading), math.sin(agent.heading))
        facing = agent.position + facing_direction * 12
        pygame.draw.line(screen, (38, 20, 25) if is_predator else (65, 35, 25),
                         agent.position, facing, 2)
        if is_predator:
            eye_center = agent.position + facing_direction * 3
            eye_side = pygame.Vector2(-facing_direction.y, facing_direction.x) * 2
            for eye in (eye_center - eye_side, eye_center + eye_side):
                pygame.draw.circle(screen, (255, 228, 208), eye, 1)
                pygame.draw.circle(screen, (35, 17, 24), eye + facing_direction, 1)
        satiety_max = 100.0 if is_predator else MAX_SATIETY
        satiety = max(0.0, min(1.0, agent.satiety / satiety_max))
        bar = pygame.Rect(0, 0, 48, 7)
        bar.centerx = round(agent.position.x)
        bar.bottom = round(agent.position.y - agent.radius - 8)
        bar.clamp_ip(screen.get_rect())
        pygame.draw.rect(screen, (12, 16, 22), bar.inflate(4, 4), border_radius=4)
        pygame.draw.rect(screen, (220, 65, 70), bar, border_radius=5)
        fill = bar.copy()
        fill.width = round(bar.width * satiety)
        if fill.width > 0:
            color = (round(220 - 155 * satiety), round(65 + 155 * satiety), 70)
            pygame.draw.rect(screen, color, fill, border_radius=5)
        full_hunger_mark = round(bar.width * (100.0 / satiety_max))
        pygame.draw.line(screen, (245, 245, 235),
                         (bar.left + full_hunger_mark, bar.top + 1),
                         (bar.left + full_hunger_mark, bar.bottom - 1), 1)
        pygame.draw.rect(screen, (225, 232, 235), bar, 1, border_radius=5)

    for egg in eggs:
        egg_color = (235, 103, 113) if egg.species == "predator" else (232, 214, 125)
        _draw_glow(screen, egg.position, egg_color, 20)
        pygame.draw.ellipse(screen, (16, 22, 31),
                            pygame.Rect(round(egg.position.x - 8), round(egg.position.y - 10), 16, 20))
        pygame.draw.ellipse(screen, egg_color,
                            pygame.Rect(round(egg.position.x - 6), round(egg.position.y - 8), 12, 16))
        pygame.draw.ellipse(screen, (255, 244, 218),
                            pygame.Rect(round(egg.position.x - 3), round(egg.position.y - 6), 4, 6))
        hatch_progress = max(0.0, min(1.0, 1.0 - egg.remaining / EGG_HATCH_SECONDS))
        pygame.draw.arc(screen, (247, 250, 255),
                        pygame.Rect(round(egg.position.x - 11), round(egg.position.y - 11), 22, 22),
                        -math.pi / 2, -math.pi / 2 + math.tau * hatch_progress, 2)


def read_settings(shared_settings):
    with shared_settings.get_lock():
        return tuple(shared_settings[:])


def _pretrain_amoeba_episode(agent, walls, spots, maps, shared_settings, stop_event):
    agent.set_start(_dark_start_position(maps, spots))
    for spot in spots:
        spot.value = 100.0
    eggs_before = agent.total_eggs_laid
    sessions_before = agent.accepted_sessions
    episode_before = agent.episode
    steps_in_episode = 0
    while agent.episode == episode_before and steps_in_episode < 1500 and not stop_event.is_set():
        agent.update(agent.decision_interval, walls, spots, maps, read_settings(shared_settings))
        steps_in_episode += 1
    if agent.episode == episode_before:
        if agent.previous_state is not None:
            agent.episode_experiences.append((
                agent.previous_state, agent.previous_action,
                agent.pending_reward - 100.0, None, True,
            ))
        agent.reset_episode()
    return (walls, spots, maps, agent.total_eggs_laid - eggs_before,
            agent.accepted_sessions - sessions_before)


def _pretrain_predator_episode(predator_template, amoeba_template, walls, spots, maps,
                               shared_settings, stop_event):
    amoebas = []
    starts = []
    for _ in range(MIN_AMOEBA_FOR_PREDATOR):
        start = _dark_start_position(maps, spots, avoid_positions=starts)
        starts.append(start)
        prey = Ameba()
        prey.defer_training_save = True
        prey.training_brain_bank = amoeba_template.training_brain_bank
        prey.bind_genome(amoeba_template.genome, amoeba_template.q_table)
        prey.shared_training_steps = amoeba_template.shared_training_steps
        prey.training_state_path = amoeba_template.training_state_path
        prey.set_start(start)
        amoebas.append(prey)

    predator_start = _predator_spawn_position(walls, maps, spots, amoebas, prefer_near=True)
    if predator_start is None:
        return walls, spots, maps, 0

    hunter = Predator()
    hunter.defer_training_save = True
    hunter.training_brain_bank = predator_template.training_brain_bank
    hunter.bind_genome(predator_template.genome, predator_template.q_table)
    hunter.shared_training_steps = predator_template.shared_training_steps
    hunter.training_state_path = predator_template.training_state_path
    hunter.set_start(predator_start)
    eggs = []
    eggs_before = hunter.total_eggs_laid
    settings = read_settings(shared_settings)

    for _ in range(1500):
        if stop_event.is_set() or hunter.dead or hunter.total_eggs_laid > eggs_before:
            break
        for prey in tuple(amoebas):
            prey.neighbor_agents = tuple(amoebas)
            laid_before = prey.total_eggs_laid
            prey.update(prey.decision_interval, walls, spots, maps, settings)
            for _ in range(prey.total_eggs_laid - laid_before):
                eggs.append(Egg.from_parent("amoeba", prey.position, prey))
        amoebas[:] = [prey for prey in amoebas if not prey.dead]
        hunter.update(hunter.decision_interval, walls, spots, maps, settings, amoebas)
        for egg in tuple(eggs):
            egg.remaining -= hunter.decision_interval
            if egg.remaining <= 0:
                prey = Ameba()
                prey.defer_training_save = True
                prey.training_brain_bank = egg.brain_bank or amoeba_template.training_brain_bank
                prey.bind_genome(egg.genome, egg.inherited_q_table or amoeba_template.q_table)
                prey.shared_training_steps = amoeba_template.shared_training_steps
                prey.training_state_path = amoeba_template.training_state_path
                prey.set_start(egg.position)
                amoebas.append(prey)
                eggs.remove(egg)

    if not hunter.dead and hunter.total_eggs_laid == eggs_before and not stop_event.is_set():
        hunter.mark_dead(-100.0)
    return (walls, spots, maps, hunter.total_eggs_laid - eggs_before,
            hunter.accepted_sessions)


def pretrain_agents(amoeba, predator, walls, spots, maps, shared_settings,
                    stop_event, progress, cycles, logger=None):
    amoeba.defer_training_save = True
    predator.defer_training_save = True
    with progress.get_lock():
        progress[0] = 0
        progress[1] = cycles
        progress[2] = 1
        progress[3] = 0
        progress[4] = 0
        progress[5] = 0
        progress[6] = 0

    amoeba_eggs = 0
    predator_eggs = 0
    amoeba_learned_sessions = 0
    predator_learned_sessions = 0
    for cycle_index in range(cycles):
        if stop_event.is_set():
            break
        amoeba_steps_before = amoeba.steps
        predator_steps_before = predator.steps
        predator_kills_before = predator.total_kills
        if logger is not None:
            logger.record_settings(read_settings(shared_settings))
        walls, spots = make_world()
        maps = SensorMaps(walls, spots)
        with progress.get_lock():
            progress[2] = 1
        walls, spots, maps, laid, learned = _pretrain_amoeba_episode(
            amoeba, walls, spots, maps, shared_settings, stop_event,
        )
        amoeba_eggs += laid
        amoeba_learned_sessions += learned
        with progress.get_lock():
            progress[3] = amoeba_eggs
            progress[5] = amoeba_learned_sessions
            progress[2] = 3
        for spot in spots:
            spot.value = 100.0
        walls, spots, maps, laid, learned = _pretrain_predator_episode(
            predator, amoeba, walls, spots, maps, shared_settings, stop_event,
        )
        predator_eggs += laid
        predator_learned_sessions += learned
        amoeba.save_training_state()
        predator.save_training_state()
        if logger is not None:
            logger.write(
                "pretrain_cycle", cycle=cycle_index + 1,
                amoeba_eggs=amoeba_eggs,
                amoeba_sessions=amoeba_learned_sessions,
                predator_eggs=predator_eggs,
                predator_sessions=predator_learned_sessions,
                predator_kills=predator.total_kills - predator_kills_before,
                amoeba_steps=amoeba.steps - amoeba_steps_before,
                predator_steps=predator.steps - predator_steps_before,
                amoeba_genotypes=len(amoeba.training_brain_bank),
                predator_genotypes=len(predator.training_brain_bank),
            )
        with progress.get_lock():
            progress[0] = cycle_index + 1
            progress[4] = predator_eggs
            progress[6] = predator_learned_sessions

    if not stop_event.is_set():
        walls, spots = make_world()
        maps = SensorMaps(walls, spots)
        amoeba.set_start(_dark_start_position(maps, spots))
    amoeba.defer_training_save = False
    predator.defer_training_save = False
    return walls, spots, maps


def run_simulation(shared_settings, stop_event, pretrain_episodes, progress):
    pygame.init()
    seed = random.randrange(2 ** 32)
    random.seed(seed)
    logger = SimulationLogger()
    logger.write(
        "run_start", seed=seed, pretrain_cycles=pretrain_episodes,
        settings=read_settings(shared_settings),
        gene_levels=GENE_LEVELS,
        genome_model={
            "speed_scale": "0.8 + 0.04 * speed_gene",
            "vision_scale": "0.75 + 0.05 * sense_gene",
            "smell_scale": "1.25 - 0.05 * sense_gene",
            "move_energy_per_pixel": MOVE_ENERGY_PER_PIXEL,
        },
    )
    logger.record_settings(read_settings(shared_settings))
    walls, spots = make_world()
    maps = SensorMaps(walls, spots)
    amoeba_state_path = Path(__file__).resolve().with_name("q_table.json")
    predator_state_path = Path(__file__).resolve().with_name("predator_q_table.json")
    first_amoeba = Ameba()
    first_amoeba.load_training_state(amoeba_state_path)
    first_amoeba.shared_training_steps = [first_amoeba.steps]
    first_amoeba.set_start(_dark_start_position(maps, spots))
    predator = Predator()
    predator.load_training_state(predator_state_path)
    predator.shared_training_steps = [predator.steps]

    if pretrain_episodes > 0:
        walls, spots, maps = pretrain_agents(
            first_amoeba, predator, walls, spots, maps, shared_settings,
            stop_event, progress, pretrain_episodes, logger,
        )

    with progress.get_lock():
        progress[0] = pretrain_episodes
        progress[1] = pretrain_episodes
        progress[2] = 2

    if stop_event.is_set():
        logger.close("stopped_during_pretraining")
        pygame.quit()
        return

    screen = pygame.display.set_mode((WORLD_WIDTH, WORLD_HEIGHT))
    pygame.display.set_caption("Амёбы и хищники — 800 × 600")
    clock = pygame.time.Clock()
    wall_overlay = make_wall_overlay(walls)
    amoebas = [first_amoeba]
    predators = []
    eggs = []
    amoeba_brain = first_amoeba.q_table
    predator_brain = predator.q_table
    amoeba_steps = first_amoeba.shared_training_steps
    predator_steps = predator.shared_training_steps
    predator_ever_spawned = False
    running = True

    while running and not stop_event.is_set():
        dt = min(clock.tick(FPS) / 1000.0, 0.05)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        settings = read_settings(shared_settings)
        logger.record_settings(settings)

        for amoeba in tuple(amoebas):
            amoeba.reproduction_allowed = True
            amoeba.neighbor_agents = tuple(amoebas)
            eggs_before = amoeba.total_eggs_laid
            amoeba.update(dt, walls, spots, maps, settings, auto_respawn=False)
            for _ in range(amoeba.total_eggs_laid - eggs_before):
                egg = Egg.from_parent("amoeba", amoeba.position, amoeba)
                eggs.append(egg)
                logger.write("egg_laid", species="amoeba", parent_genome=egg.parent_genome,
                             child_genome=egg.genome)
            if amoeba.dead:
                logger.deaths["amoeba"] += 1
                logger.write("death", species="amoeba", genome=amoeba.genome,
                             satiety=round(amoeba.satiety, 2),
                             visited_cells=len(amoeba.visited_cells))
                amoebas.remove(amoeba)

        if not predator_ever_spawned and len(amoebas) >= MIN_AMOEBA_FOR_PREDATOR:
            predator_start = _predator_spawn_position(walls, maps, spots, amoebas)
            if predator_start is not None:
                predators.append(_hatch_agent(
                    Egg("predator", predator_start), amoeba_brain, predator_brain,
                    amoeba_steps, predator_steps, amoeba_state_path, predator_state_path,
                    first_amoeba.training_brain_bank, predator.training_brain_bank,
                ))
                predator_ever_spawned = True

        for hunter in tuple(predators):
            hunter.reproduction_allowed = True
            kills_before = hunter.total_kills
            living_prey_before = {id(prey): prey for prey in amoebas if not prey.dead}
            hunter.update(dt, walls, spots, maps, settings, amoebas)
            new_kills = hunter.total_kills - kills_before
            if new_kills:
                logger.kills += new_kills
                logger.write("predation", predator_genome=hunter.genome, kills=new_kills,
                             total_kills=hunter.total_kills,
                             remaining_prey=sum(not prey.dead for prey in amoebas))
            for prey in living_prey_before.values():
                if prey.dead:
                    logger.deaths["amoeba"] += 1
                    logger.write("death", species="amoeba", cause="predation",
                                 genome=prey.genome, satiety=round(prey.satiety, 2),
                                 visited_cells=len(prey.visited_cells))
            while hunter.hatch_eggs_pending:
                egg = Egg.from_parent("predator", hunter.position, hunter)
                eggs.append(egg)
                logger.write("egg_laid", species="predator", parent_genome=egg.parent_genome,
                             child_genome=egg.genome)
                hunter.hatch_eggs_pending -= 1
            if hunter.dead:
                logger.deaths["predator"] += 1
                logger.write("death", species="predator", genome=hunter.genome,
                             satiety=round(hunter.satiety, 2),
                             visited_cells=len(hunter.visited_cells))
                predators.remove(hunter)
        amoebas[:] = [amoeba for amoeba in amoebas if not amoeba.dead]

        if not amoebas or (predator_ever_spawned and not predators):
            if not predators:
                for amoeba in amoebas:
                    if amoeba.previous_state is not None:
                        amoeba.episode_experiences.append((
                            amoeba.previous_state, amoeba.previous_action,
                            amoeba.pending_reward, None, True,
                        ))
                    amoeba.finish_learning_session()
            for hunter in predators:
                hunter.mark_dead(-100.0)
            predators.clear()
            eggs.clear()
            logger.write("world_reset", reason="population_extinction")
            walls, spots = make_world()
            maps = SensorMaps(walls, spots)
            wall_overlay = make_wall_overlay(walls)
            amoeba_start = _dark_start_position(maps, spots)
            amoeba_egg = Egg("amoeba", amoeba_start)
            amoebas.append(_hatch_agent(
                amoeba_egg, amoeba_brain, predator_brain, amoeba_steps, predator_steps,
                amoeba_state_path, predator_state_path,
                first_amoeba.training_brain_bank, predator.training_brain_bank,
            ))
            predator_ever_spawned = False
            continue

        for egg in tuple(eggs):
            egg.remaining -= dt
            if egg.remaining > 0:
                continue
            if egg.species == "amoeba":
                agent = _hatch_agent(
                    egg, amoeba_brain, predator_brain, amoeba_steps, predator_steps,
                    amoeba_state_path, predator_state_path,
                    first_amoeba.training_brain_bank, predator.training_brain_bank,
                )
                amoebas.append(agent)
                logger.births["amoeba"] += 1
                logger.write("birth", species="amoeba", parent_genome=egg.parent_genome,
                             genome=agent.genome, q_entries=len(agent.q_table))
                eggs.remove(egg)
            elif egg.species == "predator":
                if len(amoebas) < MIN_AMOEBA_FOR_PREDATOR:
                    continue
                agent = _hatch_agent(
                    egg, amoeba_brain, predator_brain, amoeba_steps, predator_steps,
                    amoeba_state_path, predator_state_path,
                    first_amoeba.training_brain_bank, predator.training_brain_bank,
                )
                predators.append(agent)
                logger.births["predator"] += 1
                logger.write("birth", species="predator", parent_genome=egg.parent_genome,
                             genome=agent.genome, q_entries=len(agent.q_table))
                eggs.remove(egg)

        if not amoebas and not predators and not eggs:
            walls, spots = make_world()
            maps = SensorMaps(walls, spots)
            wall_overlay = make_wall_overlay(walls)
            amoeba_egg = Egg("amoeba", _dark_start_position(maps, spots))
            predator_egg = Egg("predator", _dark_start_position(
                maps, spots, amoeba_egg.position,
            ))
            amoebas.append(_hatch_agent(
                amoeba_egg, amoeba_brain, predator_brain, amoeba_steps, predator_steps,
                amoeba_state_path, predator_state_path,
                first_amoeba.training_brain_bank, predator.training_brain_bank,
            ))
            predators.append(_hatch_agent(
                predator_egg, amoeba_brain, predator_brain, amoeba_steps, predator_steps,
                amoeba_state_path, predator_state_path,
                first_amoeba.training_brain_bank, predator.training_brain_bank,
            ))
        draw(screen, amoebas, predators, eggs, spots, maps, wall_overlay)
        pygame.display.flip()
        logger.advance(dt, amoebas, predators, eggs, spots,
                       first_amoeba.steps, predator.steps)

    logger.close("stopped" if stop_event.is_set() else "window_closed")
    pygame.quit()


def run_control_window():
    shared_settings = multiprocessing.Array("d", SETTING_DEFAULTS, lock=True)
    progress = multiprocessing.Array("d", [0, 0, 0, 0, 0, 0, 0], lock=True)
    simulation = {
        "process": None,
        "stop_event": None,
        "pretrain_episodes": 0,
        "completion_prompted": False,
    }
    root = tk.Tk()
    root.title("Параметры амёбы")
    root.resizable(False, False)
    root.geometry("460x690")

    ttk.Label(root, text="Настройки симуляции", font=("Segoe UI", 15, "bold")).pack(pady=(18, 4))
    ttk.Label(root, text="Изменения применяются сразу", foreground="#52616b").pack(pady=(0, 12))
    controls = ttk.Frame(root, padding=(20, 0, 20, 0))
    controls.pack(fill="x")

    for index, (name, bounds, initial) in enumerate(zip(SETTING_NAMES, SETTING_RANGES, SETTING_DEFAULTS)):
        minimum, maximum, resolution = bounds
        row = ttk.Frame(controls)
        row.pack(fill="x", pady=5)
        value_label = ttk.Label(row, width=7, anchor="e")
        value_label.pack(side="right")
        ttk.Label(row, text=name).pack(side="left")

        def on_change(value, setting_index=index, output=value_label, step=resolution):
            numeric_value = round(float(value) / step) * step
            with shared_settings.get_lock():
                shared_settings[setting_index] = numeric_value
            output.configure(text=f"{numeric_value:g}")

        scale = tk.Scale(controls, from_=minimum, to=maximum, resolution=resolution,
                         orient="horizontal", showvalue=False, length=385, command=on_change)
        scale.set(initial)
        scale.pack(fill="x")
        value_label.configure(text=f"{initial:g}")

    status = ttk.Label(root, text="Симуляция остановлена", foreground="#52616b")
    status.pack(pady=(8, 4))
    training_row = ttk.Frame(root)
    training_row.pack(pady=(8, 2))
    ttk.Label(training_row, text="Циклов предобучения:").pack(side="left", padx=(0, 8))
    episode_count = tk.IntVar(value=100)
    tk.Spinbox(training_row, from_=1, to=5000, increment=25, textvariable=episode_count, width=8).pack(side="left")
    ttk.Label(
        root,
        text="В каждом цикле тренируются оба вида. Амёбе нужна заданная норма яиц; хищнику — 3 пойманные амёбы и отложенное яйцо. Только такие сессии обучают Q-таблицы.",
        wraplength=405,
        justify="center",
    ).pack(pady=(2, 8))

    def start_simulation(pretrain_episodes=0):
        process = simulation["process"]
        if process is not None and process.is_alive():
            simulation["stop_event"].set()
            process.join(timeout=2)
            if process.is_alive():
                process.terminate()
                process.join(timeout=1)
        stop_event = multiprocessing.Event()
        with progress.get_lock():
            progress[0] = 0
            progress[1] = pretrain_episodes
            progress[2] = 0
            progress[3] = 0
            progress[4] = 0
            progress[5] = 0
            progress[6] = 0
        process = multiprocessing.Process(
            target=run_simulation,
            args=(shared_settings, stop_event, pretrain_episodes, progress),
            daemon=True,
        )
        process.start()
        simulation["process"] = process
        simulation["stop_event"] = stop_event
        simulation["pretrain_episodes"] = pretrain_episodes
        simulation["completion_prompted"] = False
        status.configure(text="Подготовка симуляции…", foreground="#52616b")

    ttk.Button(root, text="Предобучить и запустить", command=lambda: start_simulation(max(1, episode_count.get()))).pack(pady=5)
    ttk.Button(root, text="Запустить без предобучения", command=lambda: start_simulation(0)).pack(pady=5)
    ttk.Label(root, text="Закройте окно симуляции или это меню, чтобы завершить работу.", wraplength=390).pack(pady=(4, 12))

    def refresh_status():
        process = simulation["process"]
        with progress.get_lock():
            completed, total, phase = int(progress[0]), int(progress[1]), int(progress[2])
            amoeba_eggs, predator_eggs = int(progress[3]), int(progress[4])
            amoeba_sessions, predator_sessions = int(progress[5]), int(progress[6])
        if process is not None and process.is_alive():
            if phase == 1:
                status.configure(
                    text=f"Цикл {completed + 1} / {total}: амёба · яйца {amoeba_eggs}, сессии {amoeba_sessions} · хищник · яйца {predator_eggs}, сессии {predator_sessions}",
                    foreground="#8a6414",
                )
            elif phase == 3:
                status.configure(
                    text=f"Цикл {completed + 1} / {total}: хищник · яйца амёбы {amoeba_eggs} (сессии {amoeba_sessions}) · хищника {predator_eggs} (сессии {predator_sessions})",
                    foreground="#8a6414",
                )
            elif phase == 2 and total:
                status.configure(
                    text=f"Завершено {total} циклов · яиц амёбы {amoeba_eggs} (сессий {amoeba_sessions}), хищника {predator_eggs} (сессий {predator_sessions}) · запущено",
                    foreground="#287a45",
                )
            else:
                status.configure(text="Симуляция запущена", foreground="#287a45")
        elif process is not None:
            status.configure(text="Сессия завершена", foreground="#52616b")
            if not simulation["completion_prompted"]:
                simulation["completion_prompted"] = True
                if messagebox.askyesno(
                    "Сессия завершена",
                    "Продолжить симуляцию с теми же параметрами?",
                    parent=root,
                ):
                    start_simulation(simulation["pretrain_episodes"])
                else:
                    close_window()
                    return
        root.after(400, refresh_status)

    root.after(400, refresh_status)

    def close_window():
        process = simulation["process"]
        stop_event = simulation["stop_event"]
        if process is not None and process.is_alive():
            stop_event.set()
            process.join(timeout=2)
            if process.is_alive():
                process.terminate()
                process.join(timeout=1)
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", close_window)
    root.mainloop()


def run_startup_tests():
    project_dir = Path(__file__).resolve().parent
    suite = unittest.defaultTestLoader.discover(
        start_dir=str(project_dir / "tests"),
        pattern="test_*.py",
        top_level_dir=str(project_dir),
    )
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit("Автоматические тесты не пройдены; запуск симуляции отменён.")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    run_startup_tests()
    run_control_window()
