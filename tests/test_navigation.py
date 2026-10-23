import unittest
import math
from unittest.mock import patch
from collections import deque
from pathlib import Path
from tempfile import TemporaryDirectory

import pygame

import main


class NavigationRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pygame.init()

    @classmethod
    def tearDownClass(cls):
        pygame.quit()

    def setUp(self):
        self.agent = main.Ameba()

    def test_midi_music_asset_is_a_standard_midi_file(self):
        midi_path = Path(__file__).resolve().parents[1] / "music.mid"
        data = midi_path.read_bytes()
        self.assertTrue(data.startswith(b"MThd"))
        self.assertEqual(int.from_bytes(data[10:12], "big"), 3)

    def test_audio_disables_cleanly_when_no_device_is_available(self):
        with patch.object(main.pygame.mixer, "init", side_effect=main.pygame.error):
            audio = main.SimulationAudio(Path(__file__).resolve().parents[1])
        self.assertFalse(audio.available)
        audio.play("egg")
        audio.close()

    def test_circle_can_pass_clear_of_a_wall_corner(self):
        wall = pygame.Rect(100, 100, 10, 10)
        self.assertFalse(self.agent._wall_ahead([wall], pygame.Vector2(120, 120)))

    def test_circle_stops_at_a_wall_face(self):
        wall = pygame.Rect(100, 80, 6, 40)
        self.assertTrue(self.agent._wall_ahead([wall], pygame.Vector2(95, 100)))

    def test_open_corridor_is_not_reported_as_blocked(self):
        self.agent.position.update(100, 100)
        self.assertTrue(self.agent._path_clear([], pygame.Vector2(1, 0), 100))

    def test_wall_across_corridor_blocks_the_path(self):
        self.agent.position.update(100, 100)
        wall = pygame.Rect(140, 70, 6, 60)
        self.assertFalse(self.agent._path_clear([wall], pygame.Vector2(1, 0), 100))

    def test_satiety_above_100_lays_egg_and_resets_to_100(self):
        self.agent.satiety = 129.0
        self.assertFalse(self.agent._gain_satiety(0.5))
        self.assertEqual(self.agent.satiety, 129.5)
        self.assertTrue(self.agent._gain_satiety(0.5))
        self.assertEqual(self.agent.eggs_laid, 1)
        self.assertEqual(self.agent.satiety, 100.0)

    def test_low_satiety_still_uses_normal_sensor_decisions(self):
        class FlatMaps:
            def light_at(self, point, spots):
                return 0.3

            def heat_at(self, point, spots):
                return 0.2

        self.agent.satiety = 20.0
        self.agent._food_at_position = lambda spots, maps: None
        self.agent.sensors = lambda *args: (4, 4, 0, 0, 0, 0, 0, 0)
        self.agent._learning_potential = lambda *args, **kwargs: 0.0
        choices = []
        self.agent.choose_action = lambda *args: choices.append(True) or 0

        self.agent.update(0.14, [], [], FlatMaps(), (100, 260, 143, 14, 3.5, 8, 2))

        self.assertEqual(choices, [True])
        self.assertAlmostEqual(self.agent.satiety, 19.51)

    def test_agent_keeps_course_when_sensors_give_no_direction(self):
        class FlatMaps:
            def light_at(self, point, spots):
                return 0.3

            def heat_at(self, point, spots):
                return 0.3

        center = pygame.Vector2(main.WORLD_WIDTH / 2, main.WORLD_HEIGHT / 2)
        self.agent.position.update(center)
        self.agent.heading = 0.0
        self.agent.trail.clear()
        self.agent.trail.extend((center - pygame.Vector2(20, 0), center.copy()))
        self.agent.epsilon = 1.0

        action = self.agent.choose_action((0,) * 8, [], [], FlatMaps(), 100, 260)

        self.assertEqual(action, 0)

    def test_amoeba_does_not_immediately_reverse_its_turn(self):
        class FlatMaps:
            def light_at(self, point, spots):
                return 0.3

            def heat_at(self, point, spots):
                return 0.3

        state = (0,) * self.agent.state_size
        center = pygame.Vector2(main.WORLD_WIDTH / 2, main.WORLD_HEIGHT / 2)
        self.agent.position.update(center)
        self.agent.heading = 0.0
        self.agent.trail.clear()
        self.agent.trail.extend((center - pygame.Vector2(20, 0), center.copy()))
        self.agent.last_action = 1
        self.agent.epsilon = 0.0
        self.agent.q_table[state] = [0.0, 0.8, 0.95]
        self.agent._lookahead_cell = lambda walls, direction, distance: (12, 8)
        self.agent.visited_cells[(12, 8)] = 1

        action = self.agent.choose_action(state, [], [], FlatMaps(), 100, 260)

        self.assertEqual(action, 1)

    def test_session_below_egg_threshold_is_not_learned(self):
        self.agent.training_egg_threshold = 2
        self.agent.eggs_laid = 1
        self.agent.episode_experiences.append(((1, 2, 0), 0, 10.0, None, True))
        self.agent.reset_episode()
        self.assertEqual(self.agent.accepted_sessions, 0)
        self.assertEqual(self.agent.q_table, {})

    def test_more_eggs_increase_session_learning_weight_geometrically(self):
        experience = ((1, 2, 0), 0, 10.0, None, True)
        one_egg = main.Ameba()
        one_egg.training_egg_threshold = 1
        one_egg.eggs_laid = 1
        one_egg.episode_experiences.append(experience)
        one_egg.reset_episode()

        two_eggs = main.Ameba()
        two_eggs.training_egg_threshold = 1
        two_eggs.eggs_laid = 2
        two_eggs.episode_experiences.append(experience)
        two_eggs.reset_episode()

        reward_one_egg = one_egg.q_table[(1, 2, 0)][0]
        reward_two_eggs = two_eggs.q_table[(1, 2, 0)][0]
        self.assertAlmostEqual(reward_one_egg, 1.0)
        self.assertAlmostEqual(reward_two_eggs, 1.9)
        self.assertEqual(one_egg.accepted_sessions, 1)
        self.assertEqual(two_eggs.accepted_sessions, 1)

    def test_learning_state_includes_satiety_and_light_intensity(self):
        class FlatMaps:
            light = 0.4
            heat = 0.2

            def light_at(self, point, spots):
                return self.light

            def heat_at(self, point, spots):
                return self.heat

        maps = FlatMaps()
        self.agent.position.update(main.WORLD_WIDTH / 2, main.WORLD_HEIGHT / 2)
        well_fed = self.agent.sensors([], maps, [], 100, 260)
        self.agent.satiety = 30
        hungry = self.agent.sensors([], maps, [], 100, 260)
        maps.light = 0.9
        brighter = self.agent.sensors([], maps, [], 100, 260)

        self.assertEqual(len(well_fed), 8)
        self.assertNotEqual(well_fed[-1], hungry[-1])
        self.assertNotEqual(well_fed[5], brighter[5])

    def test_successful_training_session_persists_q_table(self):
        with TemporaryDirectory() as temporary_directory:
            state_path = Path(temporary_directory) / "q_table.json"
            trained_agent = main.Ameba()
            self.assertFalse(trained_agent.load_training_state(state_path))
            trained_agent.training_egg_threshold = 1
            trained_agent.eggs_laid = 1
            trained_agent.episode_experiences.append(((1, 2, 0, 1, 0, 0, 0, 3), 0, 10.0, None, True))
            trained_agent.reset_episode()

            restored_agent = main.Ameba()
            self.assertTrue(restored_agent.load_training_state(state_path))
            self.assertEqual(restored_agent.q_table, trained_agent.q_table)
            self.assertEqual(restored_agent.steps, trained_agent.steps)

    def test_generated_maze_has_clear_routes_from_amoeba_and_predator_spawns(self):
        walls, spots = main.make_world()
        maps = main.SensorMaps(walls, spots)
        amoeba_start = main._dark_start_position(maps, spots)
        predator_start = main._dark_start_position(maps, spots, amoeba_start)
        agent = main.Ameba()
        directions = ((1, 0), (-1, 0), (0, 1), (0, -1))

        for species, start in (("амёбы", amoeba_start), ("хищника", predator_start)):
            self.assertLess(maps.light_at(start, spots), main.FOOD_LIGHT_THRESHOLD)
            agent.set_start(start)
            start_cell = agent._cell_for_position(start)
            reachable = {start_cell}
            queue = deque([start_cell])

            while queue:
                cell_x, cell_y = queue.popleft()
                origin = pygame.Vector2(main._cell_center(cell_x, cell_y))
                for dx, dy in directions:
                    neighbor = (cell_x + dx, cell_y + dy)
                    if not (0 <= neighbor[0] < main.MAZE_COLS and 0 <= neighbor[1] < main.MAZE_ROWS):
                        continue
                    if neighbor in reachable:
                        continue
                    destination = pygame.Vector2(main._cell_center(*neighbor))
                    delta = destination - origin
                    agent.position.update(origin)
                    if agent._path_clear(walls, delta.normalize(), delta.length()):
                        reachable.add(neighbor)
                        queue.append(neighbor)

            for spot in spots:
                target_cell = agent._cell_for_position(spot.position)
                self.assertIn(
                    target_cell, reachable,
                    f"Нет прохода от старта {species} к ресурсу в клетке {target_cell}",
                )

    def test_predator_spawn_requires_routes_to_at_least_two_amoebas(self):
        walls, spots = main.make_world()
        maps = main.SensorMaps(walls, spots)
        amoebas = []
        for spot in spots[:4]:
            amoeba = main.Ameba()
            amoeba.position.update(spot.position)
            amoebas.append(amoeba)

        start = main._predator_spawn_position(walls, maps, spots, amoebas)
        self.assertIsNotNone(start)
        self.assertGreaterEqual(
            main._reachable_amoeba_count(walls, start, amoebas),
            main.MIN_REACHABLE_PREY_FOR_PREDATOR,
        )

    def test_amoeba_speed_advantage_decreases_with_hunger(self):
        self.agent.satiety = 100
        full_speed = self.agent._movement_speed(100)
        self.agent.satiety = 35
        hungry_speed = self.agent._movement_speed(100)
        self.agent.satiety = 0
        starving_speed = self.agent._movement_speed(100)
        self.assertEqual(full_speed, 100)
        self.assertGreater(full_speed, hungry_speed)
        self.assertGreater(hungry_speed, starving_speed)

    def test_amoeba_avoids_stronger_combined_neighbor_trails_and_crowds(self):
        center = pygame.Vector2(main.WORLD_WIDTH / 2, main.WORLD_HEIGHT / 2)
        first = main.Ameba()
        second = main.Ameba()
        for neighbor in (first, second):
            neighbor.position.update(center)
            neighbor.scent_clock = 0.0
            neighbor.scent_trail.clear()
            neighbor.scent_trail.append((center.copy(), 0.0))

        self.agent.position.update(center)
        self.agent.neighbor_agents = (first,)
        one_trail = self.agent._neighbor_scent_at(center)
        one_neighbor_density = self.agent._neighbor_density_at(center)
        self.agent.neighbor_agents = (first, second)

        self.assertGreater(self.agent._neighbor_scent_at(center), one_trail)
        self.assertGreater(self.agent._neighbor_density_at(center), one_neighbor_density)

    def test_amoeba_turns_away_from_neighbor_trail(self):
        class FlatMaps:
            def light_at(self, point, spots):
                return 0.0

            def heat_at(self, point, spots):
                return 0.0

        center = pygame.Vector2(main.WORLD_WIDTH / 2, main.WORLD_HEIGHT / 2)
        self.agent.position.update(center)
        self.agent.heading = 0.0
        self.agent.speed = 140.0
        self.agent.epsilon = 0.0
        neighbor = main.Ameba()
        neighbor.set_start(center + pygame.Vector2(50, 0))
        self.agent.neighbor_agents = (neighbor,)

        action = self.agent.choose_action((0,) * 8, [], [], FlatMaps(), 100, 260)

        self.assertIn(action, (1, 2))

    def test_predator_is_attracted_to_amoeba_trails_and_clusters(self):
        predator = main.Predator()
        prey = main.Ameba()
        other_prey = main.Ameba()
        prey.set_start((650, 300))
        prey.scent_clock = 10.0
        prey.scent_trail.clear()
        prey.scent_trail.append((pygame.Vector2(450, 300), 9.0))
        other_prey.set_start((650, 320))
        other_prey.scent_trail.clear()
        other_prey.scent_trail.append((pygame.Vector2(650, 320), 10.0))
        predator._scent_targets = (prey,)
        predator.prey_agents = [prey]
        predator._scent_cache.clear()
        one_prey_scent = predator._scent_at(pygame.Vector2(650, 300))
        one_prey_trail_scent = predator._scent_at(pygame.Vector2(450, 300))
        single_density = predator._prey_density_at(pygame.Vector2(650, 300))
        predator._scent_targets = (prey, other_prey)
        predator._scent_cache.clear()
        clustered_scent = predator._scent_at(pygame.Vector2(650, 300))
        clustered_density = predator._prey_density_at(pygame.Vector2(650, 300))

        self.assertGreater(one_prey_trail_scent, 0.0)
        self.assertGreater(clustered_scent, one_prey_scent)
        self.assertGreater(clustered_density, single_density)

    def test_predator_follows_amoeba_trail_corridor(self):
        class FlatMaps:
            def light_at(self, point, spots):
                return 0.0

            def heat_at(self, point, spots):
                return 0.0

        predator = main.Predator()
        center = pygame.Vector2(main.WORLD_WIDTH / 2, main.WORLD_HEIGHT / 2)
        predator.position.update(center)
        predator.heading = 0.0
        predator.speed = 140.0
        prey = main.Ameba()
        prey.set_start(center)
        prey.scent_clock = 1.0
        prey.scent_trail.clear()
        prey.scent_trail.append((center.copy(), 1.0))
        prey.scent_trail.append((center + pygame.Vector2(200, 0), 0.5))
        predator.prey_agents = [prey]

        action = predator.choose_action((0,) * predator.state_size, [], [], FlatMaps(), 100, 260)

        self.assertEqual(action, 0)

    def test_predator_scent_sensor_points_toward_amoeba(self):
        class FlatMaps:
            def light_at(self, point, spots):
                return 0.3

            def heat_at(self, point, spots):
                return 0.2

        predator = main.Predator()
        predator.heading = 0
        predator.position.update(main.WORLD_WIDTH / 2, main.WORLD_HEIGHT / 2)
        prey = main.Ameba()
        prey.set_start((predator.position.x + 300, predator.position.y))
        state = predator.sensors([], FlatMaps(), [], 100, 260, [prey])
        self.assertEqual(len(state), 11)
        self.assertEqual(state[8], 1)

    def test_predator_turns_toward_scent_bearing_instead_of_following_a_loop(self):
        class FlatMaps:
            def light_at(self, point, spots):
                return 0.3

            def heat_at(self, point, spots):
                return 0.2

        predator = main.Predator()
        predator.heading = 0.0
        center = pygame.Vector2(main.WORLD_WIDTH / 2, main.WORLD_HEIGHT / 2)
        predator.position.update(center)
        prey = main.Ameba()
        prey.set_start(center - pygame.Vector2(300, 300))
        predator.prey_agents = [prey]

        action = predator.choose_action((0,) * predator.state_size, [], [], FlatMaps(), 100, 260)

        self.assertEqual(action, 1)

    def test_predator_search_without_prey_scent_does_not_sample_food_fields(self):
        class UnusedMaps:
            def light_at(self, point, spots):
                raise AssertionError("predator search should not sample light")

            def heat_at(self, point, spots):
                raise AssertionError("predator search should not sample heat")

        predator = main.Predator()
        predator.position.update(main.WORLD_WIDTH / 2, main.WORLD_HEIGHT / 2)
        predator.epsilon = 0.0

        action = predator.choose_action(
            (0,) * predator.state_size, [], [], UnusedMaps(), 100, 260,
        )

        self.assertIn(action, (0, 1, 2))

    def test_predator_reduces_scent_priority_for_its_own_trail(self):
        class FlatMaps:
            def light_at(self, point, spots):
                return 0.3

            def heat_at(self, point, spots):
                return 0.2

        predator = main.Predator()
        predator.heading = 0.0
        center = pygame.Vector2(main.WORLD_WIDTH / 2, main.WORLD_HEIGHT / 2)
        predator.position.update(center)
        prey = main.Ameba()
        prey.set_start(center + pygame.Vector2(300, 0))
        predator.prey_agents = [prey]
        predator._lookahead_cell = lambda walls, direction, distance: (
            (1, 1) if abs(direction.y) < 0.1 else
            ((2, 1) if direction.y < 0 else (3, 1))
        )
        predator.visited_cells[(1, 1)] = 4

        action = predator.choose_action(
            (0,) * predator.state_size, [], [], FlatMaps(), 100, 260,
        )

        self.assertIn(action, (1, 2))

    def test_predator_scent_is_bounded_decays_and_does_not_cross_walls(self):
        predator = main.Predator()
        prey = main.Ameba()
        predator.position.update(200, 200)
        prey.set_start((800, 200))
        predator.prey_agents = [prey]

        strong_scent = predator._scent_at(pygame.Vector2(500, 200))
        weak_scent = predator._scent_at(pygame.Vector2(200, 200))
        self.assertGreater(strong_scent, weak_scent)
        beyond_scent_reach = prey.position.x + main.AMOEBA_SCENT_REACH + 1
        self.assertEqual(
            predator._scent_at(pygame.Vector2(beyond_scent_reach, 200)), 0.0,
        )

        predator.scent_walls = [pygame.Rect(490, 150, 10, 100)]
        predator.scent_blockers = tuple(
            wall.inflate(predator.radius * 2, predator.radius * 2)
            for wall in predator.scent_walls
        )
        self.assertEqual(predator._scent_at(pygame.Vector2(200, 200)), 0.0)

    def test_predator_scent_grid_blocks_walls_and_keeps_clear_lines(self):
        predator = main.Predator()
        predator.scent_walls = [pygame.Rect(490, 150, 10, 100)]
        predator.scent_blockers = tuple(
            wall.inflate(predator.radius * 2, predator.radius * 2)
            for wall in predator.scent_walls
        )
        predator._index_scent_blockers()

        self.assertTrue(predator._scent_blocked(
            pygame.Vector2(200, 200), pygame.Vector2(700, 200),
        ))
        self.assertFalse(predator._scent_blocked(
            pygame.Vector2(200, 100), pygame.Vector2(700, 100),
        ))

    def test_predator_lays_egg_after_three_catches(self):
        class FlatMaps:
            def light_at(self, point, spots):
                return 0.2

            def heat_at(self, point, spots):
                return 0.2

        predator = main.Predator()
        prey = main.Ameba()
        predator.position.update(400, 400)
        prey.set_start((400, 400))
        predator.kills_since_egg = 2
        settings = (100, 260, 10, 14, 5, 8, 2)
        predator.update(0.14, [], [], FlatMaps(), settings, [prey])
        for _ in range(math.ceil(main.PREDATOR_EAT_SECONDS / 0.14)):
            predator.update(0.14, [], [], FlatMaps(), settings, [prey])
        self.assertTrue(prey.dead)
        self.assertEqual(predator.total_eggs_laid, 1)
        self.assertEqual(predator.hatch_eggs_pending, 1)
        self.assertEqual(predator.episode, 2)

    def test_predator_stays_with_prey_while_eating_for_three_seconds(self):
        class FlatMaps:
            def light_at(self, point, spots):
                return 0.2

            def heat_at(self, point, spots):
                return 0.2

        predator = main.Predator()
        predator.position.update(400, 300)
        prey = main.Ameba()
        prey.set_start((400, 300))
        other_prey = main.Ameba()
        other_prey.set_start((600, 300))
        settings = (100, 260, 10, 14, 5, 8, 2)
        prey_agents = [prey, other_prey]

        predator.update(0.14, [], [], FlatMaps(), settings, prey_agents)
        eating_position = predator.position.copy()
        self.assertTrue(prey.being_eaten)
        self.assertFalse(prey.dead)

        for _ in range(29):
            prey.update(0.1, [], [], None, settings)
            predator.update(0.1, [], [], FlatMaps(), settings, prey_agents)

        self.assertEqual(predator.position, eating_position)
        self.assertFalse(prey.dead)
        self.assertFalse(other_prey.dead)
        self.assertEqual(predator.total_kills, 0)

        predator.update(0.1, [], [], FlatMaps(), settings, prey_agents)

        self.assertTrue(prey.dead)
        self.assertEqual(predator.total_kills, 1)
        self.assertFalse(other_prey.dead)

    def test_predator_escapes_wall_instead_of_staying_blocked(self):
        class FlatMaps:
            def light_at(self, point, spots):
                return 0.2

            def heat_at(self, point, spots):
                return 0.2

        predator = main.Predator()
        predator.position.update(200, 200)
        predator.heading = 0.0
        walls = [
            pygame.Rect(220, 150, 8, 100),
            pygame.Rect(150, 170, 70, 6),
            pygame.Rect(150, 224, 70, 6),
        ]

        moved = predator._escape_wall(walls, [], FlatMaps(), 100, 260, 25)

        self.assertTrue(moved)
        self.assertLess(predator.position.x, 200)

    def test_bumper_avoids_reverse_when_a_side_route_is_open(self):
        class FlatMaps:
            def light_at(self, point, spots):
                return 0.2

            def heat_at(self, point, spots):
                return 0.2

        self.agent.position.update(200, 200)
        self.agent.heading = 0.0
        self.agent.trail.clear()
        self.agent.trail.extend((pygame.Vector2(180, 200), pygame.Vector2(200, 200)))
        self.agent._path_clear = lambda walls, direction, distance: direction.y < -0.7

        moved = self.agent._escape_wall([], [], FlatMaps(), 100, 260, 25)

        self.assertTrue(moved)
        self.assertLess(self.agent.position.y, 200)

    def test_bumper_prefers_a_cell_without_blue_trail_memory(self):
        class FlatMaps:
            def light_at(self, point, spots):
                return 0.2

            def heat_at(self, point, spots):
                return 0.2

        self.agent.position.update(200, 200)
        self.agent.heading = 0.0
        self.agent.trail.clear()
        self.agent.trail.extend((pygame.Vector2(180, 200), pygame.Vector2(200, 200)))
        self.agent._path_clear = lambda walls, direction, distance: abs(direction.y) > 0.7
        self.agent._lookahead_cell = lambda walls, direction, distance: (
            (1, 1) if direction.y > 0 else (2, 2)
        )
        self.agent.visited_cells[(1, 1)] = 3

        moved = self.agent._escape_wall([], [], FlatMaps(), 100, 260, 25)

        self.assertTrue(moved)
        self.assertLess(self.agent.position.y, 200)

    def test_route_memories_are_private_for_amoebas_and_predators(self):
        for agent_type in (main.Ameba, main.Predator):
            with self.subTest(agent_type=agent_type.__name__):
                first = agent_type()
                second = agent_type()
                first.position.update(main._cell_center(2, 2))
                second.position.update(main._cell_center(3, 2))
                first._record_visit(first.position)

                first_cell = first._cell_for_position(first.position)
                self.assertIn(first_cell, first.visited_cells)
                self.assertNotIn(first_cell, second.visited_cells)
                self.assertIsNot(first.visited_cells, second.visited_cells)
                self.assertIsNot(first.trail, second.trail)

    def test_lookahead_does_not_skip_visited_cells_on_the_way_to_new_ones(self):
        self.agent.position.update(
            main.MAZE_OFFSET_X + 2 * main.CELL_SIZE + main.CELL_SIZE / 2,
            main.MAZE_OFFSET_Y + main.CELL_SIZE / 2,
        )
        current_cell = self.agent._cell_for_position(self.agent.position)
        next_cell = (current_cell[0] + 1, current_cell[1])
        self.agent.visited_cells[next_cell] = 1

        target_cell = self.agent._lookahead_cell(
            [], pygame.Vector2(1, 0), main.CELL_SIZE * 3,
        )

        self.assertEqual(target_cell, next_cell)

    def test_bumper_escape_runs_after_partial_movement_into_a_wall(self):
        class FlatMaps:
            def light_at(self, point, spots):
                return 0.2

            def heat_at(self, point, spots):
                return 0.2

        self.agent.sensors = lambda *args: (4, 4, 0, 0, 0, 0, 0, 3)
        self.agent.choose_action = lambda *args: 0
        self.agent._food_at_position = lambda spots, maps: None
        self.agent._learning_potential = lambda *args, **kwargs: 0.0

        def partial_move(walls, direction, distance):
            self.agent.position.x += 2
            self.agent.bumper = True
            return True

        escaped = []

        def escape(walls, spots, maps, light_radius, heat_radius, distance):
            escaped.append(True)
            self.assertEqual(self.agent.trail[-1], self.agent.position)
            self.agent.position.y += 2
            return True

        self.agent._move_forward = partial_move
        self.agent._escape_wall = escape

        self.agent.update(0.14, [], [], FlatMaps(), (100, 260, 143, 14, 3.5, 8, 2))

        self.assertEqual(escaped, [True])
        self.assertEqual(len(self.agent.trail), 3)

    def test_predator_hunger_depletes_faster(self):
        class FlatMaps:
            def light_at(self, point, spots):
                return 0.2

            def heat_at(self, point, spots):
                return 0.2

        predator = main.Predator()
        predator.update(0.1, [], [], FlatMaps(), (100, 260, 110, 14, 5, 8, 2), [])

        self.assertAlmostEqual(predator.satiety, 99.8)

    def test_predator_runtime_configuration_is_applied(self):
        class FlatMaps:
            def light_at(self, point, spots):
                return 0.2

            def heat_at(self, point, spots):
                return 0.2

        predator = main.Predator()
        settings = (*main.SETTING_DEFAULTS[:7], 0.9, 0.6, 600.0, 1.5, 2.0,
                    4.5, 9.0, 5.0, 400.0, 4.0, 3.0, 2.5, 5.0, 4.0)

        predator.update(0.0, [], [], FlatMaps(), settings, [])

        self.assertAlmostEqual(predator.speed, settings[2] * settings[7] * predator._speed_scale())
        self.assertEqual(predator.speed_ratio, settings[7])
        self.assertEqual(predator.hunger_drain_ratio, settings[8])
        self.assertEqual(predator.scent_reach, settings[9])
        self.assertEqual(predator.eating_duration, settings[10])
        self.assertEqual(predator.kills_per_egg, int(settings[11]))
        self.assertEqual(predator.prey_scent_weight, settings[16])
        self.assertEqual(predator.prey_trail_weight, settings[17])
        self.assertEqual(predator.prey_crowding_weight, settings[18])
        self.assertEqual(predator.predator_trail_avoidance, settings[19])
        self.assertEqual(predator.predator_crowding_avoidance, settings[20])

    def test_hunger_increases_predator_exploration_and_aggression(self):
        predator = main.Predator()
        predator.satiety = 100
        fed_revisit_weight = predator._exploration_revisit_weight()
        fed_aggression = predator._aggression_level()
        predator.satiety = 25
        hungry_revisit_weight = predator._exploration_revisit_weight()
        hungry_aggression = predator._aggression_level()

        self.assertGreater(hungry_revisit_weight, fed_revisit_weight)
        self.assertGreater(hungry_aggression, fed_aggression)


if __name__ == "__main__":
    unittest.main()
