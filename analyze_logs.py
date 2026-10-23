"""Summarize simulation JSONL telemetry for behavior and tuning reviews."""

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path


DEFAULT_LOG_DIR = Path(__file__).resolve().parent / "simulation_logs"


def read_records(path):
    records = []
    errors = []
    with path.open("r", encoding="utf-8") as log_file:
        for line_number, line in enumerate(log_file, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                errors.append(f"строка {line_number}: {error.msg}")
    return records, errors


def _population_totals(snapshot):
    return sum(snapshot.get("population", {}).values())


def summarize(path):
    records, parse_errors = read_records(path)
    by_event = Counter(record.get("event", "unknown") for record in records)
    starts = [record for record in records if record.get("event") == "run_start"]
    snapshots = [record for record in records if record.get("event") == "snapshot"]
    cycles = [record for record in records if record.get("event") == "pretrain_cycle"]
    endings = [record for record in records if record.get("event") == "run_end"]
    settings_changes = [record for record in records if record.get("event") == "settings_changed"]
    deaths = Counter(
        f"{record.get('species', 'unknown')}:{record.get('cause', 'unknown')}"
        for record in records if record.get("event") == "death"
    )

    final_snapshot = snapshots[-1] if snapshots else None
    first_snapshot = snapshots[0] if snapshots else None
    final_cycle = cycles[-1] if cycles else None
    warnings = []

    if cycles and final_cycle:
        if final_cycle.get("predator_sessions", 0) == 0:
            warnings.append("предобучение не записало ни одной успешной сессии хищника")
        if final_cycle.get("amoeba_sessions", 0) == 0:
            warnings.append("предобучение не записало ни одной успешной сессии амёбы")

    if snapshots:
        total = _population_totals(final_snapshot)
        if total == 0:
            warnings.append("последний снимок показывает пустую популяцию")
        if first_snapshot and _population_totals(first_snapshot) > 0:
            first_total = _population_totals(first_snapshot)
            if total < first_total * 0.25:
                warnings.append("популяция сократилась более чем на 75% от первого снимка")
        route_visits = final_snapshot.get("route_visits", {})
        visits = route_visits.get("total", 0)
        revisits = route_visits.get("revisits", 0)
        if visits >= 100 and revisits / visits >= 0.75:
            warnings.append("повторные посещения занимают не менее 75% учтённых маршрутов")

    return {
        "file": str(path),
        "seed": starts[0].get("seed") if starts else None,
        "status": endings[-1].get("status") if endings else "нет run_end",
        "wall_seconds": max((record.get("wall_s", 0) for record in records), default=0),
        "simulated_seconds": max((record.get("sim_s", 0) for record in records), default=0),
        "pretrain_cycles": len(cycles),
        "pretrain_final": final_cycle,
        "settings_changes": [record.get("settings") for record in settings_changes],
        "event_counts": dict(by_event),
        "death_causes": dict(deaths),
        "first_snapshot": first_snapshot,
        "final_snapshot": final_snapshot,
        "warnings": warnings,
        "parse_errors": parse_errors,
        "snapshots": snapshots,
    }


def _mean_satiety(genotypes):
    agents = sum(item.get("count", 0) for item in genotypes.values())
    if not agents:
        return None
    return round(sum(item.get("mean_satiety", 0) * item.get("count", 0)
                     for item in genotypes.values()) / agents, 2)


def write_csv(path, summaries):
    columns = (
        "run_file", "sim_s", "wall_s", "amoebas", "predators", "amoeba_eggs",
        "predator_eggs", "food_total", "amoeba_genotypes", "predator_genotypes",
        "amoeba_mean_satiety", "predator_mean_satiety", "route_revisits", "route_visits",
    )
    with path.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        for summary in summaries:
            for snapshot in summary["snapshots"]:
                amoebas = snapshot.get("amoeba_genotypes", {})
                predators = snapshot.get("predator_genotypes", {})
                population = snapshot.get("population", {})
                eggs = snapshot.get("eggs", {})
                routes = snapshot.get("route_visits", {})
                writer.writerow({
                    "run_file": summary["file"],
                    "sim_s": snapshot.get("sim_s"),
                    "wall_s": snapshot.get("wall_s"),
                    "amoebas": population.get("amoeba", 0),
                    "predators": population.get("predator", 0),
                    "amoeba_eggs": eggs.get("amoeba", 0),
                    "predator_eggs": eggs.get("predator", 0),
                    "food_total": snapshot.get("food_total"),
                    "amoeba_genotypes": len(amoebas),
                    "predator_genotypes": len(predators),
                    "amoeba_mean_satiety": _mean_satiety(amoebas),
                    "predator_mean_satiety": _mean_satiety(predators),
                    "route_revisits": routes.get("revisits", 0),
                    "route_visits": routes.get("total", 0),
                })


def print_report(summary):
    print(f"\n{Path(summary['file']).name} · seed {summary['seed']} · {summary['status']}")
    print(f"Время: {summary['wall_seconds']:.1f} с реального / "
          f"{summary['simulated_seconds']:.1f} с симуляции")
    if summary["pretrain_cycles"]:
        result = summary["pretrain_final"] or {}
        print(f"Предобучение: {summary['pretrain_cycles']} циклов; "
              f"яйца амёб {result.get('amoeba_eggs', 0)}, "
              f"сессии амёб {result.get('amoeba_sessions', 0)}; "
              f"яйца хищников {result.get('predator_eggs', 0)}, "
              f"сессии хищников {result.get('predator_sessions', 0)}")
    print("События: " + ", ".join(
        f"{name} {count}" for name, count in sorted(summary["event_counts"].items())
    ))

    snapshot = summary["final_snapshot"]
    if snapshot:
        population = snapshot.get("population", {})
        routes = snapshot.get("route_visits", {})
        print(f"Последний снимок ({snapshot.get('sim_s', 0):.1f} с): "
              f"амёбы {population.get('amoeba', 0)}, хищники {population.get('predator', 0)}, "
              f"еда {snapshot.get('food_total', 0):.1f}")
        print(f"Генотипы: амёбы {len(snapshot.get('amoeba_genotypes', {}))}, "
              f"хищники {len(snapshot.get('predator_genotypes', {}))}; "
              f"повторные маршруты {routes.get('revisits', 0)}/"
              f"{routes.get('total', 0)}")
        for species in ("amoeba", "predator"):
            genotypes = snapshot.get(f"{species}_genotypes", {})
            dominant = max(genotypes.values(), key=lambda item: item.get("count", 0), default=None)
            if dominant:
                print(f"  {species}: ведущий генотип "
                      f"{dominant['speed_gene']}:{dominant['sense_gene']} "
                      f"({dominant['count']} агентов), средняя скорость "
                      f"{dominant['mean_speed']}, сытость {dominant['mean_satiety']}")

    for warning in summary["warnings"]:
        print(f"Проверить: {warning}")
    for error in summary["parse_errors"]:
        print(f"Ошибка чтения: {error}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", type=Path, default=DEFAULT_LOG_DIR,
                        help="JSONL-файл или папка логов (по умолчанию simulation_logs)")
    parser.add_argument("--json", action="store_true", dest="as_json",
                        help="вывести полную сводку и временные ряды в JSON")
    parser.add_argument("--csv", type=Path, metavar="FILE",
                        help="экспортировать снимки популяции в CSV")
    args = parser.parse_args()

    if args.path.is_file():
        paths = [args.path]
    elif args.path.is_dir():
        paths = sorted(args.path.glob("*.jsonl"))
    else:
        paths = []
    if not paths:
        print(f"Логи не найдены: {args.path}", file=sys.stderr)
        return 2

    summaries = [summarize(path) for path in paths]
    if args.csv:
        write_csv(args.csv, summaries)
        print(f"CSV записан: {args.csv}")
    if args.as_json:
        print(json.dumps(summaries, ensure_ascii=False, indent=2))
    else:
        for summary in summaries:
            print_report(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
