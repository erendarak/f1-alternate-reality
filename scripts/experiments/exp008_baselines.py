"""EXP-008: how good are the baselines the learned AI must beat? (M5 slice 2, D-039)

Cases: the 5 dry test races x 4 cars (grid slots ``--slots``, default P1/P5/P10/P15) x seeds.
Each case is played by every method on the same race (paired): B1 rules (the engine's
planner), B0 the real driver's 2024 stops, B2 enumerator, B3 MPC, and random. One AI car per
race; the other 19 cars follow the rules. Scoring: the team principal's objective without a
championship context (points) under the driver's risk utility.

Reported: mean reward / points / position per method, paired differences vs B1 with a bootstrap
95 % interval over cases, stop counts, compute time per race.

Usage:
    uv run python scripts/experiments/exp008_baselines.py --seeds 5 --workers 14 \\
        --json docs/experiments/exp008.json
"""

import argparse
import json
import statistics
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np

from f1ar.domain.race_state import RaceSetup
from f1ar.policy.strategists import (
    Enumerator,
    FixedPlan,
    MonteCarlo,
    RandomStrategist,
    Strategist,
    historical_plan,
)
from f1ar.training.evaluate import evaluate
from f1ar.training.objectives import scoring_for
from f1ar.training.scenarios import repo_setup_factory

ROOT = Path(__file__).resolve().parents[2]
EVENTS = (
    "2024_italian_gp",
    "2024_japanese_gp",
    "2024_monaco_gp",
    "2024_mexico_city_gp",
    "2024_abu_dhabi_gp",
)
METHODS = ("rules", "historical", "enumerator", "mpc", "random")


@cache
def _setup(event_id: str) -> RaceSetup:
    return repo_setup_factory(ROOT)(event_id, False)


def strategist(method: str, event_id: str, code: str, seed: int) -> Strategist | None:
    match method:
        case "rules":
            return None
        case "historical":
            plan = historical_plan(ROOT / "data" / "processed" / event_id, code)
            return FixedPlan(plan, "historical")
        case "enumerator":
            return Enumerator()
        case "mpc":
            return MonteCarlo()
        case "random":
            return RandomStrategist(seed)
    raise ValueError(method)


def run_case(case: tuple[str, str, int, str]) -> dict[str, Any]:
    event_id, code, seed, method = case
    setup = _setup(event_id)
    driver = next(d for d in setup.drivers if d.code == code)
    scoring = scoring_for(setup, [driver.driver_id], None)[driver.driver_id]
    s = strategist(method, event_id, code, seed)
    result = asdict(evaluate(setup, driver.driver_id, seed, scoring, s))
    result["code"] = code
    result["grid"] = driver.grid_position
    if isinstance(s, MonteCarlo):
        result["rollouts"] = s.rollouts
        result["deliberations"] = s.deliberations
    return result


def bootstrap(diffs: list[float], n: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    a = np.array(diffs)
    means = rng.choice(a, size=(n, len(a)), replace=True).mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--slots", type=int, nargs="+", default=[1, 5, 10, 15])
    parser.add_argument("--workers", type=int, default=14)
    parser.add_argument("--methods", nargs="+", default=list(METHODS))
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    t0 = time.perf_counter()

    cases = []
    for event_id in EVENTS:
        by_grid = {d.grid_position: d.code for d in _setup(event_id).drivers}
        for slot in args.slots:
            for seed in range(1, args.seeds + 1):
                for method in args.methods:
                    cases.append((event_id, by_grid[slot], seed, method))
    # slow cases first so the pool stays busy to the end
    cases.sort(key=lambda c: c[3] != "mpc")
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, row in enumerate(pool.map(run_case, cases, chunksize=1), start=1):
            rows.append(row)
            if i % 10 == 0 or i == len(cases):
                elapsed = time.perf_counter() - t0
                print(f"{i}/{len(cases)} cases done, {elapsed / 60:.1f} min", flush=True)

    by = {(r["event_id"], r["code"], r["seed"], r["method"]): r for r in rows}
    keys = sorted({(r["event_id"], r["code"], r["seed"]) for r in rows})
    summary: dict[str, Any] = {}
    for method in args.methods:
        mine = [by[(*k, method)] for k in keys]
        entry: dict[str, Any] = {
            "reward": round(statistics.fmean(r["reward"] for r in mine), 4),
            "points": round(statistics.fmean(r["points"] for r in mine), 2),
            "position": round(statistics.fmean(r["position"] for r in mine), 2),
            "dsq_or_dnf": sum(r["disqualified"] or not r["finished"] for r in mine),
            "stops": dict(sorted(Counter(r["stops"] for r in mine).items())),
            "seconds_per_race": round(statistics.fmean(r["seconds"] for r in mine), 2),
        }
        if "rules" in args.methods and method != "rules":
            for metric in ("reward", "points", "position"):
                diffs = [by[(*k, method)][metric] - by[(*k, "rules")][metric] for k in keys]
                lo, hi = bootstrap(diffs)
                entry[f"{metric}_vs_rules"] = {
                    "mean": round(statistics.fmean(diffs), 4),
                    "ci95": [round(lo, 4), round(hi, 4)],
                    "better": sum(d > 0 for d in diffs)
                    if metric != "position"
                    else sum(d < 0 for d in diffs),
                    "worse": sum(d < 0 for d in diffs)
                    if metric != "position"
                    else sum(d > 0 for d in diffs),
                }
        summary[method] = entry
    per_race: dict[str, Any] = {}
    for event_id in EVENTS:
        per_race[event_id] = {
            m: round(
                statistics.fmean(
                    r["reward"] for r in rows if r["event_id"] == event_id and r["method"] == m
                ),
                4,
            )
            for m in args.methods
        }
    print(json.dumps({"summary": summary, "per_race_reward": per_race}, indent=1))
    if args.json:
        out = {
            "cases": len(keys),
            "seeds": args.seeds,
            "slots": args.slots,
            "events": EVENTS,
            "summary": summary,
            "per_race_reward": per_race,
            "rows": rows,
            "seconds": round(time.perf_counter() - t0, 1),
        }
        args.json.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
