"""EXP-001: do data-derived inputs make the simulated Monza 2024 resemble the real race?

Compares two configurations over the same seeds (paired):
  A. synthetic  - everyone on MEDIUM, equal cars (engine defaults)
  B. data       - real start compounds + team car-pace offsets estimated from real laps

Metrics vs the official classification (classified finishers only):
  - Spearman rank correlation of finishing order
  - mean absolute position error
  - winner's team, and how often the real podium teams fill the simulated podium

Usage:
    uv run python scripts/experiments/exp001_data_inputs.py --seeds 30

Caveat printed with the results: the car offsets are estimated from this same race (in-sample),
and the engine still has no pit stops, traffic or lapping, so this is a sanity check, not a
validation of predictive skill.
"""

import argparse
import statistics
import time
from pathlib import Path

import pandas as pd

from f1ar.domain.race_state import RaceSetup, WorldStatus
from f1ar.simulation import lightweight_engine as engine
from f1ar.simulation.scenarios import MONZA_2024, event_data_dir, load_event, setup_from_codes

ROOT = Path(__file__).resolve().parents[2]


def spearman(a: list[str], b: list[str]) -> float:
    rank_b = {d: i for i, d in enumerate(b)}
    n = len(a)
    d2 = sum((i - rank_b[d]) ** 2 for i, d in enumerate(a))
    return 1 - 6 * d2 / (n * (n * n - 1))


def run(setup: RaceSetup, seed: int) -> list[str]:
    state = engine.start(engine.create_world(setup, seed=seed, world_id="exp001"))
    while state.world_status is WorldStatus.RUNNING:
        state = engine.step(state).state
    return [setup.driver(d).code for d in engine.standings(state)]


def main() -> None:
    parser = argparse.ArgumentParser(description="EXP-001 data inputs vs synthetic")
    parser.add_argument("--seeds", type=int, default=30)
    args = parser.parse_args()

    data_dir = event_data_dir(ROOT, MONZA_2024)
    if data_dir is None:
        raise SystemExit("fetch the data first: uv run python scripts/fetch_monza_2024.py")
    roster, profiles = load_event(ROOT / "configs")
    results = pd.read_parquet(data_dir / "results.parquet")
    classified = results[results["status"].isin(["Finished", "Lapped"])].sort_values("position")
    real = [str(c) for c in classified["driver_code"]]
    team_of = {e.code: e.team_id for e in roster.entries}
    real_podium_teams = sorted(team_of[c] for c in real[:3])

    configs = {
        "A synthetic": setup_from_codes(roster, profiles, None),
        "B data": setup_from_codes(roster, profiles, None, data_dir=data_dir),
    }
    print(f"EXP-001 | {MONZA_2024} | {args.seeds} paired seeds | engine {engine.ENGINE_VERSION}")
    print(f"real classified order ({len(real)} cars): {' '.join(real)}\n")
    summary = {}
    for name, setup in configs.items():
        t0 = time.perf_counter()
        rhos, maes, winners, podium_hits = [], [], [], 0
        for seed in range(1, args.seeds + 1):
            order = [c for c in run(setup, seed) if c in real]
            rhos.append(spearman(order, real))
            maes.append(statistics.fmean(abs(order.index(c) - real.index(c)) for c in real))
            winners.append(team_of[order[0]])
            podium_hits += sorted(team_of[c] for c in order[:3]) == real_podium_teams
        summary[name] = (rhos, maes)
        top = max(set(winners), key=winners.count)
        print(
            f"{name:<12} spearman {statistics.fmean(rhos):+.3f} "
            f"+/- {statistics.pstdev(rhos):.3f} | "
            f"mean |pos err| {statistics.fmean(maes):.2f} | most common winner team {top} "
            f"({winners.count(top)}/{args.seeds}) | real podium teams {podium_hits}/{args.seeds} "
            f"| {time.perf_counter() - t0:.0f}s"
        )
        print(f"{'':12} example order (seed 1): {' '.join(run(setup, 1))}")
    better = sum(
        b > a for a, b in zip(summary["A synthetic"][0], summary["B data"][0], strict=True)
    )
    print(f"\npaired: B beats A on rank correlation in {better}/{args.seeds} seeds")
    print("caveat: car offsets are estimated from this same race (in-sample); engine has no pit")
    print("stops, traffic or lapping -> sanity check of the inputs, not a predictive validation.")


if __name__ == "__main__":
    main()
