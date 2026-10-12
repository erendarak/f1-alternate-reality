"""Batch headless benchmark: how many full races per hour can the engine run? (plan §16 M2)

Runs complete 20-car races (pit stops, traffic, incidents, VSC/SC, data-derived inputs) with no
pacing, rendering or persistence, first sequentially, then in parallel worker processes, and
reports seconds per race, ticks per second and races per hour. M5 (RL training) needs this:
the plan's floor is 100 episodes (races) per day.

Timings depend heavily on the machine and on what else is running; the script prints the CPU
count and Python version with the results, and EXP-002 / PROGRESS record the run they quote.

Usage:
    uv run python scripts/benchmark.py                  # 10 races sequential + 40 parallel
    uv run python scripts/benchmark.py --races 20 --parallel 80 --workers 8
"""

import argparse
import os
import platform
import statistics
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from f1ar.domain.race_state import RaceSetup, WorldStatus
from f1ar.simulation import lightweight_engine as engine
from f1ar.simulation.scenarios import MONZA_2024, event_data_dir, load_event, setup_from_codes

ROOT = Path(__file__).resolve().parents[1]


def make_setup(event_id: str) -> RaceSetup:
    roster, profiles = load_event(ROOT / "configs", event_id)
    return setup_from_codes(roster, profiles, None, data_dir=event_data_dir(ROOT, event_id))


def race(args: tuple[str, int]) -> tuple[float, int]:
    """One full race; returns (seconds, ticks). Builds its own setup in worker processes."""
    event_id, seed = args
    setup = _SETUPS.get(event_id) or _SETUPS.setdefault(event_id, make_setup(event_id))
    t0 = time.perf_counter()
    state = engine.start(engine.create_world(setup, seed=seed, world_id=f"bench_{seed}"))
    while state.world_status is WorldStatus.RUNNING:
        state = engine.step(state).state
    return time.perf_counter() - t0, state.sim_tick


_SETUPS: dict[str, RaceSetup] = {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch headless race benchmark")
    parser.add_argument("--event", default=MONZA_2024)
    parser.add_argument("--races", type=int, default=10, help="sequential races")
    parser.add_argument("--parallel", type=int, default=40, help="races across worker processes")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    args = parser.parse_args()

    print(
        f"{engine.ENGINE_VERSION} | {args.event} | 20 cars | Python {platform.python_version()} "
        f"| {os.cpu_count()} logical CPUs | {platform.processor() or platform.machine()}"
    )
    make_setup(args.event)  # load data once (also warms the file cache)

    runs = [race((args.event, seed)) for seed in range(1, args.races + 1)]
    seconds = [s for s, _ in runs]
    ticks = statistics.fmean(t for _, t in runs)
    per_race = statistics.fmean(seconds)
    print(
        f"sequential: {args.races} races, {per_race:.2f} s/race "
        f"(min {min(seconds):.2f}, max {max(seconds):.2f}), {ticks / per_race:,.0f} ticks/s, "
        f"{3600 / per_race:,.0f} races/hour on one core"
    )

    if args.parallel > 0 and args.workers > 1:
        t0 = time.perf_counter()
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            jobs = [(args.event, seed) for seed in range(1, args.parallel + 1)]
            list(pool.map(race, jobs, chunksize=max(1, args.parallel // (4 * args.workers))))
        wall = time.perf_counter() - t0
        print(
            f"parallel:   {args.parallel} races on {args.workers} processes in {wall:.1f} s "
            f"-> {3600 * args.parallel / wall:,.0f} races/hour "
            f"(incl. worker start-up and data loading)"
        )
    print("plan floor for M5 training: 100 races/day")


if __name__ == "__main__":
    main()
