"""Run a headless race and print a lap-by-lap timing sheet.

Usage:
    uv run python scripts/demo_headless.py                     # VER vs HAM, seed 42
    uv run python scripts/demo_headless.py --drivers VER HAM LEC NOR --seed 7
    uv run python scripts/demo_headless.py --all               # full 20-car grid

The race model is a synthetic placeholder (no pit stops, no traffic, uncalibrated).
"""

import argparse
from pathlib import Path

from f1ar.domain.race_state import WorldStatus
from f1ar.simulation import lightweight_engine as engine
from f1ar.simulation.scenarios import MONZA_2024, load_event, setup_from_codes
from f1ar.simulation.timing import LapTimingSheet

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def main() -> None:
    parser = argparse.ArgumentParser(description="Headless race demo")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--drivers", nargs="+", default=["VER", "HAM"], metavar="CODE")
    parser.add_argument("--all", action="store_true", help="race the full 20-car grid")
    parser.add_argument("--every", type=int, default=1, help="print every Nth lap")
    args = parser.parse_args()

    roster, profiles = load_event(CONFIGS)
    try:
        setup = setup_from_codes(roster, profiles, None if args.all else args.drivers)
    except ValueError as exc:
        parser.error(str(exc))
    state = engine.start(engine.create_world(setup, seed=args.seed, world_id="demo"))
    code_of = {d.driver_id: d.code for d in setup.drivers}
    total = setup.track.total_laps

    print(f"{MONZA_2024} | {len(setup.drivers)} cars | {total} laps | seed {args.seed}")
    print("grid: " + "  ".join(f"P{d.grid_position} {d.code}" for d in setup.drivers))
    print(f"{engine.ENGINE_VERSION}: synthetic lap model, no pit stops, no traffic\n")

    sheet = LapTimingSheet(code_of, total, every=args.every)
    while state.world_status is WorldStatus.RUNNING:
        result = engine.step(state)
        state = result.state
        for row in sheet.feed(result.events):
            print(row)

    print("\nFinal classification")
    winner_t = state.car(engine.standings(state)[0]).finish_time_s or 0.0
    for pos, driver_id in enumerate(engine.standings(state), start=1):
        t = state.car(driver_id).finish_time_s or 0.0
        gap = f"{t // 60:.0f}:{t % 60:06.3f}" if pos == 1 else f"+{t - winner_t:.3f}s"
        grid = setup.driver(driver_id).grid_position
        print(f"  P{pos:<2} {code_of[driver_id]}  {gap:>12}   (started P{grid})")
    print(f"\nticks {state.sim_tick} | events {state.event_seq} | state {state.canonical_hash()}")


if __name__ == "__main__":
    main()
