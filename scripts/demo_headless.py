"""Run a headless race and print a lap-by-lap timing sheet.

Usage:
    uv run python scripts/demo_headless.py                     # VER vs HAM, seed 42
    uv run python scripts/demo_headless.py --drivers VER HAM LEC NOR --seed 7
    uv run python scripts/demo_headless.py --all               # full 20-car grid

The race model is a synthetic placeholder (no pit stops, no traffic, uncalibrated).
"""

import argparse
from collections import defaultdict
from pathlib import Path

from f1ar.domain.driver_profiles import load_profiles
from f1ar.domain.identities import load_roster
from f1ar.domain.race_state import WorldStatus
from f1ar.simulation import lightweight_engine as engine
from f1ar.simulation.scenarios import build_setup

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
EVENT_ID = "2024_italian_gp"


def main() -> None:
    parser = argparse.ArgumentParser(description="Headless race demo")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--drivers", nargs="+", default=["VER", "HAM"], metavar="CODE")
    parser.add_argument("--all", action="store_true", help="race the full 20-car grid")
    parser.add_argument("--every", type=int, default=1, help="print every Nth lap")
    args = parser.parse_args()

    roster = load_roster(CONFIGS / "rosters" / f"{EVENT_ID}.json", expected_event_id=EVENT_ID)
    profiles = load_profiles(
        CONFIGS / "drivers" / "seed_2024_monza_drivers.yaml",
        CONFIGS / "drivers" / "archetypes.yaml",
        roster,
    )
    by_code = {e.code: e.driver_id for e in roster.entries}
    codes = [e.code for e in roster.entries] if args.all else [c.upper() for c in args.drivers]
    unknown = [c for c in codes if c not in by_code]
    if unknown:
        parser.error(f"unknown driver code(s) {unknown}; choose from {sorted(by_code)}")

    setup = build_setup(roster, profiles, [by_code[c] for c in codes])
    state = engine.start(engine.create_world(setup, seed=args.seed, world_id="demo"))
    code_of = {d.driver_id: d.code for d in setup.drivers}
    total = setup.track.total_laps
    n_cars = len(setup.drivers)

    print(f"{EVENT_ID} | {n_cars} cars | {total} laps | seed {args.seed} | {engine.ENGINE_VERSION}")
    print("grid: " + "  ".join(f"P{d.grid_position} {d.code}" for d in setup.drivers))
    print("synthetic lap model: no pit stops, no traffic, uncalibrated\n")

    crossings: dict[int, list[tuple[float, str, float]]] = defaultdict(list)
    while state.world_status is WorldStatus.RUNNING:
        result = engine.step(state)
        state = result.state
        for e in result.events:
            if e.type != "LAP_COMPLETED" or e.driver_id is None:
                continue
            lap = int(e.payload["lap"])
            crossings[lap].append(
                (float(e.payload["crossing_time_s"]), e.driver_id, float(e.payload["lap_time_s"]))
            )
            if len(crossings[lap]) == n_cars and (lap % args.every == 0 or lap == total):
                order = sorted(crossings[lap])
                leader_t = order[0][0]
                cells = [f"{code_of[order[0][1]]} {order[0][2]:6.2f}"]
                cells += [f"{code_of[d]} +{t - leader_t:.3f}" for t, d, _ in order[1:]]
                print(f"Lap {lap:>2}/{total}  " + "  ".join(cells))

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
