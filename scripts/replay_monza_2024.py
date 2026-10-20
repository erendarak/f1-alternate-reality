"""Replay the REAL 2024 Italian GP lap by lap in the terminal (mode HISTORICAL_REPLAY).

Usage:
    uv run python scripts/fetch_monza_2024.py          # once
    uv run python scripts/replay_monza_2024.py          # every 5th lap + all events
    uv run python scripts/replay_monza_2024.py --every 1 --top 10

This is recorded history, not a simulation: nothing here comes from the engine.
"""

import argparse
import sys
from pathlib import Path

from f1ar.data.status import is_running_finish
from f1ar.history.reference import HistoricalRace
from f1ar.simulation.scenarios import MONZA_2024

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / MONZA_2024


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay the real Monza 2024 race")
    parser.add_argument("--every", type=int, default=5, help="print every Nth lap")
    parser.add_argument("--top", type=int, default=8, help="cars per line")
    args = parser.parse_args()
    if not (DATA / "laps.parquet").exists():
        sys.exit("no data yet; run: uv run python scripts/fetch_monza_2024.py")

    race = HistoricalRace.from_processed_dir(MONZA_2024, DATA)
    mismatches = race.validate_positions()
    print(f"{MONZA_2024} | {race.mode} | {race.total_laps} laps | {len(race.cars)} cars")
    print(f"order check vs timing feed: {len(mismatches)} mismatching lap positions\n")

    events = race.events()
    for lap in range(1, race.total_laps + 1):
        for e in events:
            if e.lap == lap and e.type in ("PIT_STOP", "RETIREMENT"):
                print(f"         lap {lap:>2}  {e.type:<10} {e.code}  {e.detail}")
        if lap % args.every and lap not in (1, race.total_laps):
            continue
        cells = []
        for i, (code, gap) in enumerate(race.classification_after_lap(lap)[: args.top], start=1):
            cells.append(f"{code}" if i == 1 else f"{code} +{gap:.1f}" if gap is not None else code)
        print(f"Lap {lap:>2}/{race.total_laps}  " + "  ".join(cells))

    print("\nFinal classification (official order)")
    for car in sorted(race.cars, key=lambda c: c.classified_position or 99):
        pos = f"P{car.classified_position}" if is_running_finish(car.status) else "DNF"
        print(
            f"  {pos:<4} {car.code}  {car.team:<16} {car.status:<9} {car.laps_completed} laps"
            f"  (grid P{car.grid_position})"
        )


if __name__ == "__main__":
    main()
