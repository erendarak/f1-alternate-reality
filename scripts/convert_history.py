"""Convert cached Jolpica races into the common race tables (H2b, D-055): rosters in
configs/rosters/, tables in data/processed/<event>/, manifests data/manifests/<event>_jolpica.yaml.

Offline (reads the raw cache written by fetch_history.py). Seasons with FastF1 data (2018+) are
refused: their FastF1 tables are richer.

Usage:
    uv run python scripts/convert_history.py --from 2010 --to 2017
    uv run python scripts/convert_history.py --from 2013 --to 2013 --rounds 5
"""

import argparse
import json
from pathlib import Path

from f1ar.data.history_race import convert_race
from f1ar.data.jolpica_history import race_done

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "jolpica"
FIRST_FASTF1_SEASON = 2018


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--from", dest="first", type=int, required=True)
    parser.add_argument("--to", dest="last", type=int, required=True)
    parser.add_argument("--rounds", type=int, nargs="+")
    args = parser.parse_args()
    if args.last >= FIRST_FASTF1_SEASON:
        parser.error(f"{FIRST_FASTF1_SEASON}+ comes from FastF1 (scripts/fetch_season.py)")
    for season in range(args.first, args.last + 1):
        schedule = json.loads((RAW / str(season) / "schedule.json").read_text(encoding="utf-8"))
        rounds = [int(r["round"]) for r in schedule["MRData"]["RaceTable"]["Races"]]
        done = 0
        for round_ in args.rounds or rounds:
            if not race_done(season, round_, RAW):
                print(f"{season} R{round_:02d}: not in the cache yet (fetch_history.py)")
                continue
            try:
                event_id, rows = convert_race(season, round_, ROOT)
            except ValueError as exc:
                print(f"{season} R{round_:02d}: ERROR {exc}")
                continue
            done += 1
            print(f"{season} R{round_:02d}: {event_id:<28} {rows:>5} lap rows", flush=True)
        print(f"{season}: {done} races converted")


if __name__ == "__main__":
    main()
