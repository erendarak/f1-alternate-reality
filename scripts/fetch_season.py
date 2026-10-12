"""Fetch + validate every race of a season (M4 calibration data), one race at a time.

Usage:
    uv run python scripts/fetch_season.py --season 2024                 # timing only (~9 MB/race)
    uv run python scripts/fetch_season.py --season 2024 --rounds 9 12 21
    uv run python scripts/fetch_season.py --season 2024 --telemetry     # + position samples

Races already in data/processed/ are skipped (use --refresh to fetch again). A race whose
quality checks FAIL is reported and the season continues. Pauses between races keep well
inside Jolpica's unauthenticated limits (docs/LEGAL_AND_LICENSES.md).
"""

import argparse
import time
from pathlib import Path

from f1ar.data.fastf1_adapter import DataSourceError
from f1ar.data.pipeline import fetch_event

ROOT = Path(__file__).resolve().parents[1]
ROUNDS_2024 = 24
PAUSE_S = 2.0


def fetched_rounds(season: int) -> set[int]:
    """Rounds whose processed laps exist (by roster file round number)."""
    import json

    done = set()
    for roster in (ROOT / "configs" / "rosters").glob(f"{season}_*.json"):
        data = json.loads(roster.read_text(encoding="utf-8"))
        if (ROOT / "data" / "processed" / data["event_id"] / "laps.parquet").exists():
            done.add(int(data["round"]))
    return done


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--rounds", type=int, nargs="+")
    parser.add_argument("--telemetry", action="store_true", help="also fetch position samples")
    parser.add_argument("--refresh", action="store_true", help="fetch already-fetched races again")
    args = parser.parse_args()
    if args.season != 2024 and not args.rounds:
        parser.error("only the 2024 round count is known; pass --rounds")

    rounds = args.rounds or list(range(1, ROUNDS_2024 + 1))
    done = set() if args.refresh else fetched_rounds(args.season)
    summary: list[str] = []
    for round_ in rounds:
        if round_ in done:
            summary.append(f"  round {round_:>2}: already fetched")
            continue
        started = time.monotonic()
        try:
            result = fetch_event(args.season, round_, ROOT, telemetry=args.telemetry)
        except (DataSourceError, OSError, ValueError) as exc:
            summary.append(f"  round {round_:>2}: ERROR {type(exc).__name__}: {exc}")
            continue
        took = time.monotonic() - started
        summary.append(
            f"  round {round_:>2}: {result.paths.event_id:<28} quality {result.report.worst:<4} "
            f"({took:.0f} s)"
        )
        print(summary[-1], flush=True)
        time.sleep(PAUSE_S)
    print("\nSeason summary:")
    print("\n".join(summary))


if __name__ == "__main__":
    main()
