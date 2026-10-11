"""Fetch + validate any race: roster, normalised tables, manifest and quality report.

Usage:
    uv run python scripts/fetch_event.py --season 2024 --round 14     # 2024 Belgian GP
    uv run python scripts/fetch_event.py --season 2024 --round 16 --no-telemetry
    uv run python scripts/fetch_event.py --verify 2024_italian_gp     # re-hash files vs manifest

Writes data/processed/<event_id>/ (gitignored), configs/rosters/<event_id>.json,
data/manifests/<event_id>_fastf1.yaml and docs/data/<event_id>_quality.md.
"""

import argparse
import sys
from pathlib import Path

from f1ar.data.pipeline import fetch_event, format_result, verify_event

ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Fetch + validate one race")
    parser.add_argument("--season", type=int)
    parser.add_argument("--round", type=int)
    parser.add_argument("--no-telemetry", action="store_true", help="skip position samples")
    parser.add_argument("--refresh-roster", action="store_true", help="rebuild the roster file")
    parser.add_argument("--verify", metavar="EVENT_ID", help="only verify files vs manifest")
    args = parser.parse_args(argv)

    if args.verify:
        results = verify_event(ROOT, args.verify)
        for path, ok in results.items():
            print(f"  {'OK ' if ok else 'BAD'} {path}")
        sys.exit(0 if all(results.values()) else 1)
    if args.season is None or args.round is None:
        parser.error("--season and --round are required (or use --verify EVENT_ID)")

    result = fetch_event(
        args.season,
        args.round,
        ROOT,
        telemetry=not args.no_telemetry,
        refresh_roster=args.refresh_roster,
    )
    print("\n".join(format_result(result, ROOT)))
    if result.report.worst == "FAIL":
        sys.exit(1)


if __name__ == "__main__":
    main()
