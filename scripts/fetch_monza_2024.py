"""Shortcut: fetch + validate the 2024 Italian GP (see scripts/fetch_event.py for any race).

Usage:
    uv run python scripts/fetch_monza_2024.py                 # cached after the first run
    uv run python scripts/fetch_monza_2024.py --no-telemetry  # skip ~600k position samples
    uv run python scripts/fetch_monza_2024.py --verify        # only check files vs the manifest
"""

import argparse
import sys
from pathlib import Path

from f1ar.data.pipeline import fetch_event, format_result, verify_event
from f1ar.simulation.scenarios import MONZA_2024

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch + validate Monza 2024 race data")
    parser.add_argument("--no-telemetry", action="store_true")
    parser.add_argument("--verify", action="store_true", help="only verify files vs manifest")
    args = parser.parse_args()
    if args.verify:
        results = verify_event(ROOT, MONZA_2024)
        for path, ok in results.items():
            print(f"  {'OK ' if ok else 'BAD'} {path}")
        sys.exit(0 if all(results.values()) else 1)

    result = fetch_event(2024, 16, ROOT, telemetry=not args.no_telemetry)
    print("\n".join(format_result(result, ROOT)))
    if result.report.worst == "FAIL":
        sys.exit(1)


if __name__ == "__main__":
    main()
