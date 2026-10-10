"""Fetch a race's entry list + grid from Jolpica-F1 and write a race-specific roster file.

Usage:
    uv run python scripts/build_roster.py --season 2024 --round 16 --event-id 2024_italian_gp
"""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from f1ar.data.jolpica_adapter import fetch_json, results_url, roster_from_results

ROSTER_DIR = Path(__file__).resolve().parents[1] / "configs" / "rosters"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--event-id", required=True)
    args = parser.parse_args()

    url = results_url(args.season, args.round)
    retrieved = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    roster = roster_from_results(
        fetch_json(url), event_id=args.event_id, source_url=url, retrieved_utc=retrieved
    )

    ROSTER_DIR.mkdir(parents=True, exist_ok=True)
    out = ROSTER_DIR / f"{args.event_id}.json"
    out.write_text(
        json.dumps(roster.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {out} ({len(roster.entries)} entries, source {url})")
    for e in roster.entries:
        grid = f"P{e.grid_position:>2}" if e.grid_position else "PIT"
        print(f"  {grid}  #{e.car_number:<2} {e.code}  {e.driver_id:<24} {e.team_name}")


if __name__ == "__main__":
    main()
