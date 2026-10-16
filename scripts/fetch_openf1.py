"""Fetch tyres, pit stops, race control, weather and laps for 2023+ races from OpenF1 (H0).

Writes tidy Parquet per race to ``data/processed/openf1/<year>_<nn>/`` (nn = race order in the
year); raw responses are cached under ``data/raw/openf1/``. ~5 requests per race at 30/minute.

Usage:
    uv run python scripts/fetch_openf1.py --years 2023 2024 2025
"""

import argparse
import functools
import json
from pathlib import Path

from f1ar.data.openf1_history import Throttle, fetch_session, race_sessions, session_tables

print = functools.partial(print, flush=True)  # noqa: A001 (progress through pipes)
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "openf1"
OUT = ROOT / "data" / "processed" / "openf1"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--years", type=int, nargs="+", default=[2023, 2024, 2025])
    args = parser.parse_args()
    throttle = Throttle()
    for year in args.years:
        sessions = race_sessions(year, RAW, throttle)
        print(f"{year}: {len(sessions)} races")
        for n, s in enumerate(sessions, 1):
            key = int(s["session_key"])
            counts = fetch_session(year, key, RAW, throttle)
            folder = OUT / f"{year}_{n:02d}"
            folder.mkdir(parents=True, exist_ok=True)
            for name, df in session_tables(year, key, RAW).items():
                df.to_parquet(folder / f"{name}.parquet", index=False)
            meta = {
                k: s.get(k)
                for k in (
                    "session_key",
                    "meeting_key",
                    "date_start",
                    "circuit_short_name",
                    "country_name",
                    "location",
                )
            }
            (folder / "session.json").write_text(json.dumps(meta) + "\n", encoding="utf-8")
            print(f"  {year} #{n:02d} {s.get('location')}: {counts}")
    print("done")


if __name__ == "__main__":
    main()
