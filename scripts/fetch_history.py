"""Fetch every race of 2000-2025 from Jolpica-F1 into the local raw cache (H0, D-052).

Order: 2017 -> 2010, then 2009 -> 2000, then 2018-2025 (in full detail from FastF1). Throttled
below the 500 requests/hour limit (``--interval``, default 9 s/request = 400/hour, headroom for
FastF1's own Jolpica calls); resumable: cached pages are never fetched again. Prints one line
per race with an ETA. Then writes tidy Parquet tables per race to
``data/processed/history/<season>_<round>/``.

Usage:
    uv run python scripts/fetch_history.py --from 2000 --to 2025
"""

import argparse
import functools
import time
from pathlib import Path

from f1ar.data.jolpica_history import (
    Throttle,
    fetch_race,
    race_done,
    race_tables,
    season_schedule,
)

print = functools.partial(print, flush=True)  # noqa: A001 (progress through pipes)
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "jolpica"
OUT = ROOT / "data" / "processed" / "history"
REQUESTS_PER_RACE = 15


def write_tables(season: int, round_: int) -> None:
    folder = OUT / f"{season}_{round_:02d}"
    folder.mkdir(parents=True, exist_ok=True)
    for name, df in race_tables(season, round_, RAW).items():
        df.to_parquet(folder / f"{name}.parquet", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--from", dest="first", type=int, default=2000)
    parser.add_argument("--to", dest="last", type=int, default=2025)
    parser.add_argument("--interval", type=float, default=9.0, help="seconds per request")
    args = parser.parse_args()
    throttle = Throttle(args.interval)
    # priority: the years nothing else covers first (FastF1 has 2018-2025 in full detail)
    span = range(args.first, args.last + 1)
    seasons = [y for y in range(2017, 2009, -1) if y in span]
    seasons += [y for y in range(2009, 1999, -1) if y in span and y not in seasons]
    seasons += [y for y in range(args.last, args.first - 1, -1) if y not in seasons]
    schedules = {s: season_schedule(s, RAW, throttle) for s in seasons}
    todo = [
        (s, int(r["round"]), r["raceName"])
        for s in seasons
        for r in schedules[s]
        if not race_done(s, int(r["round"]), RAW)
    ]
    total = sum(len(v) for v in schedules.values())
    print(
        f"{total} races in {args.first}-{args.last}; {len(todo)} to fetch "
        f"(~{len(todo) * REQUESTS_PER_RACE * args.interval / 3600:.1f} h)"
    )
    t0 = time.monotonic()
    for i, (season, round_, name) in enumerate(todo, 1):
        before = throttle.requests
        try:
            counts = fetch_race(season, round_, RAW, throttle)
        except Exception as exc:  # noqa: BLE001 (one bad race must not stop the history)
            print(f"{season} R{round_:02d} {name}: ERROR {type(exc).__name__}: {exc}")
            continue
        write_tables(season, round_)
        per_req = (time.monotonic() - t0) / max(1, throttle.requests)
        left = (len(todo) - i) * REQUESTS_PER_RACE * per_req / 3600
        print(
            f"{season} R{round_:02d} {name}: {counts} ({throttle.requests - before} requests) "
            f"[{i}/{len(todo)}, ~{left:.1f} h left]"
        )
    # tables for races cached earlier
    for s in seasons:
        for r in schedules[s]:
            rd = int(r["round"])
            if race_done(s, rd, RAW) and not (OUT / f"{s}_{rd:02d}" / "laps.parquet").exists():
                write_tables(s, rd)
    print("done")


if __name__ == "__main__":
    main()
