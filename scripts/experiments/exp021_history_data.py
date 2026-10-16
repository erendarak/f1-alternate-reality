"""EXP-021: the 2000-2025 history data: coverage, and how well the inference works (H0, D-052).

1. **Coverage** per season: races fetched, laps per race, real pit stops (Jolpica, 2011+),
   OpenF1 tyre stints (2023+).
2. **Pit stops inferred from lap times** (what 2000-2010 has to rely on) vs Jolpica's real ones
   on every fetched 2011+ race: precision, recall, per season.
3. **Neutralised laps inferred from lap times** vs OpenF1 race control (SC / VSC / red flag
   messages, 2023+): share of real neutralised laps found, false alarms.
4. **The same on FastF1 races** (real pit-in laps and track status, 2018-2025): thresholds
   tuned on 2018-2021, tested on 2022-2025.

Runs on whatever has been fetched so far (scripts/fetch_history.py, scripts/fetch_openf1.py).

Usage:
    uv run python scripts/experiments/exp021_history_data.py --json docs/experiments/exp021.json
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from f1ar.data.history_inference import inferred_pit_laps, match, neutralised_laps

ROOT = Path(__file__).resolve().parents[2]
HISTORY = ROOT / "data" / "processed" / "history"
OPENF1 = ROOT / "data" / "processed" / "openf1"


def openf1_neutral_laps(folder: Path) -> set[int]:
    """Laps with an SC / VSC / red flag in OpenF1's race control (deployed until ending)."""
    rc = pd.read_parquet(folder / "race_control.parquet")
    if rc.empty or "message" not in rc.columns:
        return set()
    rc = rc.dropna(subset=["lap_number"]).sort_values("date")
    laps: set[int] = set()
    start: int | None = None
    for r in rc.to_dict("records"):
        msg, lap = str(r["message"]).upper(), int(r["lap_number"])
        if (
            "SAFETY CAR DEPLOYED" in msg or "VSC DEPLOYED" in msg or "RED FLAG" in msg
        ) and start is None:
            start = lap
        elif start is not None and (
            "IN THIS LAP" in msg or "ENDING" in msg or "GREEN" in msg or "RESUME" in msg
        ):
            laps.update(range(start, lap + 1))
            start = None
    return laps


def fastf1_validation(first: int, last: int) -> dict[str, float | int]:
    """The inference on FastF1 races (real pit-in laps and track status per lap)."""
    hits = real = inferred = found = missed = false = races = 0
    for p in sorted(ROOT.glob("data/processed/20[12][0-9]_*/laps.parquet")):
        year = int(p.parent.name[:4])
        if not first <= year <= last:
            continue
        d = pd.read_parquet(p)
        laps = pd.DataFrame(
            {
                "driver": d["driver_code"].astype(str),
                "lap": d["lap_number"].astype(int),
                "lap_time_s": d["lap_time_s"].astype(float),
            }
        )
        stops = d[d["pit_in_time_s"].notna()][["driver_code", "lap_number"]]
        stops = stops.rename(columns={"driver_code": "driver", "lap_number": "lap"})
        m = match(stops, inferred_pit_laps(laps))
        hits, real = hits + int(m["hits"]), real + int(m["real"])
        inferred += int(m["inferred"])
        status = d.assign(n=d["track_status"].astype(str).str.contains("4|5|6|7"))
        share = status.groupby("lap_number")["n"].mean().reset_index().to_dict("records")
        truth = {int(r["lap_number"]) for r in share if float(r["n"]) > 0.5}
        guess = neutralised_laps(laps)
        found, missed = found + len(truth & guess), missed + len(truth - guess)
        false += len(guess - truth)
        races += 1
    return {
        "races": races,
        "pit_precision": round(hits / max(1, inferred), 3),
        "pit_recall": round(hits / max(1, real), 3),
        "neutral_found_share": round(found / max(1, found + missed), 3),
        "neutral_false_laps_per_race": round(false / max(1, races), 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    coverage: dict[int, dict[str, Any]] = defaultdict(lambda: {"races": 0, "laps": 0, "stops": 0})
    pit: dict[int, list[dict[str, float]]] = defaultdict(list)
    for folder in sorted(HISTORY.glob("*_*")):
        season = int(folder.name.split("_")[0])
        laps = pd.read_parquet(folder / "laps.parquet")
        stops = pd.read_parquet(folder / "pitstops.parquet")
        c = coverage[season]
        c["races"] += 1
        c["laps"] += len(laps)
        c["stops"] += len(stops)
        if season >= 2011 and len(stops):
            pit[season].append(match(stops[["driver", "lap"]], inferred_pit_laps(laps)))
    result: dict[str, Any] = {"coverage": {}, "pit_inference": {}, "neutral_inference": {}}
    for season in sorted(coverage):
        c = coverage[season]
        result["coverage"][season] = {
            "races": c["races"],
            "laps_per_race": round(c["laps"] / max(1, c["races"])),
            "real_stops_per_race": round(c["stops"] / max(1, c["races"]), 1),
        }
    for season, rows in sorted(pit.items()):
        real = sum(r["real"] for r in rows)
        hits = sum(r["hits"] for r in rows)
        inferred = sum(r["inferred"] for r in rows)
        result["pit_inference"][season] = {
            "races": len(rows),
            "precision": round(hits / max(1, inferred), 3),
            "recall": round(hits / max(1, real), 3),
        }
    # neutralised laps: OpenF1 race control vs the inference (match races by year and order)
    found = missed = false = 0
    for folder in sorted(OPENF1.glob("*_*")):
        year, n = (int(x) for x in folder.name.split("_"))
        hist = sorted(HISTORY.glob(f"{year}_*"))
        if n > len(hist) or not (folder / "race_control.parquet").exists():
            continue
        real_laps = openf1_neutral_laps(folder)
        guess = neutralised_laps(pd.read_parquet(hist[n - 1] / "laps.parquet"))
        found += len(real_laps & guess)
        missed += len(real_laps - guess)
        false += len(guess - real_laps)
    result["neutral_inference"] = {
        "real_neutral_laps": found + missed,
        "found_share": round(found / max(1, found + missed), 3),
        "false_alarm_laps": false,
    }
    result["fastf1_validation"] = {
        "tuned_on_2018_2021": fastf1_validation(2018, 2021),
        "test_2022_2025": fastf1_validation(2022, 2025),
    }
    print(json.dumps(result, indent=1))
    if args.json:
        args.json.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
