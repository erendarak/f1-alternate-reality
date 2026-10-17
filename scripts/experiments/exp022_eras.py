"""EXP-022: the strategy game per season 2010-2025 from the real data, and a check of the season
rule catalogue against the official points (H1, D-053).

Per season: stops per classified finisher, stint lengths, neutralised laps per race (inferred
from lap times for every season; FastF1's track status, SC vs VSC, from 2018), retirements per
race; and the points every classified driver would get under ``configs/regulations/seasons.json``
vs Jolpica's official points (mismatches are listed per race with the distance run).

Pit stops: Jolpica's from 2011 (real); 2010 has none, so they are inferred from lap times
(``estimated_from_data``, EXP-021: precision 0.88, recall 0.73).

Usage:
    uv run python scripts/experiments/exp022_eras.py --json docs/experiments/exp022.json
"""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from f1ar.data.history_inference import inferred_pit_laps, neutralised_laps
from f1ar.data.jolpica_history import race_done, race_tables
from f1ar.domain.season_rules import load_catalogue

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "jolpica"
PROCESSED = ROOT / "data" / "processed"


def fastf1_event_dirs() -> dict[tuple[int, int], Path]:
    """(season, round) -> FastF1 race folder, from the rosters."""
    out = {}
    for roster in (ROOT / "configs" / "rosters").glob("*.json"):
        data = json.loads(roster.read_text(encoding="utf-8"))
        folder = PROCESSED / data["event_id"]
        if (folder / "laps.parquet").exists():
            out[(int(data["season"]), int(data["round"]))] = folder
    return out


def fastf1_neutralised(folder: Path) -> tuple[int, int]:
    """(SC laps, VSC laps) from FastF1's track status: a lap counts when most cars saw it."""
    laps = pd.read_parquet(folder / "laps.parquet", columns=["lap_number", "track_status"])
    laps = laps.dropna()
    sc = vsc = 0
    for _, g in laps.groupby("lap_number"):
        codes = g["track_status"].astype(str)
        if (codes.str.contains("4") | codes.str.contains("5")).mean() > 0.5:
            sc += 1
        elif (codes.str.contains("6") | codes.str.contains("7")).mean() > 0.5:
            vsc += 1
    return sc, vsc


def stints(stop_laps: list[int], laps_run: int) -> list[int]:
    edges = [0, *sorted(stop_laps), laps_run]
    return [b - a for a, b in zip(edges, edges[1:], strict=False) if b > a]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--from", dest="first", type=int, default=2010)
    parser.add_argument("--to", dest="last", type=int, default=2025)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    catalogue = load_catalogue(ROOT / "configs")
    fastf1 = fastf1_event_dirs()
    per_season: dict[int, dict[str, Any]] = {}
    mismatches: list[dict[str, Any]] = []
    for season in range(args.first, args.last + 1):
        rules = catalogue[season]
        schedule_file = RAW / str(season) / "schedule.json"
        if not schedule_file.exists():
            continue
        schedule = json.loads(schedule_file.read_text(encoding="utf-8"))
        races = schedule["MRData"]["RaceTable"]["Races"]
        final_round = max(int(r["round"]) for r in races)
        acc: dict[str, list[float]] = defaultdict(list)
        n_races = checked = agree = 0
        for race in races:
            round_ = int(race["round"])
            if not race_done(season, round_, RAW):
                continue
            t = race_tables(season, round_, RAW)
            res, laps, stops = t["results"], t["laps"], t["pitstops"]
            if res.empty or laps.empty:
                continue
            n_races += 1
            real_stops = season >= 2011 and not stops.empty
            stop_table = stops if real_stops else inferred_pit_laps(laps)
            classified = res[res["classified"]]
            by_driver = stop_table.groupby("driver")["lap"].apply(list).to_dict()
            for r in classified.to_dict("records"):
                driver_stops = [int(x) for x in by_driver.get(r["driver"], [])]
                acc["stops"].append(len(driver_stops))
                s = stints(driver_stops, int(r["laps"]))
                acc["stint"] += [float(x) for x in s]
                if s:
                    acc["first_stint_share"].append(s[0] / max(1, int(r["laps"])))
            acc["dnf"].append(float((~res["classified"]).sum()))
            acc["neutral_inferred"].append(float(len(neutralised_laps(laps))))
            if (season, round_) in fastf1:
                sc, vsc = fastf1_neutralised(fastf1[(season, round_)])
                acc["sc_laps"].append(float(sc))
                acc["vsc_laps"].append(float(vsc))
            # the catalogue vs the official points
            wrong = []
            for r in res.to_dict("records"):
                ours = (
                    rules.points_for(
                        int(r["position"]),
                        fastest_lap=r["fastest_lap_rank"] == 1,
                        final_round=round_ == final_round,
                    )
                    if r["classified"]
                    else 0
                )
                checked += 1
                if abs(ours - float(r["points"])) < 1e-9:
                    agree += 1
                else:
                    wrong.append((r["code"] or r["driver"], ours, float(r["points"])))
            if wrong:
                winner_laps = int(res["laps"].max())
                mismatches.append(
                    {
                        "race": f"{season} R{round_:02d} {race['raceName']}",
                        "laps_run": winner_laps,
                        "drivers": wrong[:6],
                        "n": len(wrong),
                    }
                )

        def mean(key: str, acc: dict[str, list[float]] = acc) -> float | None:
            return round(statistics.fmean(acc[key]), 2) if acc[key] else None

        per_season[season] = {
            "races": n_races,
            "of": len(races),
            "stops_source": "jolpica (real)" if season >= 2011 else "inferred from lap times",
            "stops_per_finisher": mean("stops"),
            "zero_stop_share": (
                round(sum(1 for x in acc["stops"] if x == 0) / len(acc["stops"]), 3)
                if acc["stops"]
                else None
            ),
            "median_stint_laps": statistics.median(acc["stint"]) if acc["stint"] else None,
            "first_stint_share": mean("first_stint_share"),
            "retirements_per_race": mean("dnf"),
            "neutralised_laps_inferred": mean("neutral_inferred"),
            "fastf1_races": len(acc["sc_laps"]),
            "sc_laps_fastf1": mean("sc_laps"),
            "vsc_laps_fastf1": mean("vsc_laps"),
            "points_agree": f"{agree}/{checked}",
            "rules": {
                "vsc": rules.vsc,
                "drs": rules.drs,
                "fastest_lap_point": rules.fastest_lap_point,
                "tyre_supplier": rules.tyre_supplier,
                "top10_start_tyres": rules.top10_start_tyres,
            },
        }
        p = per_season[season]
        print(
            f"{season}: {p['races']}/{p['of']} races, stops/finisher {p['stops_per_finisher']} "
            f"(0-stop {p['zero_stop_share']}), stint {p['median_stint_laps']}, "
            f"DNF {p['retirements_per_race']}, neutral {p['neutralised_laps_inferred']} "
            f"(SC {p['sc_laps_fastf1']} / VSC {p['vsc_laps_fastf1']}), points {p['points_agree']}",
            flush=True,
        )
    print("\nPoints mismatches (catalogue vs official):")
    for m in mismatches:
        print(f"  {m['race']} ({m['laps_run']} laps): {m['n']} drivers, e.g. {m['drivers'][:3]}")
    if args.json:
        args.json.write_text(
            json.dumps({"seasons": per_season, "points_mismatches": mismatches}, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )


if __name__ == "__main__":
    main()
