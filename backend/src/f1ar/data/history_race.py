"""A 2010-2017 race from the Jolpica history in the tables a FastF1 race has (H2b, D-055).

Jolpica gives each car's lap times and positions, pit stops (from 2011) and results; no tyre
compounds, track status, weather or positions on track. The converter writes the same
``data/processed/<event>/`` tables as the FastF1 pipeline so the dataset and estimators run on
both, with what each column really is recorded in the manifest:

- lap times, positions, results, grid: ``historical_verified``;
- pit stops (in-lap = Jolpica's stop lap): ``historical_verified`` from 2011; 2010 has none, so
  they are inferred from lap times (``history_inference``, EXP-021): ``estimated_from_data``;
- stints and tyre age = laps since the start or the last stop, each set assumed new (the start
  set's real age is unknown): ``estimated_from_data``;
- track status: laps the whole field ran slow are neutralised (``"4"``; SC and VSC not told
  apart): ``estimated_from_data``;
- compound unknown: ``compound_name`` is ``UNKNOWN`` and ``compound`` a ``MEDIUM`` placeholder
  (the engine needs a dry compound); weather unknown (no rows).
"""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from f1ar.data import jolpica_adapter
from f1ar.data.history_inference import inferred_pit_laps, neutralised_laps
from f1ar.data.jolpica_history import race_tables
from f1ar.data.manifests import DatasetManifest, write_table
from f1ar.domain.identities import RaceRoster
from f1ar.domain.provenance import SourceType

UNKNOWN_COMPOUND = "UNKNOWN"
PLACEHOLDER_COMPOUND = "MEDIUM"
"""What ``compound`` holds when the real one is unknown (``compound_name`` says UNKNOWN): the
engine and estimators need a dry compound on every lap; nothing reads it as the real tyre."""
NEUTRAL_STATUS = "4"
FIRST_SEASON_WITH_PIT_STOPS = 2011


WET_LAP_BAND = (1.08, 1.22)
"""A lap is wet when the field's median lap is 8-22 % slower than the race's fastest green laps
(5th percentile); slower still is a neutralisation. ``WET_RACE_SHARE`` of the laps must be wet.
Checked on 2018-2025 from lap times alone (FastF1 knows the weather): 14 of the 25 wet races
found, no false alarm (the misses: light rain that hardly slowed the cars)."""
WET_RACE_SHARE = 0.10


def wet_laps(clean: pd.DataFrame) -> set[int]:
    """``clean``: lap times without lap 1 and pit in/out laps (columns ``lap``, ``lap_time_s``)."""
    g = clean.dropna(subset=["lap_time_s"])
    if g.empty:
        return set()
    ref = float(g["lap_time_s"].quantile(0.05))
    lo, hi = WET_LAP_BAND
    med = g.groupby("lap")["lap_time_s"].median()
    wet = {int(str(k)) for k, m in med.items() if lo * ref < m < hi * ref}
    return wet if len(wet) >= WET_RACE_SHARE * len(med) else set()


def results_payload(season: int, round_: int, raw: Path) -> dict[str, Any]:
    """The cached Jolpica results pages of a race merged into one ``results.json`` payload."""
    folder = raw / str(season) / f"{round_:02d}"
    pages = sorted(folder.glob("results_*.json"))
    if not pages:
        raise ValueError(f"no cached Jolpica results for {season} round {round_}")
    races = [
        json.loads(p.read_text(encoding="utf-8"))["MRData"]["RaceTable"]["Races"] for p in pages
    ]
    first = dict(next(r for rs in races for r in rs))
    first["Results"] = [res for rs in races for r in rs for res in r.get("Results", [])]
    return {"MRData": {"RaceTable": {"Races": [first]}}}


def unique_grid(results: list[dict[str, Any]], quali: dict[str, int]) -> list[str]:
    """Jolpica occasionally gives two cars the same grid slot (2015 Italian GP: SAI and NAS 17,
    both penalised). Tied cars are ordered by qualifying position (then car number) and each
    following car moves back only as far as needed. Changes ``results`` in place; returns notes."""
    started = [r for r in results if int(r["grid"]) > 0]
    started.sort(
        key=lambda r: (
            int(r["grid"]),
            quali.get(r["Driver"]["driverId"], 99),
            int(r["number"]),
        )
    )
    original = [int(r["grid"]) for r in started]
    notes, last = [], 0
    for r in started:
        grid = int(r["grid"])
        if grid <= last:
            why = "shares its slot" if original.count(grid) > 1 else "behind a shared slot"
            notes.append(f"{r['Driver']['code']} grid {grid} -> {last + 1} ({why} in Jolpica)")
            r["grid"] = str(last + 1)
        last = int(r["grid"])
    return notes


def fill_codes(results: list[dict[str, Any]]) -> list[str]:
    """Jolpica has no three-letter code for some 2000-2004 drivers: one is made from the family
    name (ASCII, upper case), unique within the race. Changes ``results``; returns notes."""
    import unicodedata

    used = {r["Driver"]["code"] for r in results if r["Driver"].get("code")}
    notes = []
    for r in results:
        d = r["Driver"]
        if d.get("code"):
            continue
        plain = unicodedata.normalize("NFKD", f"{d['familyName']}{d['givenName']}")
        letters = "".join(
            c for c in plain.encode("ascii", "ignore").decode().upper() if c.isalpha()
        )
        candidates = [letters[:3]] + [letters[:2] + c for c in letters[2:]]
        code = next((c for c in candidates if len(c) == 3 and c not in used), "XXX")
        d["code"] = code
        used.add(code)
        notes.append(f"{d['driverId']} -> {code}")
    return [f"driver codes generated (none in Jolpica): {', '.join(notes)}"] if notes else []


def history_roster(season: int, round_: int, raw: Path) -> tuple[RaceRoster, list[str]]:
    from f1ar.data.pipeline import event_id_for

    payload = results_payload(season, round_, raw)
    race = payload["MRData"]["RaceTable"]["Races"][0]
    q = race_tables(season, round_, raw)["qualifying"]
    quali = (
        {} if q.empty else {str(d): int(p) for d, p in zip(q["driver"], q["position"], strict=True)}
    )
    notes = fill_codes(race["Results"]) + unique_grid(race["Results"], quali)
    return jolpica_adapter.roster_from_results(
        payload,
        event_id=event_id_for(season, race["raceName"]),
        source_url=jolpica_adapter.results_url(season, round_),
        retrieved_utc=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    ), notes


def history_tables(
    season: int, round_: int, raw: Path, roster: RaceRoster
) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    """The race's tables (FastF1 columns) and a note per derived column."""
    t = race_tables(season, round_, raw)
    jl, res, stops = t["laps"], t["results"], t["pitstops"]
    if jl.empty or res.empty:
        raise ValueError(f"{season} round {round_}: no laps or results in the Jolpica cache")
    by_jolpica = {e.external_ids["jolpica_driver_id"]: e for e in roster.entries}
    real_stops = season >= FIRST_SEASON_WITH_PIT_STOPS and not stops.empty
    stop_table = stops if real_stops else inferred_pit_laps(jl)
    stop_laps: dict[str, set[int]] = {}
    lane_s: dict[tuple[str, int], float] = {}  # Jolpica's pit-lane time (entry to exit), 2011+
    for s in stop_table.to_dict("records"):
        stop_laps.setdefault(str(s["driver"]), set()).add(int(s["lap"]))
        if real_stops and s.get("duration_s") is not None and not pd.isna(s["duration_s"]):
            lane_s[(str(s["driver"]), int(s["lap"]))] = float(s["duration_s"])
    pit_laps = {(d, lap) for d, ls in stop_laps.items() for x in ls for lap in (x, x + 1)}
    in_out = [
        (str(d), int(lap)) in pit_laps for d, lap in zip(jl["driver"], jl["lap"], strict=True)
    ]
    wet = wet_laps(jl[(jl["lap"] > 1) & ~pd.Series(in_out, index=jl.index)])
    neutral = neutralised_laps(jl) - wet  # wet laps are slow too, but not neutralised
    lap_median = jl.groupby("lap")["lap_time_s"].median()

    rows = []
    for driver, g in jl.sort_values(["driver", "lap"]).groupby("driver"):
        entry = by_jolpica.get(str(driver))
        if entry is None:
            continue
        clock = 0.0
        stint, age = 1, 0
        pit_entry = np.nan
        mine = stop_laps.get(str(driver), set())
        for r in g.to_dict("records"):
            lap = int(r["lap"])
            lap_time = r["lap_time_s"]
            known = lap_time is not None and not pd.isna(lap_time)
            used = float(lap_time) if known else float(lap_median.get(lap, np.nan))
            start = clock
            clock += used
            age += 1
            pit_in = lap in mine
            pit_out = (lap - 1) in mine
            rows.append(
                {
                    "driver_number": entry.car_number,
                    "driver_code": entry.code,
                    "team": entry.team_name,
                    "lap_number": lap,
                    "lap_time_s": float(lap_time) if known else np.nan,
                    "lap_start_time_s": start,
                    "lap_end_time_s": clock,
                    "pit_in_time_s": clock if pit_in else np.nan,
                    # pit exit = entry + the pit-lane time where Jolpica has it (else no lane time)
                    "pit_out_time_s": (pit_entry + lane_s.get((str(driver), lap - 1), 0.0))
                    if pit_out
                    else np.nan,
                    "stint": stint,
                    "compound": PLACEHOLDER_COMPOUND,
                    "tyre_life_laps": float(age),
                    "position": r["position"],
                    "track_status": NEUTRAL_STATUS if lap in neutral else "1",
                    "is_accurate": known and lap not in neutral and not pit_in and not pit_out,
                }
            )
            if pit_in:
                stint, age, pit_entry = stint + 1, 0, clock
    ints = ("driver_number", "lap_number", "stint", "position")
    floats = (
        "lap_time_s",
        "lap_start_time_s",
        "lap_end_time_s",
        "pit_in_time_s",
        "pit_out_time_s",
        "tyre_life_laps",
    )
    laps = pd.DataFrame(rows).astype(
        {**dict.fromkeys(ints, "Int64"), **dict.fromkeys(floats, "Float64")}
    )
    for col in ("sector1_s", "sector2_s", "sector3_s", "speed_trap_kph"):
        laps[col] = pd.Series(np.nan, index=laps.index, dtype="Float64")
    laps["fresh_tyre"] = pd.Series(pd.NA, index=laps.index, dtype="boolean")
    laps["is_accurate"] = laps["is_accurate"].astype("boolean")
    laps["deleted"] = pd.Series(False, index=laps.index, dtype="boolean")
    laps["fastf1_generated"] = pd.Series(False, index=laps.index, dtype="boolean")
    for col in ("driver_code", "team", "compound", "track_status"):
        laps[col] = laps[col].astype("string")
    laps["compound_name"] = pd.Series(UNKNOWN_COMPOUND, index=laps.index, dtype="string")
    laps = laps.sort_values(["driver_number", "lap_number"]).reset_index(drop=True)

    results = pd.DataFrame(
        [
            {
                "driver_number": by_jolpica[str(r["driver"])].car_number,
                "driver_code": by_jolpica[str(r["driver"])].code,
                "full_name": f"{by_jolpica[str(r['driver'])].given_name} "
                f"{by_jolpica[str(r['driver'])].family_name}",
                "team_name": by_jolpica[str(r["driver"])].team_name,
                "grid_position": by_jolpica[str(r["driver"])].grid_position,
                "position": int(r["position"]),
                "classified_position": str(r["position"]) if r["classified"] else "R",
                "status": str(r["status"]),
                "laps": int(r["laps"]),
                "points": float(r["points"]),
                "team_color": None,
            }
            for r in res.to_dict("records")
            if str(r["driver"]) in by_jolpica
        ]
    )
    for col in ("driver_number", "grid_position", "position", "laps"):
        results[col] = results[col].astype("Int64")
    results["points"] = results["points"].astype("Float64")
    for col in ("driver_code", "full_name", "team_name", "classified_position", "status"):
        results[col] = results[col].astype("string")
    results["team_color"] = results["team_color"].astype("string")

    # track status changes at the leader's lap starts (neutralised laps inferred)
    leader = laps.groupby("lap_number")["lap_start_time_s"].min()
    status_rows, current = [], None
    for lap_key, t0 in leader.items():
        code = NEUTRAL_STATUS if int(str(lap_key)) in neutral else "1"
        if code != current:
            status_rows.append(
                {
                    "session_time_s": float(t0),
                    "status": code,
                    "status_name": "neutralised (inferred)" if code != "1" else "green",
                    "message": "inferred from lap times",
                }
            )
            current = code
    track_status = pd.DataFrame(status_rows).astype(
        {
            "session_time_s": "Float64",
            "status": "string",
            "status_name": "string",
            "message": "string",
        }
    )
    n = len(leader)
    weather = pd.DataFrame(
        {
            "session_time_s": pd.Series(leader.to_numpy(dtype=float), dtype="Float64"),
            "air_temp_c": pd.Series([np.nan] * n, dtype="Float64"),
            "track_temp_c": pd.Series([np.nan] * n, dtype="Float64"),
            "humidity_pct": pd.Series([np.nan] * n, dtype="Float64"),
            "pressure_mbar": pd.Series([np.nan] * n, dtype="Float64"),
            "rainfall": pd.Series([int(str(k)) in wet for k in leader.index], dtype="boolean"),
            "wind_speed_ms": pd.Series([np.nan] * n, dtype="Float64"),
            "wind_direction_deg": pd.Series([pd.NA] * n, dtype="Int64"),
        }
    )
    notes = {
        "pit_stops": "historical_verified: Jolpica pit stops"
        if real_stops
        else "estimated_from_data: inferred from lap times (no Jolpica pit stops before 2011)",
        "stints_tyre_age": "estimated_from_data: laps since the start / last stop; new sets",
        "track_status": "estimated_from_data: neutralised laps inferred from the field's lap times",
        "compound": "unknown (Jolpica has no tyre data): compound = MEDIUM placeholder, "
        "compound_name = UNKNOWN",
        "weather": f"estimated_from_data: rain only, inferred from lap times ({len(wet)} wet laps: "
        f"field median {WET_LAP_BAND} x the race's 5th-percentile green lap); temperatures "
        "unknown",
    }
    return {
        "laps": laps,
        "results": results,
        "track_status": track_status,
        "weather": weather,
        "race_control": pd.DataFrame(
            {
                c: pd.Series(dtype="Int64" if c == "lap" else "string")
                for c in ("utc_time", "category", "flag", "scope", "lap", "message")
            }
        ),
    }, notes


def convert_race(season: int, round_: int, root: Path) -> tuple[str, int]:
    """Write the roster, tables and manifest of one cached race; returns (event_id, lap rows)."""
    raw = root / "data" / "raw" / "jolpica"
    roster, grid_notes = history_roster(season, round_, raw)
    tables, notes = history_tables(season, round_, raw, roster)
    if grid_notes:
        notes["grid"] = "; ".join(grid_notes)
    roster_path = root / "configs" / "rosters" / f"{roster.event_id}.json"
    roster_path.parent.mkdir(parents=True, exist_ok=True)
    roster_path.write_text(
        json.dumps(roster.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    out = root / "data" / "processed" / roster.event_id
    kinds = {
        "laps": (
            SourceType.HISTORICAL_VERIFIED,
            "lap times and positions (Jolpica); stints, "
            "tyre age, pit flags and track status derived, see missingness notes",
        ),
        "results": (SourceType.HISTORICAL_VERIFIED, "official classification (Jolpica)"),
        "track_status": (SourceType.ESTIMATED_FROM_DATA, "neutralised laps inferred"),
        "weather": (SourceType.ESTIMATED_FROM_DATA, "rain per lap inferred from lap times"),
        "race_control": (SourceType.EDITORIAL_ASSUMPTION, "no race control messages: empty"),
    }
    files = [
        write_table(
            tables[name], out / f"{name}.parquet", root=root, source_type=st, description=desc
        )
        for name, (st, desc) in kinds.items()
    ]
    manifest = DatasetManifest(
        dataset_id=f"{roster.event_id}_jolpica_v1",
        event=roster.event_id,
        source="jolpica-f1 (history cache, H2b)",
        source_api_version="ergast-compatible v1",
        retrieved_utc=roster.source.retrieved_utc,
        session_key=f"{season}/round {round_}/R",
        license_review_status="research_only",
        files=files,
        unit_conventions={"time": "seconds from the start (lap times summed per car)"},
        missingness_notes=[f"{k}: {v}" for k, v in notes.items()],
    )
    path = root / "data" / "manifests" / f"{roster.event_id}_jolpica.yaml"
    path.write_text(manifest.to_yaml(), encoding="utf-8", newline="\n")
    return roster.event_id, len(tables["laps"])
