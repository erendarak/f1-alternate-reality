"""Data-quality report for one race (plan §6.4 validate/annotate/audit steps).

Pure functions over normalised tables (``fastf1_adapter.RaceTables``) plus our verified roster
and, optionally, an independent results source (Jolpica). Produces typed checks with
PASS/WARN/FAIL plus descriptive sections, and renders them as Markdown for ``docs/data/``.

A WARN is not an error: it marks something the simulator or later calibration must handle
(retirements, lapped cars, missing sector 1 on lap 1, ...).
"""

from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict

from f1ar.data.fastf1_adapter import NEUTRALISED_STATUSES, TRACK_STATUS_NAMES, RaceTables
from f1ar.domain.identities import RaceRoster

Status = Literal["PASS", "WARN", "FAIL"]


class Check(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    check_id: str
    title: str
    status: Status
    detail: str


class QualityReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    event: dict[str, Any]
    source: dict[str, str]
    checks: tuple[Check, ...]
    table_rows: dict[str, int]
    lap_missingness: dict[str, int]
    stints: list[dict[str, Any]]
    pit_stops: list[dict[str, Any]]
    weather_summary: dict[str, Any]
    track_status_changes: list[dict[str, Any]]
    race_control_categories: dict[str, int]
    position_coverage: dict[str, Any]
    green_lap_pace: dict[str, Any]
    notes: tuple[str, ...]

    @property
    def worst(self) -> Status:
        statuses = {c.status for c in self.checks}
        return "FAIL" if "FAIL" in statuses else "WARN" if "WARN" in statuses else "PASS"


def _check(check_id: str, title: str, ok: bool, detail: str, *, warn: bool = False) -> Check:
    status: Status = "PASS" if ok else ("WARN" if warn else "FAIL")
    return Check(check_id=check_id, title=title, status=status, detail=detail)


def representative_laps(laps: pd.DataFrame) -> pd.DataFrame:
    """Green-flag racing laps usable for pace estimates.

    Excludes lap 1 (standing start), pit-in and pit-out laps, non-green laps, laps FastF1
    marks inaccurate or generated, deleted laps, and laps without a time.
    """
    mask = (
        (laps["lap_number"] > 1)
        & laps["pit_in_time_s"].isna()
        & laps["pit_out_time_s"].isna()
        & (laps["track_status"] == "1")
        & laps["is_accurate"].fillna(False)
        & ~laps["deleted"].fillna(False)
        & ~laps["fastf1_generated"].fillna(False)
        & laps["lap_time_s"].notna()
    )
    return laps[mask]


def stint_table(laps: pd.DataFrame) -> pd.DataFrame:
    grouped = laps.groupby(["driver_number", "driver_code", "stint"], dropna=False)
    out = grouped.agg(
        compound=("compound", "first"),
        first_lap=("lap_number", "min"),
        last_lap=("lap_number", "max"),
        laps=("lap_number", "count"),
        tyre_life_start=("tyre_life_laps", "first"),
        fresh_tyre=("fresh_tyre", "first"),
    ).reset_index()
    return out.sort_values(["driver_number", "stint"]).reset_index(drop=True)


def _records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """JSON-friendly records: pandas NA -> None, numpy scalars -> Python."""
    clean = df.astype(object).where(df.notna(), None)
    return [
        {str(k): (v.item() if hasattr(v, "item") else v) for k, v in r.items()}
        for r in clean.to_dict("records")
    ]


def build_quality_report(
    tables: RaceTables,
    roster: RaceRoster,
    *,
    source: dict[str, str],
    expected_total_laps: int,
    reference_results: list[dict[str, Any]] | None = None,
) -> QualityReport:
    laps, results = tables.laps, tables.results
    checks: list[Check] = []

    # --- identity -----------------------------------------------------------------------
    ev = tables.event
    checks.append(
        _check(
            "event",
            "Session is the expected race",
            ev["round"] == roster.round
            and ev["event_date"] == roster.race_date
            and ev["session_name"] == "Race",
            f"{ev['event_name']} round {ev['round']}, {ev['event_date']}, {ev['session_name']}",
        )
    )

    by_number = {e.car_number: e for e in roster.entries}
    # Plain Python records per car: simpler (and better typed) than scalar DataFrame lookups.
    res: dict[int, dict[str, Any]] = {int(r["driver_number"]): r for r in _records(results)}
    ff_numbers = set(res)
    checks.append(
        _check(
            "roster_numbers",
            "Same 20 car numbers as the Jolpica roster",
            ff_numbers == set(by_number),
            f"FastF1 {len(ff_numbers)} cars; only in FastF1: "
            f"{sorted(ff_numbers - set(by_number))}, "
            f"only in roster: {sorted(set(by_number) - ff_numbers)}",
        )
    )
    common = sorted(ff_numbers & set(by_number))
    code_mismatch = [n for n in common if res[n]["driver_code"] != by_number[n].code]
    checks.append(
        _check(
            "roster_codes",
            "Driver codes match per car number",
            not code_mismatch,
            f"mismatches: {code_mismatch}" if code_mismatch else f"{len(common)} codes match",
        )
    )
    grid_mismatch = [
        (n, res[n]["grid_position"], by_number[n].grid_position)
        for n in common
        if res[n]["grid_position"] != by_number[n].grid_position
    ]
    checks.append(
        _check(
            "roster_grid",
            "Starting grid matches the roster",
            not grid_mismatch,
            f"(car, fastf1, roster): {grid_mismatch}"
            if grid_mismatch
            else "all 20 grid slots match",
        )
    )

    def partition(pairs: dict[int, str]) -> set[frozenset[int]]:
        teams: dict[str, set[int]] = {}
        for number, team in pairs.items():
            teams.setdefault(team, set()).add(number)
        return {frozenset(v) for v in teams.values()}

    ff_teams = partition({n: str(res[n]["team_name"]) for n in common})
    roster_teams = partition({n: by_number[n].team_id for n in common})
    name_pairs = sorted({(by_number[n].team_name, str(res[n]["team_name"])) for n in common})
    renamed = [f"{a} = {b}" for a, b in name_pairs if a != b]
    checks.append(
        _check(
            "roster_teams",
            "Team-mates pair up identically (names may differ)",
            ff_teams == roster_teams,
            "same 10 two-car teams; naming differences: " + ("; ".join(renamed) or "none"),
        )
    )

    # --- results vs independent source -----------------------------------------------------
    if reference_results is not None:
        ref = {r["number"]: r for r in reference_results}
        diffs = []
        for n in common:
            r = ref.get(n)
            mine = (res[n]["position"], res[n]["laps"])
            theirs = (r["position"], r["laps"]) if r else None
            if mine != theirs:
                diffs.append((n, mine, theirs))
        checks.append(
            _check(
                "results_crosscheck",
                "Finishing position and laps agree with Jolpica",
                not diffs,
                f"(car, fastf1, jolpica): {diffs}"
                if diffs
                else "all 20 positions and lap counts agree",
            )
        )

    # --- laps -------------------------------------------------------------------------------
    total = int(laps["lap_number"].max())
    checks.append(
        _check(
            "race_distance",
            f"Race distance is {expected_total_laps} laps",
            total == expected_total_laps,
            f"max lap number {total}",
        )
    )
    dup = int(laps.duplicated(["driver_number", "lap_number"]).sum())
    gaps = []
    for _, g in laps.groupby("driver_number"):
        numbers = g["lap_number"].tolist()
        if numbers != list(range(1, len(numbers) + 1)):
            gaps.append(int(g["driver_number"].iloc[0]))
    checks.append(
        _check(
            "lap_sequence",
            "Each car's laps are numbered 1..N without gaps or duplicates",
            dup == 0 and not gaps,
            f"duplicates {dup}, cars with gaps {gaps}",
        )
    )
    per_car = {int(g["driver_number"].iloc[0]): len(g) for _, g in laps.groupby("driver_number")}
    lap_count_diff = [
        (n, per_car.get(n, 0), res[n]["laps"])
        for n in common
        if per_car.get(n, 0) != res[n]["laps"]
    ]
    # Explained differences (seen across the 2024 season, M4): FastF1 keeps a row for the lap a
    # car retired on, which isn't classified (+1); a disqualified car is classified with 0 laps.
    explained, unexplained = [], []
    for n, rows, classified in lap_count_diff:
        status = res[n]["status"]
        finished = status in ("Finished", "Lapped", "Disqualified")
        retired_lap = not finished and rows == classified + 1
        if retired_lap or status == "Disqualified":
            explained.append((n, rows, classified, status))
        else:
            unexplained.append((n, rows, classified, status))
    checks.append(
        _check(
            "laps_vs_results",
            "Lap rows per car equal classified laps",
            not lap_count_diff,
            "all cars consistent"
            if not lap_count_diff
            else (
                f"unexplained (car, rows, classified, status): {unexplained}; "
                if unexplained
                else ""
            )
            + (
                f"explained (retirement lap kept as a row, or disqualified): {explained}"
                if explained
                else ""
            ),
            warn=not unexplained,
        )
    )
    short = results[results["laps"] < total]
    non_finishers = results[~results["status"].isin(["Finished", "Lapped"])]
    checks.append(
        _check(
            "classification",
            "Every car completes full race distance",
            short.empty,
            f"lapped: {sorted(r['driver_code'] for r in res.values() if r['status'] == 'Lapped')}; "
            f"not finished: "
            + ", ".join(
                f"{r.driver_code} ({r.status}, {r.laps} laps)" for r in non_finishers.itertuples()
            ),
            warn=True,
        )
    )
    missing = {c: int(laps[c].isna().sum()) for c in laps.columns}
    key_missing = {
        c: missing[c]
        for c in ("lap_time_s", "compound", "tyre_life_laps", "stint", "position")
        if missing[c]
    }
    checks.append(
        _check(
            "lap_core_fields",
            "Lap time, compound, tyre life, stint, position present on every lap",
            not key_missing,
            f"missing: {key_missing}" if key_missing else "no gaps in core lap fields",
            warn=True,
        )
    )
    lap1_s1 = int(laps[(laps["lap_number"] == 1)]["sector1_s"].isna().sum())
    other_s1 = missing["sector1_s"] - lap1_s1
    checks.append(
        _check(
            "sector_times",
            "Sector times present (lap 1 sector 1 is expected to be missing)",
            other_s1 == 0 and missing["sector2_s"] == 0 and missing["sector3_s"] == 0,
            f"sector1 missing on lap 1: {lap1_s1}, elsewhere: {other_s1}; "
            f"sector2 {missing['sector2_s']}, sector3 {missing['sector3_s']}",
            warn=True,
        )
    )
    tyre_jumps = []
    for _, g in laps.groupby(["driver_number", "stint"]):
        life = g["tyre_life_laps"].dropna().tolist()
        if any(b - a != 1 for a, b in zip(life, life[1:], strict=False)):
            tyre_jumps.append((int(g["driver_number"].iloc[0]), int(g["stint"].iloc[0])))
    checks.append(
        _check(
            "tyre_life_continuity",
            "Tyre life increases by exactly 1 per lap within each stint",
            not tyre_jumps,
            f"(car, stint) with jumps: {tyre_jumps}" if tyre_jumps else "continuous in every stint",
            warn=True,
        )
    )
    flags = {
        "inaccurate": int((~laps["is_accurate"].fillna(False)).sum()),
        "deleted": int(laps["deleted"].fillna(False).sum()),
        "fastf1_generated": int(laps["fastf1_generated"].fillna(False).sum()),
    }
    checks.append(
        _check(
            "lap_flags",
            "Lap accuracy flags",
            flags["deleted"] == 0 and flags["fastf1_generated"] == 0,
            f"{flags}; inaccurate laps are excluded from pace statistics",
            warn=True,
        )
    )

    # --- track status, weather, positions ------------------------------------------------------
    neutral_laps = laps[
        laps["track_status"].fillna("").map(lambda s: any(ch in NEUTRALISED_STATUSES for ch in s))
    ]
    checks.append(
        _check(
            "neutralisations",
            "No safety car / VSC / red flag",
            neutral_laps.empty,
            f"{len(neutral_laps)} lap rows under SC/VSC/red"
            if not neutral_laps.empty
            else "fully green race (only status changes: "
            + ", ".join(tables.track_status["status_name"].dropna().tolist())
            + ")",
            warn=True,
        )
    )
    w = tables.weather
    rain = bool(w["rainfall"].fillna(False).any())
    checks.append(
        _check(
            "weather",
            "Weather data covers the race and it stayed dry",
            not w.empty and not rain,
            f"{len(w)} samples, rainfall {'YES' if rain else 'no'}",
            warn=True,
        )
    )
    pos = tables.positions
    pos_cars = int(pos["driver_number"].nunique()) if not pos.empty else 0
    checks.append(
        _check(
            "positions",
            "Position samples for every car",
            pos_cars == len(common),
            f"{len(pos)} samples for {pos_cars} cars",
            warn=True,
        )
    )

    # --- descriptive sections ------------------------------------------------------------------
    stints = stint_table(laps)
    pits = laps[laps["pit_in_time_s"].notna()][["driver_code", "lap_number", "compound"]]
    pit_out = laps[laps["pit_out_time_s"].notna() & (laps["lap_number"] > 1)]
    pit_rows = []
    for row in _records(pits):
        nxt = pit_out[
            (pit_out["driver_code"] == row["driver_code"])
            & (pit_out["lap_number"] == row["lap_number"] + 1)
        ]
        pit_rows.append(
            {
                "driver": row["driver_code"],
                "in_lap": row["lap_number"],
                "from": row["compound"],
                "to": nxt["compound"].iloc[0] if len(nxt) else None,
            }
        )
    rep = representative_laps(laps)
    pace = rep.groupby("driver_code")["lap_time_s"].median().sort_values()
    weather_summary: dict[str, Any] = {}
    if not w.empty:
        weather_summary = {
            "samples": len(w),
            "air_temp_c": [float(w["air_temp_c"].min()), float(w["air_temp_c"].max())],
            "track_temp_c": [float(w["track_temp_c"].min()), float(w["track_temp_c"].max())],
            "humidity_pct": [float(w["humidity_pct"].min()), float(w["humidity_pct"].max())],
            "rainfall_any": rain,
        }
    coverage: dict[str, Any] = {}
    if not pos.empty:
        span = pos.groupby("driver_number")["session_time_s"].agg(["min", "max", "count"])
        coverage = {
            "samples_per_car_min": int(span["count"].min()),
            "samples_per_car_max": int(span["count"].max()),
            "approx_hz": round(float((span["count"] / (span["max"] - span["min"])).median()), 2),
            "x_range_m": [float(pos["x_m"].min()), float(pos["x_m"].max())],
            "y_range_m": [float(pos["y_m"].min()), float(pos["y_m"].max())],
        }

    start_compounds = (
        laps[laps["lap_number"] == 1].set_index("driver_code")["compound"].value_counts().to_dict()
    )
    notes = (
        f"Start compounds: {start_compounds}. The synthetic engine currently starts every car on "
        "MEDIUM; real start tyres should come from this table (M1/M2).",
        "Not observable in this data: fuel load, tyre temperatures/pressures, car setup, the "
        "team's reasons for pit calls, driver intent. These stay estimated or assumed.",
        "Positions are the timing feed's car location (~4 Hz, metres), not a measured racing line.",
        "Lap 1 includes the standing start and is excluded from pace statistics.",
    )

    return QualityReport(
        event=tables.event,
        source=source,
        checks=tuple(checks),
        table_rows={
            "laps": len(laps),
            "results": len(results),
            "weather": len(w),
            "track_status": len(tables.track_status),
            "race_control": len(tables.race_control),
            "positions": len(pos),
        },
        lap_missingness={k: v for k, v in missing.items() if v},
        stints=_records(stints),
        pit_stops=pit_rows,
        weather_summary=weather_summary,
        track_status_changes=_records(tables.track_status),
        race_control_categories={
            str(k): int(v) for k, v in tables.race_control["category"].value_counts().items()
        },
        position_coverage=coverage,
        green_lap_pace={
            "representative_laps": len(rep),
            "overall_median_s": round(float(rep["lap_time_s"].median()), 3) if len(rep) else None,
            "per_driver_median_s": {k: round(float(v), 3) for k, v in pace.items()},
        },
        notes=notes,
    )


def render_markdown(report: QualityReport) -> str:
    ev, src = report.event, report.source
    lines = [
        f"# Data quality: {ev['event_name']} {ev['event_date'][:4]} (round {ev['round']}, race)",
        "",
        "> Generated by `scripts/fetch_monza_2024.py`. Do not edit by hand.",
        "",
        f"- Source: {src.get('source', '?')} {src.get('source_version', '')}; "
        f"retrieved {src.get('retrieved_utc', '?')}",
        f"- Cross-check source: {src.get('reference', 'none')}",
        f"- Overall: **{report.worst}** "
        f"({sum(c.status == 'PASS' for c in report.checks)} pass, "
        f"{sum(c.status == 'WARN' for c in report.checks)} warn, "
        f"{sum(c.status == 'FAIL' for c in report.checks)} fail)",
        "",
        "## Checks",
        "",
        "| | Check | Detail |",
        "|---|---|---|",
    ]
    lines += [f"| {c.status} | {c.title} | {c.detail} |" for c in report.checks]
    lines += ["", "## Tables", "", "| Table | Rows |", "|---|---:|"]
    lines += [f"| {k} | {v} |" for k, v in report.table_rows.items()]
    lines += ["", "Missing values in lap table (column: count):", ""]
    lines += [f"- `{k}`: {v}" for k, v in report.lap_missingness.items()] or ["- none"]

    lines += ["", "## Pit stops", "", "| Driver | In-lap | From | To |", "|---|---:|---|---|"]
    lines += [
        f"| {p['driver']} | {p['in_lap']} | {p['from']} | {p['to']} |" for p in report.pit_stops
    ]

    lines += [
        "",
        "## Stints",
        "",
        "| Driver | Stint | Compound | Laps | From–to | Tyre life at start | New |",
        "|---|---:|---|---:|---|---:|---|",
    ]
    for s in report.stints:
        lines.append(
            f"| {s['driver_code']} | {s['stint']} | {s['compound']} | {s['laps']} | "
            f"{s['first_lap']}–{s['last_lap']} | {s['tyre_life_start']} | {s['fresh_tyre']} |"
        )

    pace = report.green_lap_pace
    lines += [
        "",
        "## Green-flag pace (representative laps)",
        "",
        f"{pace['representative_laps']} laps after excluding lap 1, pit in/out laps, non-green, "
        f"inaccurate, deleted and generated laps. Overall median **{pace['overall_median_s']} s**.",
        "",
        "| Driver | Median lap (s) |",
        "|---|---:|",
    ]
    lines += [f"| {d} | {t} |" for d, t in pace["per_driver_median_s"].items()]

    lines += ["", "## Weather", ""]
    lines += [f"- {k}: {v}" for k, v in report.weather_summary.items()] or ["- no data"]
    lines += ["", "## Track status changes", ""]
    lines += [
        f"- t={s['session_time_s']:.1f}s: {s['status']} ({s['status_name']}) {s['message']}"
        for s in report.track_status_changes
    ] or ["- none"]
    lines += ["", "## Race control messages by category", ""]
    lines += [f"- {k}: {v}" for k, v in report.race_control_categories.items()]
    lines += ["", "## Position data", ""]
    lines += [f"- {k}: {v}" for k, v in report.position_coverage.items()] or ["- not loaded"]
    lines += ["", "## Notes and limits", ""]
    lines += [f"- {n}" for n in report.notes]
    return "\n".join(lines) + "\n"


__all__ = [
    "TRACK_STATUS_NAMES",
    "Check",
    "QualityReport",
    "build_quality_report",
    "render_markdown",
    "representative_laps",
    "stint_table",
]
