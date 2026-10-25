"""The lap dataset for pace models and calibration (M4, plan §8.2, §8.4, §6.4 "derive").

One row per real lap. Every column is either

- a **feature**, known when the lap starts (``available``: ``pre_race`` = known before the race,
  ``lap_start`` = observable at the line where the lap begins, ``decision`` = the team's own
  choice for this lap, e.g. boxing at its end), or
- the **target** (``lap_time_s``), or
- **lap facts** used only to filter and audit (``during_lap``): what happened during the lap,
  e.g. its track-status string, accuracy flags, the end-of-lap position. These are never
  features: using them to predict the lap would leak the lap's own outcome.

The audit in ``tests/unit/test_lap_dataset.py`` changes everything that happens after a lap
starts and checks that the lap's feature values don't move.

Only real-data values are stored here (``historical_verified`` or ``estimated_from_data``).
Editorial persona values are joined at training time and labelled there, so the dataset itself
never mixes real and assumed numbers (plan §3.4).
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

from f1ar.data.quality import representative_laps
from f1ar.domain.provenance import SourceType
from f1ar.domain.refuelling import LAST_REFUELLING_SEASON

FEATURE_SCHEMA_VERSION = 1
DATASET_VERSION = 1

Availability = Literal["pre_race", "lap_start", "decision"]


@dataclass(frozen=True)
class Feature:
    name: str
    available: Availability
    source_type: SourceType
    description: str


V = SourceType.HISTORICAL_VERIFIED
E = SourceType.ESTIMATED_FROM_DATA

FEATURES: tuple[Feature, ...] = (
    Feature("circuit_id", "pre_race", V, "Jolpica circuit id"),
    Feature("driver_id", "pre_race", V, "season-specific driver id (roster)"),
    Feature("team_id", "pre_race", V, "team id (roster)"),
    Feature("grid_position", "pre_race", V, "official starting slot (0 = pit lane)"),
    Feature("scheduled_laps", "pre_race", V, "race distance in laps (winner's laps)"),
    Feature("lap_number", "lap_start", V, "1-based lap number"),
    Feature(
        "laps_remaining",
        "lap_start",
        E,
        "scheduled_laps - lap_number + 1 (fuel proxy); refuelling era: laps left in the stint",
    ),
    Feature("compound", "lap_start", V, "tyre compound on this lap"),
    Feature("tyre_age_start", "lap_start", E, "laps already on the set when the lap starts"),
    Feature("fresh_tyre", "lap_start", V, "the set was new when fitted (FastF1 FreshTyre)"),
    Feature("stint", "lap_start", V, "stint number"),
    Feature("stint_lap", "lap_start", E, "1-based lap index within the stint"),
    Feature("pit_out_lap", "lap_start", V, "the lap starts in the pit lane"),
    Feature("pit_in_lap", "decision", V, "the car boxes at the end of this lap"),
    Feature("position_start", "lap_start", V, "position when the lap starts (grid on lap 1)"),
    Feature(
        "gap_ahead_s",
        "lap_start",
        E,
        "seconds behind the car that crossed the line just before, same lap (NaN on lap 1)",
    ),
    Feature(
        "gap_behind_prev_s",
        "lap_start",
        E,
        "seconds ahead of the next car at the PREVIOUS line crossing (the gap to the car behind"
        " at this crossing isn't known yet when the lap starts; found by the leakage audit)",
    ),
    Feature("track_status_start", "lap_start", V, "track status code when the lap starts"),
    Feature("air_temp_c", "lap_start", V, "latest weather sample at lap start"),
    Feature("track_temp_c", "lap_start", V, "latest weather sample at lap start"),
    Feature("humidity_pct", "lap_start", V, "latest weather sample at lap start"),
    Feature("rainfall_start", "lap_start", V, "rain flag of the latest weather sample"),
    Feature("wind_speed_ms", "lap_start", V, "latest weather sample at lap start"),
)
FEATURE_NAMES: tuple[str, ...] = tuple(f.name for f in FEATURES)

TARGET = "lap_time_s"

LAP_FACTS: tuple[str, ...] = (
    "lap_status",
    "position_end",
    "is_accurate",
    "deleted",
    "fastf1_generated",
    "representative",
)
"""During-lap facts: for filtering and auditing only, never features."""

KEYS: tuple[str, ...] = ("event_id", "season", "round", "driver_number", "driver_code")
TIMES: tuple[str, ...] = ("lap_start_time_s", "lap_end_time_s")
"""Session times (seconds) of the lap's start and end, for audits and joins."""


def _asof(left: pd.DataFrame, right: pd.DataFrame, time_col: str, cols: list[str]) -> pd.DataFrame:
    """For each left row, the latest ``right`` row at or before ``left.lap_start_time_s``."""
    r = right[[time_col, *cols]].copy()
    r["_t"] = pd.to_numeric(r[time_col], errors="coerce").astype(float)
    r = r.drop(columns=[time_col]).dropna(subset=["_t"]).sort_values("_t")
    keys = pd.DataFrame(
        {"_row": range(len(left)), "_t": pd.to_numeric(left["lap_start_time_s"]).astype(float)}
    )
    known = keys.dropna(subset=["_t"]).sort_values("_t")  # a lap without a start time: no join
    merged = pd.merge_asof(known, r, on="_t", direction="backward").set_index("_row")
    return merged.reindex(range(len(left)))[cols].set_axis(left.index)


def race_laps(
    laps: pd.DataFrame,
    results: pd.DataFrame,
    weather: pd.DataFrame,
    track_status: pd.DataFrame,
    *,
    event_id: str,
    season: int,
    round_: int,
    circuit_id: str,
    driver_ids: dict[int, str],
    team_ids: dict[int, str],
) -> pd.DataFrame:
    """The dataset rows of one race (pure: the inputs are the fetched tables)."""
    df = laps.sort_values(["driver_number", "lap_number"]).reset_index(drop=True).copy()
    res = results.set_index(results["driver_number"].astype(int))
    numbers = df["driver_number"].astype(int)
    scheduled = int(results["laps"].max())

    out = pd.DataFrame(
        {
            "event_id": event_id,
            "season": season,
            "round": round_,
            "circuit_id": circuit_id,
            "driver_number": numbers,
            "driver_code": df["driver_code"].astype(str),
            "driver_id": numbers.map(driver_ids),
            "team_id": numbers.map(team_ids),
            "grid_position": numbers.map(res["grid_position"]).astype("Int64"),
            "scheduled_laps": scheduled,
            "lap_number": df["lap_number"].astype(int),
        }
    )
    out["laps_remaining"] = scheduled - out["lap_number"] + 1
    if season <= LAST_REFUELLING_SEASON:  # H1b: the fuel on board follows the stints
        stint_end = df.groupby(["driver_number", "stint"], dropna=False)["lap_number"].transform(
            "max"
        )
        out["laps_remaining"] = (stint_end.astype(int) - out["lap_number"] + 1).clip(lower=1)
    out["compound"] = df["compound"].astype("string")
    out["tyre_age_start"] = (df["tyre_life_laps"] - 1).astype("Float64")
    out["fresh_tyre"] = df["fresh_tyre"].astype("boolean")
    out["stint"] = df["stint"].astype("Int64")
    out["stint_lap"] = df.groupby(["driver_number", "stint"], dropna=False).cumcount() + 1
    out["pit_out_lap"] = df["pit_out_time_s"].notna()
    out["pit_in_lap"] = df["pit_in_time_s"].notna()

    # session times of the lap (for as-of joins and the leakage audit)
    out["lap_start_time_s"] = df["lap_start_time_s"].astype(float)
    out["lap_end_time_s"] = df["lap_end_time_s"].astype(float)

    # position when the lap starts = position at the end of the previous lap (grid on lap 1)
    prev_pos = df.groupby("driver_number")["position"].shift(1)
    out["position_start"] = prev_pos.where(out["lap_number"] > 1, out["grid_position"])
    out["position_start"] = pd.to_numeric(out["position_start"], errors="coerce").astype("Int64")

    # gaps at the line where the lap starts: crossings that ended lap (n - 1), same lap number
    ends = out[["driver_number", "lap_number", "lap_end_time_s"]].dropna()
    gap_ahead: dict[tuple[int, int], float] = {}
    gap_behind: dict[tuple[int, int], tuple[float, float]] = {}  # (gap, when it was observed)
    for lap_key, group in ends.groupby("lap_number"):
        lap = int(str(lap_key))
        ordered = group.sort_values("lap_end_time_s")
        t = ordered["lap_end_time_s"].to_list()
        d = ordered["driver_number"].to_list()
        for i, number in enumerate(d):
            key = (int(number), lap + 1)  # the crossing at the end of lap n starts lap n+1
            if i > 0:
                gap_ahead[key] = t[i] - t[i - 1]
            if i + 1 < len(t):
                # known only once the car behind crosses, i.e. after this lap has started: it is
                # attached to the NEXT lap (lap n+2), when it is fully observed
                gap_behind[(int(number), lap + 2)] = (t[i + 1] - t[i], t[i + 1])
    keys = list(zip(out["driver_number"], out["lap_number"], strict=True))
    out["gap_ahead_s"] = pd.Series([gap_ahead.get(k) for k in keys], dtype="Float64")
    starts = out["lap_start_time_s"].to_list()
    behind: list[float | None] = []
    for k, start_t in zip(keys, starts, strict=True):
        value = gap_behind.get(k)
        # a car far behind (about to be lapped) may cross after this lap began: not yet known
        behind.append(value[0] if value is not None and value[1] <= start_t else None)
    out["gap_behind_prev_s"] = pd.Series(behind, dtype="Float64")

    # conditions at the line where the lap starts (latest observation at or before it)
    status = track_status.rename(columns={"status": "track_status_start"})
    out["track_status_start"] = _asof(out, status, "session_time_s", ["track_status_start"])[
        "track_status_start"
    ].astype("string")
    wx = weather.rename(columns={"rainfall": "rainfall_start"})
    wcols = ["air_temp_c", "track_temp_c", "humidity_pct", "rainfall_start", "wind_speed_ms"]
    out[wcols] = _asof(out, wx, "session_time_s", wcols)

    # target and during-lap facts
    out[TARGET] = df["lap_time_s"].astype("Float64")
    out["lap_status"] = df["track_status"].astype("string")
    out["position_end"] = pd.to_numeric(df["position"], errors="coerce").astype("Int64")
    out["is_accurate"] = df["is_accurate"].astype("boolean")
    out["deleted"] = df["deleted"].astype("boolean")
    out["fastf1_generated"] = df["fastf1_generated"].astype("boolean")
    rep = representative_laps(df)
    out["representative"] = out.index.isin(rep.index)
    return out


def race_from_dir(data_dir: Path, roster_path: Path) -> pd.DataFrame:
    """``race_laps`` for a fetched event (``data/processed/<event>`` + its roster file)."""
    roster = json.loads(roster_path.read_text(encoding="utf-8"))
    entries = roster["entries"]
    return race_laps(
        pd.read_parquet(data_dir / "laps.parquet"),
        pd.read_parquet(data_dir / "results.parquet"),
        pd.read_parquet(data_dir / "weather.parquet"),
        pd.read_parquet(data_dir / "track_status.parquet"),
        event_id=roster["event_id"],
        season=int(roster["season"]),
        round_=int(roster["round"]),
        circuit_id=str(roster["circuit_id"]),
        driver_ids={int(e["car_number"]): e["driver_id"] for e in entries},
        team_ids={int(e["car_number"]): e["team_id"] for e in entries},
    )


# --- the season dataset, its race groups and splits --------------------------------------------

WET_COMPOUNDS = ("INTERMEDIATE", "WET")
WET_RAIN_SHARE = 0.05
"""A race counts as wet with rain at this share of lap starts (or any wet-tyre lap). A single
stray rain sample doesn't: 2024 Spain had one (1 of 154 samples, 1.5 % of lap starts) and no
wet-tyre laps; it was a dry race."""


def is_wet_race(race: pd.DataFrame) -> bool:
    """Any lap on intermediates / wets, or rain at >= 5 % of lap starts."""
    rain_share = float(race["rainfall_start"].fillna(False).astype(bool).mean())
    wet_tyres = race["compound"].isin(WET_COMPOUNDS).any()
    return bool(wet_tyres or rain_share >= WET_RAIN_SHARE)


@dataclass(frozen=True)
class Splits:
    wet: tuple[str, ...]
    dry_test: tuple[str, ...]
    dry_train: tuple[str, ...]
    rule: str

    def group_of(self, event_id: str) -> str:
        if event_id in self.wet:
            return "wet"
        return "dry_test" if event_id in self.dry_test else "dry_train"


SPLIT_RULE = (
    "wet = any intermediate/wet tyre lap or rain at >= 5% of lap starts; among dry races, rounds "
    "divisible by 4 are held out (dry_test), the rest are dry_train; leave-one-race-out folds "
    "run over dry_train. Rule fixed before any model was fitted."
)


def make_splits(rounds: dict[str, int], wet: set[str]) -> Splits:
    """Deterministic split by rule (not by looking at model results)."""
    dry = sorted((e for e in rounds if e not in wet), key=lambda e: rounds[e])
    return Splits(
        wet=tuple(sorted(wet, key=lambda e: rounds[e])),
        dry_test=tuple(e for e in dry if rounds[e] % 4 == 0),
        dry_train=tuple(e for e in dry if rounds[e] % 4 != 0),
        rule=SPLIT_RULE,
    )


def file_sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def season_events(root: Path, season: int) -> list[tuple[str, Path, Path]]:
    """``(event_id, data_dir, roster_path)`` of every fetched race of the season, by round."""
    found: list[tuple[int, str, Path, Path]] = []
    for roster_path in (root / "configs" / "rosters").glob(f"{season}_*.json"):
        roster = json.loads(roster_path.read_text(encoding="utf-8"))
        data_dir = root / "data" / "processed" / roster["event_id"]
        if (data_dir / "laps.parquet").exists():
            found.append((int(roster["round"]), roster["event_id"], data_dir, roster_path))
    return [(e, d, r) for _, e, d, r in sorted(found)]
