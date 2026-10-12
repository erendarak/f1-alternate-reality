"""Lap dataset (M4): conventions, splits, and the leakage audit (plan §8.2 "no leakage").

The audit is a property test on a synthetic race: changing anything that happens after time T
(lap times and outcomes, later weather and track-status samples) must not change the feature
values of any lap that started at or before T.
"""

import json
import math
from pathlib import Path

import pandas as pd
import pytest

from f1ar.calibration.lap_dataset import (
    FEATURE_NAMES,
    FEATURES,
    LAP_FACTS,
    TARGET,
    is_wet_race,
    make_splits,
    race_laps,
)
from f1ar.calibration.reference_metrics import red_flag_periods

CARS = (1, 16, 44)
LAPS = 6


def synthetic_race(seed_shift: float = 0.0) -> dict[str, pd.DataFrame]:
    """3 cars x 6 laps; car 16 pits at the end of lap 3 (MEDIUM -> HARD)."""
    rows = []
    start = 1000.0
    for k, number in enumerate(CARS):
        t = start + 0.5 * k  # staggered start
        for lap in range(1, LAPS + 1):
            lap_time = 90.0 + k * 0.4 + 0.1 * lap + (2.0 if lap == 1 else 0.0) + seed_shift
            pit_in = number == 16 and lap == 3
            pit_out = number == 16 and lap == 4
            stint = 2 if number == 16 and lap >= 4 else 1
            life = lap - 3 if stint == 2 else lap
            rows.append(
                {
                    "driver_number": number,
                    "driver_code": {1: "VER", 16: "LEC", 44: "HAM"}[number],
                    "team": "T",
                    "lap_number": lap,
                    "lap_time_s": lap_time,
                    "sector1_s": None,
                    "sector2_s": None,
                    "sector3_s": None,
                    "lap_start_time_s": t,
                    "lap_end_time_s": t + lap_time,
                    "pit_in_time_s": t + lap_time - 5 if pit_in else None,
                    "pit_out_time_s": t + 3 if pit_out else None,
                    "stint": stint,
                    "compound": "HARD" if stint == 2 else "MEDIUM",
                    "tyre_life_laps": life,
                    "fresh_tyre": True,
                    "position": CARS.index(number) + 1,
                    "track_status": "1",
                    "speed_trap_kph": 330.0,
                    "is_accurate": lap > 1,
                    "deleted": False,
                    "fastf1_generated": False,
                }
            )
            t += lap_time
    laps = pd.DataFrame(rows)
    results = pd.DataFrame(
        {
            "driver_number": list(CARS),
            "driver_code": ["VER", "LEC", "HAM"],
            "grid_position": [1, 2, 3],
            "laps": [LAPS] * 3,
        }
    )
    weather = pd.DataFrame(
        {
            "session_time_s": [900.0 + 60 * i for i in range(12)],
            "air_temp_c": [25.0 + 0.1 * i for i in range(12)],
            "track_temp_c": [40.0 + 0.2 * i for i in range(12)],
            "humidity_pct": [50.0] * 12,
            "rainfall": [False] * 12,
            "wind_speed_ms": [1.0] * 12,
        }
    )
    status = pd.DataFrame({"session_time_s": [900.0, 1300.0], "status": ["1", "2"]})
    return {"laps": laps, "results": results, "weather": weather, "track_status": status}


def build(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return race_laps(
        tables["laps"],
        tables["results"],
        tables["weather"],
        tables["track_status"],
        event_id="2024_test_gp",
        season=2024,
        round_=1,
        circuit_id="test",
        driver_ids={
            1: "max_verstappen_2024",
            16: "charles_leclerc_2024",
            44: "lewis_hamilton_2024",
        },
        team_ids={1: "red_bull", 16: "ferrari", 44: "mercedes"},
    )


def test_schema_separates_features_target_and_lap_facts() -> None:
    names = set(FEATURE_NAMES)
    assert TARGET not in names and not names & set(LAP_FACTS)
    assert "position_end" in LAP_FACTS and "lap_status" in LAP_FACTS  # end-of-lap outcomes
    assert {f.available for f in FEATURES} <= {"pre_race", "lap_start", "decision"}
    # only real-data provenance in the dataset itself (personas are joined later, labelled)
    assert {f.source_type.value for f in FEATURES} <= {
        "historical_verified",
        "estimated_from_data",
    }


def test_lap_conventions() -> None:
    df = build(synthetic_race()).set_index(["driver_code", "lap_number"])
    # tyre age at lap START is one less than FastF1's TyreLife (which counts the lap itself)
    assert df.loc[("VER", 1), "tyre_age_start"] == 0
    assert df.loc[("LEC", 4), "tyre_age_start"] == 0 and df.loc[("LEC", 4), "stint_lap"] == 1
    assert df.loc[("LEC", 3), "pit_in_lap"] and df.loc[("LEC", 4), "pit_out_lap"]
    # position at lap start: grid on lap 1, previous lap's end afterwards
    assert df.loc[("HAM", 1), "position_start"] == 3
    assert df.loc[("HAM", 2), "position_start"] == df.loc[("HAM", 1), "position_end"]
    assert df.loc[("VER", 6), "laps_remaining"] == 1
    # the car that crossed first has no gap ahead; gaps are line-crossing differences
    assert pd.isna(df.loc[("VER", 2), "gap_ahead_s"])
    expected = df.loc[("LEC", 1), "lap_end_time_s"] - df.loc[("VER", 1), "lap_end_time_s"]
    assert df.loc[("LEC", 2), "gap_ahead_s"] == pytest.approx(expected)
    assert pd.isna(df.loc[("LEC", 1), "gap_ahead_s"])  # standing start
    # conditions at lap start: the latest sample at or before it
    assert df.loc[("VER", 1), "track_status_start"] == "1"
    assert df.loc[("VER", 6), "track_status_start"] == "2"  # status changed at t=1300


def test_no_feature_uses_anything_after_the_lap_starts() -> None:
    """Perturb everything after T; features of laps that started by T must not change."""
    base = synthetic_race()
    before = build(base)
    for cut in sorted(before["lap_start_time_s"].unique())[1:-1]:
        changed = {k: v.copy() for k, v in base.items()}
        laps = changed["laps"]
        after_end = laps["lap_end_time_s"] > cut
        # outcomes of laps still running at T or later: time, end position, flags, end time
        laps.loc[after_end, "lap_time_s"] += 7.0
        laps.loc[after_end, "lap_end_time_s"] += 7.0
        laps.loc[after_end, "position"] = laps.loc[after_end, "position"].max() + 5
        laps.loc[after_end, "track_status"] = "4"
        laps.loc[after_end, "is_accurate"] = False
        started_later = laps["lap_start_time_s"] > cut
        laps.loc[started_later, "lap_start_time_s"] += 7.0
        weather = changed["weather"]
        late = weather["session_time_s"] > cut
        weather.loc[late, "air_temp_c"] += 10.0
        weather.loc[late, "rainfall"] = True
        status = changed["track_status"]
        changed["track_status"] = pd.concat(
            [status, pd.DataFrame({"session_time_s": [cut + 0.001], "status": ["4"]})]
        )
        after = build(changed)

        started = before["lap_start_time_s"] <= cut
        for name in FEATURE_NAMES:
            a = before.loc[started, name].to_list()
            b = after.loc[started, name].to_list()
            same = all(
                (x is pd.NA and y is pd.NA)
                or (
                    isinstance(x, float)
                    and isinstance(y, float)
                    and math.isnan(x)
                    and math.isnan(y)
                )
                or x == y
                for x, y in zip(a, b, strict=True)
            )
            assert same, f"feature {name!r} changed for a lap that started before t={cut}"
        # sanity: the perturbation did reach the target of laps still running at the cut
        running = started & (before["lap_end_time_s"] > cut)
        assert (after.loc[running, TARGET] != before.loc[running, TARGET]).all()


def test_wet_detection_and_splits() -> None:
    race = build(synthetic_race())
    assert not is_wet_race(race)
    wet = race.copy()
    wet.loc[wet.index[5], "compound"] = "INTERMEDIATE"
    assert is_wet_race(wet)
    drizzle = race.copy()
    drizzle.loc[drizzle.index[0], "rainfall_start"] = True  # 1 of 18 lap starts = 5.6 %
    assert is_wet_race(drizzle)
    many = pd.concat([race] * 3, ignore_index=True)  # 1 of 54 lap starts < 5 %: a stray sample
    many.loc[many.index[0], "rainfall_start"] = True
    assert not is_wet_race(many)

    rounds = {f"e{r}": r for r in range(1, 25)}
    splits = make_splits(rounds, wet={"e9", "e12", "e21"})
    assert splits.wet == ("e9", "e12", "e21")
    assert splits.dry_test == ("e4", "e8", "e16", "e20", "e24")
    assert len(splits.dry_train) == 24 - 3 - 5
    assert not set(splits.dry_train) & set(splits.dry_test)
    assert splits.group_of("e12") == "wet" and splits.group_of("e16") == "dry_test"


def test_red_flag_periods() -> None:
    status = pd.DataFrame(
        {"session_time_s": [0.0, 100.0, 160.0, 900.0, 950.0], "status": ["1", "5", "1", "5", "5"]}
    )
    assert red_flag_periods(status) == [(100.0, 160.0), (900.0, math.inf)]


ROOT = Path(__file__).resolve().parents[2]
MONZA = ROOT / "data" / "processed" / "2024_italian_gp"


@pytest.mark.skipif(not (MONZA / "laps.parquet").exists(), reason="Monza 2024 data not fetched")
def test_leakage_audit_on_the_real_monza_race() -> None:
    """The synthetic audit, repeated on a real race (real dtypes, missing values, lapped cars)."""
    tables = {
        "laps": pd.read_parquet(MONZA / "laps.parquet"),
        "results": pd.read_parquet(MONZA / "results.parquet"),
        "weather": pd.read_parquet(MONZA / "weather.parquet"),
        "track_status": pd.read_parquet(MONZA / "track_status.parquet"),
    }
    roster = json.loads((ROOT / "configs" / "rosters" / "2024_italian_gp.json").read_text("utf-8"))
    ids = {int(e["car_number"]): e["driver_id"] for e in roster["entries"]}
    teams = {int(e["car_number"]): e["team_id"] for e in roster["entries"]}

    def build_real(t: dict[str, pd.DataFrame]) -> pd.DataFrame:
        return race_laps(
            t["laps"],
            t["results"],
            t["weather"],
            t["track_status"],
            event_id="2024_italian_gp",
            season=2024,
            round_=16,
            circuit_id="monza",
            driver_ids=ids,
            team_ids=teams,
        )

    before = build_real(tables)
    assert len(before) == 1008 and before["driver_id"].notna().all()
    cut = float(before["lap_start_time_s"].quantile(0.5))
    changed = {k: v.copy() for k, v in tables.items()}
    laps = changed["laps"]
    after_end = laps["lap_end_time_s"].astype(float) > cut
    laps.loc[after_end, "lap_time_s"] = laps.loc[after_end, "lap_time_s"] + 9.0
    laps.loc[after_end, "lap_end_time_s"] = laps.loc[after_end, "lap_end_time_s"] + 9.0
    later = laps["lap_start_time_s"].astype(float) > cut
    laps.loc[later, "lap_start_time_s"] = laps.loc[later, "lap_start_time_s"] + 9.0
    late_wx = changed["weather"]["session_time_s"].astype(float) > cut
    changed["weather"].loc[late_wx, "air_temp_c"] = 99.0
    after = build_real(changed)
    started = before["lap_start_time_s"] <= cut
    for name in FEATURE_NAMES:
        pd.testing.assert_series_equal(
            before.loc[started, name], after.loc[started, name], check_names=False, obj=name
        )
