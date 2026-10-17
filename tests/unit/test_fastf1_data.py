"""Offline tests for the FastF1 normalisers, quality report and manifests.

Inputs are small synthetic frames shaped like FastF1's (same column names and dtypes), so these
tests never touch the network. The real fetch is covered by the opt-in ``network`` test.
"""

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from f1ar.data import jolpica_adapter
from f1ar.data.fastf1_adapter import (
    RaceTables,
    missing_classification,
    normalize_laps,
    normalize_positions,
    normalize_results,
    normalize_track_status,
    normalize_weather,
    track_outline,
)
from f1ar.data.manifests import DatasetManifest, verify_files, write_table
from f1ar.data.quality import build_quality_report, render_markdown, representative_laps
from f1ar.domain.identities import RaceRoster
from f1ar.domain.provenance import SourceType

TD = pd.Timedelta
CARS = {16: ("LEC", "Ferrari", 1), 55: ("SAI", "Ferrari", 2)}
LAPS = 4


def raw_laps(**overrides: Any) -> pd.DataFrame:
    rows = []
    for number, (code, team, _) in CARS.items():
        t = 100.0
        for lap in range(1, LAPS + 1):
            lap_time = 90.0 if lap == 1 else 85.0 + 0.1 * lap
            rows.append(
                {
                    "Time": TD(t + lap_time, unit="s"),
                    "Driver": code,
                    "DriverNumber": str(number),
                    "LapTime": TD(lap_time, unit="s"),
                    "LapNumber": float(lap),
                    "Stint": 1.0 if lap < 3 else 2.0,
                    "PitOutTime": TD(t, unit="s") if lap == 3 else pd.NaT,
                    "PitInTime": TD(t + lap_time, unit="s") if lap == 2 else pd.NaT,
                    "Sector1Time": pd.NaT if lap == 1 else TD(28.0, unit="s"),
                    "Sector2Time": TD(29.0, unit="s"),
                    "Sector3Time": TD(28.0, unit="s"),
                    "SpeedST": 320.0,
                    "Compound": "MEDIUM" if lap < 3 else "HARD",
                    "TyreLife": float(lap) if lap < 3 else float(lap - 2),
                    "FreshTyre": True,
                    "Team": team,
                    "LapStartTime": TD(t, unit="s"),
                    "Position": float(CARS[number][2]),
                    "TrackStatus": "1",
                    "IsAccurate": lap not in (1, 2, 3),
                    "Deleted": False,
                    "FastF1Generated": False,
                }
            )
            t += lap_time
    df = pd.DataFrame(rows)
    for key, value in overrides.items():
        df[key] = value
    return df


def raw_results() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "DriverNumber": str(n),
                "Abbreviation": code,
                "FullName": code,
                "TeamName": team,
                "GridPosition": float(pos),
                "Position": float(pos),
                "ClassifiedPosition": str(pos),
                "Status": "Finished",
                "Laps": float(LAPS),
                "Points": 0.0,
            }
            for n, (code, team, pos) in CARS.items()
        ]
    )


def roster() -> RaceRoster:
    return RaceRoster.model_validate(
        {
            "schema_version": 1,
            "event_id": "test_gp",
            "season": 2024,
            "round": 16,
            "circuit_id": "monza",
            "race_date": "2024-09-01",
            "source_type": "historical_verified",
            "source": {"name": "t", "url": "t", "retrieved_utc": "t", "license_review_status": "t"},
            "entries": [
                {
                    "driver_id": f"{code.lower()}_x_2024",
                    "person_id": f"{code.lower()}_x",
                    "code": code,
                    "car_number": n,
                    "given_name": code,
                    "family_name": "X",
                    "team_id": "ferrari_2024",
                    "team_name": "Ferrari",
                    "grid_position": pos,
                    "external_ids": {},
                }
                for n, (code, _, pos) in CARS.items()
            ],
        }
    )


def tables(laps: pd.DataFrame | None = None, results: pd.DataFrame | None = None) -> RaceTables:
    weather = pd.DataFrame(
        {
            "Time": [TD(0, unit="s"), TD(60, unit="s")],
            "AirTemp": [30.0, 31.0],
            "TrackTemp": [45.0, 46.0],
            "Humidity": [35.0, 36.0],
            "Pressure": [1000.0, 1000.0],
            "Rainfall": [False, False],
            "WindSpeed": [1.0, 1.2],
            "WindDirection": [100, 110],
        }
    )
    status = pd.DataFrame({"Time": [TD(5, unit="s")], "Status": ["1"], "Message": ["AllClear"]})
    pos = {
        str(n): pd.DataFrame(
            {
                "SessionTime": [TD(s, unit="s") for s in (100.0, 150.0, 400.0)],
                "X": [1000.0, 2000.0, 3000.0],
                "Y": [-500.0, 0.0, 500.0],
                "Z": [10.0, 10.0, 10.0],
                "Status": ["OnTrack", "OnTrack", "OffTrack"],
            }
        )
        for n in CARS
    }
    return RaceTables(
        event={
            "event_name": "Test GP",
            "round": 16,
            "event_date": "2024-09-01",
            "session_name": "Race",
        },
        laps=normalize_laps(laps if laps is not None else raw_laps()),
        results=normalize_results(results if results is not None else raw_results()),
        weather=normalize_weather(weather),
        track_status=normalize_track_status(status),
        race_control=pd.DataFrame({"category": pd.Series(["Flag"], dtype="string")}),
        positions=normalize_positions(pos),
    )


def report_for(t: RaceTables, **kw: Any) -> dict[str, str]:
    report = build_quality_report(t, roster(), source={}, expected_total_laps=LAPS, **kw)
    return {c.check_id: c.status for c in report.checks}


# --- normalisers ------------------------------------------------------------------------------


def test_laps_are_converted_to_seconds_with_nullable_types() -> None:
    laps = normalize_laps(raw_laps())
    first = laps.iloc[0]
    assert first["driver_number"] == 16 and first["lap_number"] == 1
    assert first["lap_time_s"] == 90.0
    assert pd.isna(first["sector1_s"])  # missing stays missing, never 0
    assert str(laps["tyre_life_laps"].dtype) == "Int64"
    assert str(laps["fresh_tyre"].dtype) == "boolean"
    assert laps["pit_in_time_s"].notna().sum() == 2


def test_unknown_compound_strings_become_missing() -> None:
    laps = normalize_laps(raw_laps(Compound=["UNKNOWN", "", "nan", None] * 2))
    assert laps["compound"].isna().all()


def test_positions_are_converted_to_metres() -> None:
    t = tables()
    assert t.positions["x_m"].tolist()[:3] == [100.0, 200.0, 300.0]
    assert set(t.positions["driver_number"]) == set(CARS)


def test_track_outline_uses_one_lap_window_and_on_track_samples() -> None:
    t = tables()
    outline = track_outline(t.positions, t.laps, driver_number=16, lap_number=1)
    # lap 1 runs 100..190 s: samples at 100 and 150 (the 400 s one is outside / off track)
    assert outline["session_time_s"].tolist() == [100.0, 150.0]
    with pytest.raises(ValueError):
        track_outline(t.positions, t.laps, driver_number=16, lap_number=99)


def test_representative_laps_exclude_start_pits_and_flags() -> None:
    reps = representative_laps(tables().laps)
    # lap 1 (start), lap 2 (pit in), lap 3 (pit out) excluded; lap 4 kept, per car
    assert sorted(reps["lap_number"].unique().tolist()) == [4]


# --- quality checks -----------------------------------------------------------------------------


def test_consistent_data_passes() -> None:
    statuses = report_for(tables())
    assert statuses["roster_numbers"] == statuses["roster_grid"] == "PASS"
    assert statuses["lap_sequence"] == statuses["tyre_life_continuity"] == "PASS"
    assert statuses["classification"] == statuses["neutralisations"] == "PASS"


def test_grid_mismatch_fails() -> None:
    results = raw_results()
    results.loc[0, "GridPosition"] = 9.0
    assert report_for(tables(results=results))["roster_grid"] == "FAIL"


def test_duplicate_lap_fails_sequence_check() -> None:
    laps = raw_laps()
    laps.loc[1, "LapNumber"] = 1.0
    assert report_for(tables(laps=laps))["lap_sequence"] == "FAIL"


def test_retirement_lap_row_is_explained_but_a_finisher_mismatch_fails() -> None:
    """FastF1 keeps a row for the lap a car retired on (seen across 2024, M4): WARN, not FAIL.
    The same +1 for a car classified as finished is unexplained and still FAILs."""
    results = raw_results()
    results.loc[0, "Laps"] = float(LAPS - 1)
    results.loc[0, "Status"] = "Retired"
    assert report_for(tables(results=results))["laps_vs_results"] == "WARN"
    results.loc[0, "Status"] = "Finished"
    assert report_for(tables(results=results))["laps_vs_results"] == "FAIL"
    dsq = raw_results()
    dsq.loc[0, "Laps"] = 0.0
    dsq.loc[0, "Status"] = "Disqualified"
    assert report_for(tables(results=dsq))["laps_vs_results"] == "WARN"


def test_tyre_life_jump_and_safety_car_warn() -> None:
    laps = raw_laps()
    laps.loc[1, "TyreLife"] = 5.0
    laps.loc[3, "TrackStatus"] = "14"
    statuses = report_for(tables(laps=laps))
    assert statuses["tyre_life_continuity"] == "WARN"
    assert statuses["neutralisations"] == "WARN"


def test_reference_results_disagreement_fails() -> None:
    ref = [
        {"number": 16, "position": 2, "laps": LAPS, "status": "Finished", "grid": 1},
        {"number": 55, "position": 1, "laps": LAPS, "status": "Finished", "grid": 2},
    ]
    assert report_for(tables(), reference_results=ref)["results_crosscheck"] == "FAIL"


def test_markdown_report_renders() -> None:
    report = build_quality_report(tables(), roster(), source={}, expected_total_laps=LAPS)
    text = render_markdown(report)
    assert text.startswith("# Data quality: Test GP 2024")
    assert "| PASS | Starting grid matches the roster |" in text


# --- manifests, jolpica ----------------------------------------------------------------------


def test_manifest_hashes_detect_changed_files(tmp_path: Path) -> None:
    entry = write_table(
        tables().laps,
        tmp_path / "data" / "laps.parquet",
        root=tmp_path,
        source_type=SourceType.HISTORICAL_VERIFIED,
        description="laps",
    )
    manifest = DatasetManifest(
        dataset_id="t",
        event="t",
        source="t",
        source_api_version="t",
        retrieved_utc="t",
        session_key="t",
        license_review_status="t",
        files=[entry],
        unit_conventions={},
        missingness_notes=[],
    )
    assert verify_files(manifest, tmp_path) == {"data/laps.parquet": True}
    (tmp_path / "data" / "laps.parquet").write_bytes(b"tampered")
    assert verify_files(manifest, tmp_path) == {"data/laps.parquet": False}
    again = DatasetManifest.model_validate_json(manifest.model_dump_json())
    assert again == manifest and "laps.parquet" in manifest.to_yaml()


def test_jolpica_result_records() -> None:
    payload = {
        "MRData": {
            "RaceTable": {
                "Races": [
                    {
                        "Results": [
                            {
                                "number": "16",
                                "position": "1",
                                "laps": "53",
                                "status": "Finished",
                                "grid": "4",
                            }
                        ]
                    }
                ]
            }
        }
    }
    assert jolpica_adapter.result_records(payload) == [
        {"number": 16, "position": 1, "laps": 53, "status": "Finished", "grid": 4}
    ]


def test_missing_classification_detects_blank_jolpica_columns() -> None:
    # H0: a throttled Jolpica call leaves FastF1's classification blank (2021 Italian GP)
    assert missing_classification(raw_results()) == []
    blank = raw_results()
    blank["Laps"] = float("nan")
    blank["Status"] = ""
    assert missing_classification(blank) == ["Status", "Laps"]
    one_dns = raw_results()
    one_dns.loc[1, "Laps"] = float("nan")  # one car without laps is a real result, not a gap
    assert missing_classification(one_dns) == []
