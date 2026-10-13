"""Data-derived engine inputs: start compounds, car pace offsets, and their effect in the engine."""

from pathlib import Path

import pandas as pd
import pytest

from f1ar.calibration.race_inputs import (
    MIN_LAPS,
    estimate_car_offsets,
    estimate_pit_loss,
    start_compounds,
)
from f1ar.data.quality import representative_laps
from f1ar.domain.driver_profiles import ProfileSet
from f1ar.domain.identities import RaceRoster
from f1ar.domain.provenance import SourceType
from f1ar.domain.race_state import Compound, RaceSetup
from f1ar.simulation import lightweight_engine as engine
from f1ar.simulation.scenarios import (
    default_lap_model,
    event_data_dir,
    pit_params_from_total_loss,
    setup_from_codes,
)
from tests.conftest import HAM, VER, run_to_end

# VER (car 1, Red Bull, pace skill 0.96) and HAM (car 44, Mercedes, 0.93)
MEDIANS = {1: 84.0, 44: 84.5}


def synthetic_laps(n_laps: int = 20, start: dict[int, str] | None = None) -> pd.DataFrame:
    rows = []
    for number, median in MEDIANS.items():
        for lap in range(1, n_laps + 1):
            rows.append(
                {
                    "driver_number": number,
                    "lap_number": lap,
                    "lap_time_s": median + (0.2 if lap % 2 else -0.2),
                    "compound": (start or {}).get(number, "MEDIUM") if lap == 1 else "HARD",
                    "pit_in_time_s": pd.NA,
                    "pit_out_time_s": pd.NA,
                    "track_status": "1",
                    "is_accurate": True,
                    "deleted": False,
                    "fastf1_generated": False,
                }
            )
    df = pd.DataFrame(rows)
    for col in ("is_accurate", "deleted", "fastf1_generated"):
        df[col] = df[col].astype("boolean")
    df["pit_in_time_s"] = df["pit_in_time_s"].astype("Float64")
    df["pit_out_time_s"] = df["pit_out_time_s"].astype("Float64")
    return df


def test_start_compounds_from_lap_one(monza_roster: RaceRoster) -> None:
    laps = synthetic_laps(start={1: "HARD", 44: "INTERMEDIATE"})
    out = start_compounds(laps, monza_roster)
    # M4: wet starts are modelled (engine v0.8.0); anything unknown is still left out
    assert out == {VER: Compound.HARD, HAM: Compound.INTERMEDIATE}


def test_car_offsets_remove_the_driver_skill_term(
    monza_roster: RaceRoster, monza_profiles: ProfileSet
) -> None:
    params = default_lap_model()
    est = estimate_car_offsets(synthetic_laps(), monza_roster, monza_profiles, params)
    # car-only pace = median - skill_gap * (1 - skill)
    ver_car = 84.0 - params.skill_gap_s * (1 - 0.96)
    ham_car = 84.5 - params.skill_gap_s * (1 - 0.93)
    assert est.offsets_s["red_bull_2024"] == 0.0
    assert est.offsets_s["mercedes_2024"] == pytest.approx(ham_car - ver_car, abs=1e-3)
    # teams without data get the median offset, so every roster team is covered
    assert {e.team_id for e in monza_roster.entries} <= set(est.offsets_s)
    assert est.note.startswith("estimated_from_data")


def test_drivers_with_too_few_laps_are_excluded(
    monza_roster: RaceRoster, monza_profiles: ProfileSet
) -> None:
    laps = synthetic_laps(n_laps=MIN_LAPS)  # lap 1 is never representative -> MIN_LAPS - 1
    with pytest.raises(ValueError, match="enough"):
        estimate_car_offsets(laps, monza_roster, monza_profiles, default_lap_model())


def test_car_offset_slows_only_that_car(isolated_setup: RaceSetup) -> None:
    two_car_setup = isolated_setup  # no traffic: VER can't be affected through HAM
    drivers = tuple(
        d.model_copy(update={"car_pace_offset_s": 1.0}) if d.driver_id == HAM else d
        for d in two_car_setup.drivers
    )
    slow = two_car_setup.model_copy(update={"drivers": drivers})
    base = run_to_end(engine.start(engine.create_world(two_car_setup, seed=3, world_id="w")))
    alt = run_to_end(engine.start(engine.create_world(slow, seed=3, world_id="w")))
    laps = two_car_setup.track.total_laps
    ham_base, ham_alt = base.car(HAM).finish_time_s, alt.car(HAM).finish_time_s
    assert ham_base is not None and ham_alt is not None
    assert ham_alt - ham_base == pytest.approx(laps * 1.0, abs=0.5)
    assert alt.car(VER).finish_time_s == base.car(VER).finish_time_s


def test_setup_uses_data_when_given(
    monza_roster: RaceRoster, monza_profiles: ProfileSet, tmp_path: Path
) -> None:
    synthetic_laps(start={1: "HARD"}).to_parquet(tmp_path / "laps.parquet")
    setup = setup_from_codes(monza_roster, monza_profiles, ["VER", "HAM"], data_dir=tmp_path)
    assert setup.driver(VER).start_compound is Compound.HARD
    assert setup.driver(HAM).start_compound is Compound.MEDIUM
    assert setup.driver(VER).car_pace_offset_s == 0.0 < setup.driver(HAM).car_pace_offset_s
    assert set(setup.inputs) == {
        "start_compounds",
        "car_pace_offsets",
        "pit_loss",
        "dataset_sha256",
        "base_lap",
        "tyre_model",
        "weather",
    }
    # M4: the calibrated tyre/fuel model, without a race adjustment (no race tables here)
    assert setup.lap_model.source_type is SourceType.ESTIMATED_FROM_DATA
    assert "season shape only" in setup.inputs["tyre_model"]
    assert setup.inputs["base_lap"].startswith("estimated_from_data")
    assert setup.inputs["dataset_sha256"].startswith("sha256:")
    # The synthetic race has no pit stops: the editorial pit model is used and labelled so.
    assert setup.inputs["pit_loss"].startswith("editorial_assumption")
    plain = setup_from_codes(monza_roster, monza_profiles, ["VER", "HAM"])
    assert plain.inputs == {} and all(d.car_pace_offset_s == 0.0 for d in plain.drivers)


def test_event_data_dir(tmp_path: Path) -> None:
    assert event_data_dir(tmp_path, "x") is None
    (tmp_path / "data" / "processed" / "x").mkdir(parents=True)
    (tmp_path / "data" / "processed" / "x" / "laps.parquet").write_bytes(b"")
    assert event_data_dir(tmp_path, "x") == tmp_path / "data" / "processed" / "x"


def test_pit_loss_from_in_and_out_laps() -> None:
    laps = synthetic_laps()
    # VER (median 84.0) pits at the end of lap 10: in-lap +5 s, out-lap +20 s -> 25 s loss.
    in_lap = (laps["driver_number"] == 1) & (laps["lap_number"] == 10)
    out_lap = (laps["driver_number"] == 1) & (laps["lap_number"] == 11)
    laps.loc[in_lap, ["lap_time_s", "pit_in_time_s"]] = [89.0, 1000.0]
    laps.loc[out_lap, ["lap_time_s", "pit_out_time_s"]] = [104.0, 1020.0]
    est = estimate_pit_loss(laps)
    reps = representative_laps(laps)
    median = float(reps[reps["driver_number"] == 1]["lap_time_s"].median())
    assert est.stops_used == 1 and est.total_loss_s == pytest.approx(89.0 + 104.0 - 2 * median)
    assert est.note.startswith("estimated_from_data")
    with pytest.raises(ValueError, match="no complete in-lap/out-lap pairs"):
        estimate_pit_loss(synthetic_laps())

    pit = pit_params_from_total_loss(est.total_loss_s, est.note)
    assert pit.lane_loss_s + pit.standstill_median_s == pytest.approx(est.total_loss_s)
    assert pit.source_type is SourceType.ESTIMATED_FROM_DATA


def test_pit_loss_ignores_stops_under_a_safety_car() -> None:
    """M4: in/out laps under SC/VSC/red flag are slow for other reasons (2024 Canada and Sao Paulo
    gave 49 s and 53 s whole-stop losses before this)."""
    laps = synthetic_laps()
    for number, lap, status in ((1, 10, "4"), (44, 12, "1")):
        in_lap = (laps["driver_number"] == number) & (laps["lap_number"] == lap)
        out_lap = (laps["driver_number"] == number) & (laps["lap_number"] == lap + 1)
        for column, value in (
            ("lap_time_s", 130.0),
            ("pit_in_time_s", 1.0),
            ("track_status", status),
        ):
            laps.loc[in_lap, column] = value
        for column, value in (("lap_time_s", 130.0), ("pit_out_time_s", 2.0)):
            laps.loc[out_lap, column] = value
    assert estimate_pit_loss(laps).stops_used == 1  # only the green one
    assert "under green" in estimate_pit_loss(laps).note
