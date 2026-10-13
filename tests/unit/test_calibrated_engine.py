"""M4 slice 3 (D-033): season personas, track specs from data, the calibrated lap model."""

from pathlib import Path

import pandas as pd
import pytest

from f1ar.calibration.lap_model_fit import (
    WEAR_MULTIPLIER_FLOOR,
    calibrated_lap_model,
    load_shape,
)
from f1ar.calibration.pace_models import RaceAdjustment, TyreShape
from f1ar.domain.driver_profiles import load_season_profiles
from f1ar.domain.provenance import SourceType
from f1ar.domain.race_state import Compound
from f1ar.simulation import lightweight_engine as engine
from f1ar.simulation.lap_model import tyre_wear_factor
from f1ar.simulation.scenarios import (
    BELGIUM_2024,
    MONZA_2024,
    default_lap_model,
    has_personas,
    load_event,
    setup_from_codes,
    track_for,
)
from tests.conftest import CONFIGS, REPO_ROOT

SEASON_SEED = CONFIGS / "drivers" / "seed_2024_season_drivers.yaml"
TEMPLATES = CONFIGS / "drivers" / "archetypes.yaml"


@pytest.mark.parametrize("event", [MONZA_2024, BELGIUM_2024])
def test_season_seed_gives_the_same_personas_as_the_race_seeds(event: str) -> None:
    roster, race_profiles = load_event(CONFIGS, event)
    season = load_season_profiles(SEASON_SEED, TEMPLATES, roster)
    for p in race_profiles.profiles:
        assert season.get(p.driver_id).archetype == p.archetype
        assert season.get(p.driver_id).metadata.confidence == p.metadata.confidence


def test_every_2024_race_has_personas() -> None:
    rosters = sorted((CONFIGS / "rosters").glob("2024_*.json"))
    assert len(rosters) == 24
    for path in rosters:
        assert has_personas(CONFIGS, path.stem)
        roster, profiles = load_event(CONFIGS, path.stem)
        assert {p.driver_id for p in profiles.profiles} == set(roster.driver_ids())
        for p in profiles.profiles:
            assert p.metadata.numeric_values_source_type is SourceType.EDITORIAL_ASSUMPTION


def test_track_spec_needs_the_outline(tmp_path: Path) -> None:
    assert track_for(MONZA_2024).length_m == 5793.0  # published figure kept
    with pytest.raises(ValueError, match="track outline"):
        track_for("2024_bahrain_gp", tmp_path)
    pd.DataFrame({"x_m": [0.0, 300.0, 300.0, 0.0], "y_m": [0.0, 0.0, 200.0, 200.0]}).to_parquet(
        tmp_path / "track_outline.parquet"
    )
    pd.DataFrame({"laps": [57, 56, 12]}).to_parquet(tmp_path / "results.parquet")
    spec = track_for("2024_bahrain_gp", tmp_path)
    assert spec.length_m == 1000.0 and spec.total_laps == 57 and spec.track_id == "bahrain_gp"
    assert spec.source_note.startswith("estimated_from_data")


SHAPE = TyreShape(
    offset={"SOFT": -0.2, "MEDIUM": 0.0, "HARD": 0.1},
    wear_lin={"SOFT": 0.1, "MEDIUM": 0.08, "HARD": 0.06},
    wear_quad={"SOFT": -0.001, "MEDIUM": -0.001, "HARD": 0.0},
    fuel_s_per_lap=0.05,
    dirty_air_s=0.4,
    fitted_on=("a", "b"),
)


def test_calibrated_model_floor_normalisation_and_deltas(monkeypatch: pytest.MonkeyPatch) -> None:
    import f1ar.calibration.lap_model_fit as fit

    editorial = default_lap_model(80.0)
    _, profiles = load_event(CONFIGS, MONZA_2024)
    drivers = [p.archetype for p in profiles.profiles]
    field = sum(tyre_wear_factor(editorial, d) for d in drivers) / len(drivers)

    monkeypatch.setattr(
        fit, "fit_race_adjustment", lambda race, shape: RaceAdjustment(-0.3, {"SOFT": 0.2})
    )
    cal = calibrated_lap_model(editorial, SHAPE, "sha256:x", pd.DataFrame(), drivers)
    assert cal.wear_multiplier_used == WEAR_MULTIPLIER_FLOOR  # a negative fit is floored
    soft = cal.params.compounds[Compound.SOFT]
    assert soft.offset_s == pytest.approx(-0.2 + 0.2)
    assert soft.deg_s_per_lap == pytest.approx(0.1 * WEAR_MULTIPLIER_FLOOR / field)
    assert soft.deg_quad_s_per_lap2 == pytest.approx(-0.001 * WEAR_MULTIPLIER_FLOOR / field)
    assert cal.params.fuel_s_per_lap_remaining == 0.05
    assert cal.params.source_type is SourceType.ESTIMATED_FROM_DATA
    # editorial parts are kept as they were
    assert cal.params.skill_gap_s == editorial.skill_gap_s
    assert cal.params.pace_modes == editorial.pace_modes
    # the field's average driver wears at the fitted rate
    mean_rate = sum(
        cal.params.compounds[Compound.MEDIUM].deg_s_per_lap * tyre_wear_factor(cal.params, d)
        for d in drivers
    ) / len(drivers)
    assert mean_rate == pytest.approx(0.08 * WEAR_MULTIPLIER_FLOOR)


def test_season_shape_file_is_the_fitted_one() -> None:
    shape, sha = load_shape(CONFIGS)
    assert len(shape.fitted_on) == 16 and sha.startswith("sha256:")
    assert 0.03 < shape.fuel_s_per_lap < 0.07


MONZA_DATA = REPO_ROOT / "data" / "processed" / MONZA_2024


@pytest.mark.skipif(not (MONZA_DATA / "laps.parquet").exists(), reason="Monza data not fetched")
def test_calibrated_monza_plans_medium_to_hard() -> None:
    """The EXP-002 symptom (every car one-stopping onto SOFT) is gone at Monza: the calibrated
    model makes the real race's main strategy, MEDIUM -> HARD, the most common plan."""
    roster, profiles = load_event(CONFIGS, MONZA_2024)
    setup = setup_from_codes(roster, profiles, None, data_dir=MONZA_DATA)
    world = engine.create_world(setup, seed=1, world_id="w")
    plans = [
        (c.compound, tuple(s.compound for s in c.planned_stops))
        for c in world.cars
        if c.compound is Compound.MEDIUM
    ]
    assert plans and all(stops == (Compound.HARD,) for _, stops in plans)
    assert setup.lap_model.source_type is SourceType.ESTIMATED_FROM_DATA
    assert "in-sample" in setup.inputs["tyre_model"]
