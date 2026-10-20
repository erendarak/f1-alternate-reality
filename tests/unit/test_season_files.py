"""H2a (D-054): calibration files per season; a race of another season never silently takes
2024's driver/team values, and borrowed tyre models / era rates are labelled."""

import json
from pathlib import Path

import pytest

from f1ar.calibration.driver_pace_fit import PACE_FILE, load_pace
from f1ar.calibration.lap_model_fit import LIFE_FILE, load_life
from f1ar.calibration.season_files import (
    borrowed_note,
    era_rates_note,
    own_season,
    season_path,
    with_fallback,
)
from f1ar.domain.driver_profiles import ProfileSet
from f1ar.domain.identities import RaceRoster
from f1ar.domain.season_rules import rules_for
from f1ar.simulation.scenarios import _label_other_seasons, build_setup
from tests.conftest import CONFIGS


def _write(configs: Path, rel: Path, data: dict[str, object]) -> None:
    (configs / rel).parent.mkdir(parents=True, exist_ok=True)
    (configs / rel).write_text(json.dumps(data), encoding="utf-8")


def test_season_path_and_lookup(tmp_path: Path) -> None:
    assert season_path(PACE_FILE, 2019).name == "driver_pace_2019_v1.json"
    with pytest.raises(ValueError, match="not a per-season"):
        season_path(Path("calibration/foo.json"), 2019)
    _write(tmp_path, PACE_FILE, {"season": 2024})
    assert own_season(tmp_path, PACE_FILE, 2019) is None
    assert with_fallback(tmp_path, PACE_FILE, 2019) == (tmp_path / PACE_FILE, 2024)
    _write(tmp_path, season_path(PACE_FILE, 2019), {"season": 2019})
    assert with_fallback(tmp_path, PACE_FILE, 2019)[1] == 2019
    assert with_fallback(tmp_path / "none", PACE_FILE, 2019) == (None, None)


def test_driver_values_are_never_borrowed_tyre_life_is(tmp_path: Path) -> None:
    _write(tmp_path, PACE_FILE, {"season": 2024})
    _write(tmp_path, LIFE_FILE, {"season": 2024})
    assert load_pace(tmp_path, 2019) is None
    assert load_pace(tmp_path) == {"season": 2024}
    life = load_life(tmp_path, 2019)
    assert life is not None and life[0] == {"season": 2024}
    assert borrowed_note(2019, 2024) == " [borrowed from 2024: no 2019 fit yet]"
    assert borrowed_note(2019, 2019) == "" and borrowed_note(2024, 2024) == ""
    assert era_rates_note(2024) == ""
    assert "2024 rates applied to 2019" in era_rates_note(2019)


def test_other_season_inputs_are_labelled(
    monza_roster: RaceRoster, monza_profiles: ProfileSet, tmp_path: Path
) -> None:
    import shutil

    only_2024 = tmp_path / "configs"  # a season with no fits of its own (H2a/H2b fit 2010+)
    (only_2024 / "calibration").mkdir(parents=True)
    for f in (CONFIGS / "calibration").glob("*_2024_v1.json"):
        shutil.copy(f, only_2024 / "calibration" / f.name)
    keys = ("tyre_model", "traffic", "race_control", "weather", "wet_incidents", "base_lap")
    inputs = {k: "fitted" for k in keys}
    for season, labelled in ((2024, False), (2015, True)):
        setup = build_setup(
            monza_roster,
            monza_profiles,
            monza_roster.driver_ids(),
            rules=rules_for(CONFIGS, season),
            inputs=inputs,
        )
        out = _label_other_seasons(setup, only_2024).inputs
        assert out["base_lap"] == "fitted"  # measured on the race itself
        for key in ("traffic", "race_control", "weather", "wet_incidents"):
            assert (out[key] != "fitted") is labelled
        assert ("borrowed from 2024" in out["tyre_model"]) is labelled


def test_track_outline_from_the_nearest_season_on_the_same_circuit(tmp_path: Path) -> None:
    import pandas as pd

    from f1ar.simulation.scenarios import _same_circuit_outline

    rosters = tmp_path / "configs" / "rosters"
    rosters.mkdir(parents=True)
    for event, circuit, season in (
        ("2018_x_gp", "albert_park", 2018),
        ("2019_x_gp", "albert_park", 2019),
        ("2023_x_gp", "albert_park", 2023),
        ("2019_y_gp", "bahrain", 2019),
    ):
        (rosters / f"{event}.json").write_text(
            json.dumps({"circuit_id": circuit, "season": season}), encoding="utf-8"
        )
        folder = tmp_path / "data" / "processed" / event
        folder.mkdir(parents=True)
        if event != "2018_x_gp":
            pd.DataFrame({"x_m": [0.0], "y_m": [0.0]}).to_parquet(folder / "track_outline.parquet")
    own = tmp_path / "data" / "processed" / "2018_x_gp"
    assert _same_circuit_outline("2018_x_gp", own) == tmp_path / "data" / "processed" / "2019_x_gp"
    assert _same_circuit_outline("2017_none_gp", own) is None
