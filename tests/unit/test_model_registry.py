"""Model registry and safe integration: every model tracked, and a race still builds and runs
without any of them (plan §16 M4: "the deterministic baseline keeps working"; D-036)."""

import shutil
from pathlib import Path

import pytest

from f1ar.calibration.driver_pace_fit import PACE_FILE
from f1ar.calibration.lap_model_fit import LIFE_FILE, SHAPE_FILE
from f1ar.calibration.race_control_fit import RACE_CONTROL_FILE
from f1ar.calibration.race_events_fit import EVENTS_FILE
from f1ar.calibration.racecraft_fit import RACECRAFT_FILE
from f1ar.calibration.registry import ENGINE_ARTIFACTS, load_registry, registry_markdown
from f1ar.calibration.traffic_fit import TRAFFIC_FILE
from f1ar.calibration.weather_fit import WEATHER_FILE
from f1ar.domain.driver_profiles import ProfileSet
from f1ar.domain.identities import RaceRoster
from f1ar.domain.provenance import SourceType
from f1ar.simulation import lightweight_engine as engine
from f1ar.simulation.scenarios import setup_from_codes
from tests.conftest import CONFIGS, REPO_ROOT, run_ticks
from tests.unit.test_race_inputs import synthetic_laps

CALIBRATION = (
    SHAPE_FILE,
    LIFE_FILE,
    WEATHER_FILE,
    TRAFFIC_FILE,
    RACE_CONTROL_FILE,
    EVENTS_FILE,
    PACE_FILE,
    RACECRAFT_FILE,
)


def test_every_model_is_registered_and_matches_its_artifact() -> None:
    entries = load_registry(REPO_ROOT)
    assert [e.problems for e in entries] == [[] for _ in entries]
    registered = {e.artifact for e in entries}
    assert set(ENGINE_ARTIFACTS) <= registered  # every file the engine loads has a manifest
    used = {e.model_name for e in entries if e.used_by_engine}
    assert used == {
        "tyre_shape",
        "tyre_life",
        "weather",
        "traffic",
        "race_control",
        "race_events",
        "driver_pace",
        "racecraft",
    }
    lgbm = next(e for e in entries if e.model_name == "pace_predictor_online")
    assert not lgbm.artifact_tracked and lgbm.used_by_engine is None
    assert "pace_predictor_online" in registry_markdown(entries)


def test_registry_flags_a_changed_artifact(tmp_path: Path) -> None:
    for d in ("configs", "data/manifests"):
        shutil.copytree(REPO_ROOT / d, tmp_path / d)
    (tmp_path / "configs" / TRAFFIC_FILE).write_text("{}\n", encoding="utf-8")
    bad = {e.model_name: e.problems for e in load_registry(tmp_path)}
    assert bad["traffic"] == ["artifact hash differs from the manifest"]
    (tmp_path / "configs" / SHAPE_FILE).unlink()
    assert (
        "tracked artifact missing"
        in {e.model_name: e.problems for e in load_registry(tmp_path)}["tyre_shape"]
    )


@pytest.mark.parametrize("missing", [*[(f,) for f in CALIBRATION], CALIBRATION])
def test_a_race_builds_and_runs_without_fitted_models(
    missing: tuple[Path, ...],
    monza_roster: RaceRoster,
    monza_profiles: ProfileSet,
    tmp_path: Path,
) -> None:
    configs = tmp_path / "configs"
    shutil.copytree(CONFIGS, configs)
    for f in missing:
        (configs / f).unlink()
    synthetic_laps().to_parquet(tmp_path / "laps.parquet")
    setup = setup_from_codes(
        monza_roster, monza_profiles, ["VER", "HAM"], data_dir=tmp_path, configs_dir=configs
    )
    parts = {
        SHAPE_FILE: ("tyre_model", setup.lap_model.source_type),
        TRAFFIC_FILE: ("traffic", setup.traffic.source_type),
        RACE_CONTROL_FILE: ("race_control", setup.race_control.source_type),
    }
    assert setup.inputs["race_events"].startswith("editorial") == (EVENTS_FILE in missing)
    assert setup.inputs["driver_pace"].startswith("editorial") == (PACE_FILE in missing)
    assert setup.inputs["racecraft"].startswith("editorial") == (RACECRAFT_FILE in missing)
    for f, (key, source) in parts.items():
        editorial = f in missing
        if f == TRAFFIC_FILE:  # W3: the racecraft refit supplies the overtaking too
            editorial = f in missing and RACECRAFT_FILE in missing
        assert setup.inputs[key].startswith("editorial") == editorial, key
        assert (source is SourceType.EDITORIAL_ASSUMPTION) == editorial, key
    # W1: the tyre-life model needs both files; without the life file the M4 tyre model runs
    if SHAPE_FILE not in missing:
        expected = "tyre life" if LIFE_FILE not in missing else "tyre/fuel shape"
        assert expected in setup.inputs["tyre_model"]
    # the synthetic race has no weather data: dry and editorial either way
    assert setup.inputs["weather"].startswith("editorial_assumption")
    state = run_ticks(engine.start(engine.create_world(setup, seed=1, world_id="w")), 2000)
    assert all(c.laps_completed >= 1 for c in state.cars)
