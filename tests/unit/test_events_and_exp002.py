"""Simulating more than one event (Spa 2024) and the EXP-002 overtake counter."""

import importlib.util
import sys
from types import ModuleType

import pytest

from f1ar.domain.race_state import WorldStatus
from f1ar.simulation import lightweight_engine as engine
from f1ar.simulation.scenarios import (
    BELGIUM_2024,
    MONZA_2024,
    SIMULATED_EVENTS,
    load_event,
    setup_from_codes,
    track_for,
)
from tests.conftest import CONFIGS, REPO_ROOT


def test_spa_personas_match_monza_for_the_same_drivers() -> None:
    _, monza = load_event(CONFIGS, MONZA_2024)
    roster, spa = load_event(CONFIGS, BELGIUM_2024)
    shared = {p.driver_id for p in monza.profiles} & {p.driver_id for p in spa.profiles}
    assert len(shared) == 19
    for d in shared:
        assert spa.get(d).archetype == monza.get(d).archetype
    assert "logan_sargeant_2024" in {p.driver_id for p in spa.profiles}
    assert "franco_colapinto_2024" not in {p.driver_id for p in spa.profiles}
    assert roster.event_id == BELGIUM_2024


def test_every_simulated_event_runs_a_full_race_without_data() -> None:
    for event_id in SIMULATED_EVENTS:
        roster, profiles = load_event(CONFIGS, event_id)
        setup = setup_from_codes(roster, profiles, None)  # synthetic: no fetched data needed
        assert setup.track == track_for(event_id)
        state = engine.start(engine.create_world(setup, seed=5, world_id="w"))
        while state.world_status is WorldStatus.RUNNING:
            state = engine.step(state).state
        winner = state.car(engine.standings(state)[0])
        assert winner.laps_completed == setup.track.total_laps
        # Spa laps (~104 s placeholder) are much longer than Monza's (~82 s)
        per_lap = (winner.finish_time_s or 0.0) / setup.track.total_laps
        assert per_lap == pytest.approx(setup.lap_model.base_lap_s + 2.5, abs=3.0)
    with pytest.raises(ValueError, match="no track spec"):
        track_for("2030_mars_gp")


def _exp002() -> ModuleType:
    path = REPO_ROOT / "scripts" / "experiments" / "exp002_m2_engine.py"
    spec = importlib.util.spec_from_file_location("exp002", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["exp002"] = module
    spec.loader.exec_module(module)
    return module


def test_overtake_counter_counts_passes_and_ignores_pit_stops() -> None:
    count = _exp002().count_overtakes
    grid = ["A", "B", "C"]
    # Lap 1: B passes A. Lap 2: C passes both, but C pitted at the end of lap 2 -> excluded
    # from lap 2; lap 3: A passes B.
    crossings = {
        1: {"B": 100.0, "A": 100.5, "C": 101.0},
        2: {"C": 199.0, "B": 200.0, "A": 200.4},
        3: {"C": 300.0, "A": 300.1, "B": 300.2},
    }
    assert count(crossings, grid, {}) == 1 + 2 + 1
    assert count(crossings, grid, {"C": {2}}) == 1 + 1  # C's laps 2 and 3 don't count
