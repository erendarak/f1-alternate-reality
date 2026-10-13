"""Simulating more than one event (Spa 2024) and the EXP-002 overtake counter."""

import pytest

from f1ar.calibration.reference_metrics import count_overtakes, pass_causes
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
from tests.conftest import CONFIGS


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


def test_overtake_counter_counts_passes_and_ignores_pit_stops() -> None:
    count = count_overtakes
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


def test_pass_causes_split_the_same_passes() -> None:
    grid = ["A", "B", "C", "D"]
    crossings = {
        1: {"B": 100.0, "A": 100.5, "C": 101.0, "D": 101.5},  # lap 1: B passes A
        2: {"B": 200.0, "C": 200.3, "A": 200.6, "D": 201.0},  # C passes A (fresher tyres)
        3: {"B": 300.0, "A": 300.2, "C": 300.4, "D": 300.6},  # A passes C (same age)
        4: {"B": 400.0, "A": 400.2, "D": 400.3, "C": 400.4},  # D passes C under SC
    }
    age = {("C", 2): 1, ("A", 2): 20, ("A", 3): 21, ("C", 3): 2}
    causes = pass_causes(crossings, grid, {}, {4}, age)
    assert causes == {"lap1": 1, "tyre_offset": 1, "green": 1, "neutral": 1}
    assert sum(causes.values()) == count_overtakes(crossings, grid, {})
