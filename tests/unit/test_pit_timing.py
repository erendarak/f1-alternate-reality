"""A2 pre-gate (EXP-032, D-063): pit-timing forks, forced stops, what the pit wall sees."""

import math

from f1ar.domain.race_state import RaceSetup
from f1ar.simulation import lightweight_engine as engine
from f1ar.training.pit_timing import LEAD, branch, fork_features, forks, points_score
from tests.unit.test_tyre_life import _with_life


def test_forks_forced_stops_and_features(two_car_setup: RaceSetup) -> None:
    setup = _with_life(two_car_setup)
    world = engine.start(engine.create_world(setup, seed=5, world_id="w"))
    found = list(forks(world))
    assert found
    state, driver, lap, compound = found[0]
    car = state.car(driver)
    assert car.laps_completed == lap - LEAD and car.pit_stops == 0
    feats = fork_features(state, driver, compound)
    assert all(math.isfinite(v) for v in feats.values())
    assert feats["laps_left"] == setup.track.total_laps - car.laps_completed
    for d in (-1, 2):
        end = branch(state, driver, lap + d, compound)
        events = end.car(driver)
        assert events.pit_stops >= 1
    # same luck: forcing the planned lap reproduces itself
    a = points_score(branch(state, driver, lap, compound), driver)
    b = points_score(branch(state, driver, lap, compound), driver)
    assert a == b
