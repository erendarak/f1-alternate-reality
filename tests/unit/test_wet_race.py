"""W6 (engine v0.14.0, D-046): fresh wet tyres under SC/VSC and the wet incident fit."""

import pandas as pd
import pytest

from f1ar.calibration.wet_race_fit import fit_refresh, profile_fit, refresh_observations
from f1ar.domain.race_state import Compound, RacePhase, WeatherState, WorldState
from f1ar.simulation import lightweight_engine as engine
from tests.conftest import VER


def test_fit_refresh_finds_the_age_threshold() -> None:
    obs = [{"age": a, "refresh": a >= 12 and i % 4 != 1} for i, a in enumerate(range(0, 30))]
    fit = fit_refresh(obs)
    assert fit["min_age_laps"] == 12
    assert fit["share"] == pytest.approx(13 / 18, abs=1e-3)
    assert fit["young_share"] == 0.0


def test_refresh_observations_cut_at_the_red_flag() -> None:
    rows = []
    for driver, age0, pit_lap in (("AAA", 20, 5), ("BBB", 2, None), ("CCC", 20, 7)):
        for lap in range(1, 9):
            rows.append(
                {
                    "driver_code": driver,
                    "lap_number": lap,
                    "compound": "INTERMEDIATE",
                    "tyre_age_start": age0 + lap - 1,
                    "pit_in_lap": lap == pit_lap,
                    "track_status_start": "4" if lap in (5, 6, 7) else "1",
                    "lap_start_time_s": 100.0 * lap,
                }
            )
    laps = pd.DataFrame(rows)
    # SC on laps 5-7 (+ lap 4 before it); a red flag during lap 7 ends the episode at lap 6
    obs = {o["driver"]: o for o in refresh_observations(laps, [750.0])}
    assert obs["AAA"] == {"first_lap": 4, "driver": "AAA", "age": 23.0, "refresh": True}
    assert not obs["BBB"]["refresh"]
    assert not obs["CCC"]["refresh"]  # its stop was the red-flag tyre change


def test_profile_fit_peaks_where_sim_matches_real() -> None:
    real = {"e": {"neutralisations": 3.0, "retirements": 3.0}}
    sim = {
        f: {"e": {"neutralisations": 1.0 + f / 2, "retirements": 1.0 + f / 2}}
        for f in (0.0, 2.0, 4.0, 6.0, 8.0)
    }
    fit = profile_fit(sim, real)
    assert fit["best"] == pytest.approx(4.0, abs=0.5)
    assert fit["range_95"][0] < 4.0 < fit["range_95"][1]


def _wet_sc_world(world: WorldState, share: float) -> WorldState:
    state = world
    while state.car(VER).laps_completed < 3:
        state = engine.step(state).state
    weather = state.setup.weather.model_copy(
        update={"wet_refresh_min_age_laps": 10, "wet_refresh_share": share}
    )
    cars = tuple(
        c.model_copy(update={"compound": Compound.INTERMEDIATE, "tyre_age_laps": 15})
        for c in state.cars
    )
    state = state.model_copy(
        update={
            "setup": state.setup.model_copy(update={"weather": weather}),
            "cars": cars,
            "weather": WeatherState(raining=False, wetness=0.5),
        }
    )
    return engine.deploy_neutralisation(state, RacePhase.SC, sc_laps=2).state


@pytest.mark.parametrize("share", [0.0, 1.0])
def test_old_wet_tyres_are_refreshed_under_the_safety_car(
    isolated_world: WorldState, share: float
) -> None:
    state = _wet_sc_world(isolated_world, share)
    stops = []
    while state.race_control.phase is RacePhase.SC:
        result = engine.step(state)
        state = result.state
        stops += [e for e in result.events if e.type == "PIT_STOP"]
    fresh = [e for e in stops if e.payload["to_compound"] == Compound.INTERMEDIATE.value]
    assert len(fresh) == (len(state.cars) if share == 1.0 else 0)
