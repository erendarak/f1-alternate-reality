"""Weather (M4 slice 4, D-034): wetness dynamics, tyre penalties, rain sources, the forecast
(no future leakage, US-06), and the engine's wet races (switching, rules, forks)."""

import math

import pandas as pd
import pytest

from f1ar.calibration.weather_fit import revealed_wetness_timeline
from f1ar.domain.race_state import (
    Compound,
    PaceMode,
    RaceSetup,
    SetPaceMode,
    SetStrategyController,
    StrategyController,
    WeatherMode,
    WeatherParams,
    WorldState,
    WorldStatus,
    default_weather_params,
)
from f1ar.policy.baseline_strategy import dry_compounds_required
from f1ar.policy.observation import build_observation
from f1ar.simulation import lightweight_engine as engine
from f1ar.simulation.timeline import fork_world
from f1ar.simulation.weather import (
    best_tyre,
    crossovers,
    forecast_rain_probability,
    historical_rain,
    historical_wetness,
    next_wetness,
    penalty_s,
    raining_at,
)
from tests.conftest import HAM, VER

W = default_weather_params()


def test_wetness_wets_and_dries_exponentially() -> None:
    w = next_wetness(W, 0.0, True, 600.0)  # one time constant of rain
    assert w == pytest.approx(1 - math.exp(-1))
    assert next_wetness(W, 1.0, False, 1800.0) == pytest.approx(math.exp(-1))
    assert next_wetness(W, 0.3, True, 0.0) == pytest.approx(0.3)


def test_each_tyre_is_best_in_its_band() -> None:
    lo, hi = crossovers(W)
    assert 0 < lo < hi < 1
    assert best_tyre(W, lo - 0.05) is Compound.MEDIUM
    assert best_tyre(W, (lo + hi) / 2) is Compound.INTERMEDIATE
    assert best_tyre(W, min(1.0, hi + 0.05)) is Compound.WET
    assert penalty_s(W, Compound.SOFT, 0.0) == 0.0  # slicks lose nothing on a dry track


def test_historical_rain_and_wetness_timelines() -> None:
    p = W.model_copy(
        update={
            "rain_timeline": ((-60.0, False), (100.0, True), (400.0, False)),
            "wetness_timeline": ((0.0, 0.0), (60.0, 0.6), (120.0, 0.2)),
        }
    )
    assert not historical_rain(p, 50.0) and historical_rain(p, 100.0)
    assert historical_rain(p, 399.0) and not historical_rain(p, 400.0)
    assert historical_wetness(p, 30.0) == pytest.approx(0.3)
    assert historical_wetness(p, 90.0) == pytest.approx(0.4)
    assert historical_wetness(p, 999.0) == 0.2


def test_random_rain_is_keyed_and_shared_by_seed() -> None:
    p = W.model_copy(
        update={
            "mode": WeatherMode.RANDOM,
            "random_rain_start_per_min": 0.3,
            "random_rain_stop_per_min": 0.3,
        }
    )

    def run(seed: int) -> list[bool]:
        raining, out = False, []
        for tick in range(1, 2400):
            raining = raining_at(p, seed, tick * 0.5, raining, 0.5)
            out.append(raining)
        return out

    a, b = run(7), run(7)
    assert a == b and any(a) and not all(a)  # it rains sometimes, identically for one seed
    assert run(8) != a


def test_forecast_is_bounded_noisy_and_blind_beyond_its_horizon() -> None:
    p = W.model_copy(update={"rain_timeline": ((0.0, False), (300.0, True), (1500.0, False))})
    values = [forecast_rain_probability(p, 1, t, False) for t in range(0, 2400, 60)]
    assert all(0.0 <= v <= 1.0 for v in values)
    exact = p.model_copy(update={"forecast_noise": 0.0})
    # rain from 5 min: half of the next 10 minutes is wet
    assert forecast_rain_probability(exact, 1, 0.0, False) == pytest.approx(0.5)
    assert forecast_rain_probability(exact, 1, 600.0, True) == pytest.approx(1.0)
    # rain far beyond the horizon doesn't reach the forecast (US-06: it can't see the future)
    later = p.model_copy(
        update={"rain_timeline": ((0.0, False), (300.0, True), (1500.0, False), (5000.0, True))}
    )
    assert forecast_rain_probability(later, 1, 1200.0, True) == forecast_rain_probability(
        p, 1, 1200.0, True
    )


# --- engine ------------------------------------------------------------------------------------


def _wet_race(setup: RaceSetup) -> RaceSetup:
    """Dry start, rain from 8 to 30 minutes, dynamics-driven wetness (no inferred timeline);
    the track dries within the race (faster than the editorial default)."""
    weather = W.model_copy(
        update={
            "rain_timeline": ((0.0, False), (480.0, True), (1800.0, False)),
            "dry_rate_per_s": 1 / 600,
        }
    )
    return setup.model_copy(update={"weather": weather})


def _run(world: WorldState) -> tuple[WorldState, list[str], dict[str, list[str]]]:
    events: list[str] = []
    tyres: dict[str, list[str]] = {
        d.driver_id: [d.start_compound.value] for d in world.setup.drivers
    }
    state = engine.start(world)
    while state.world_status is WorldStatus.RUNNING:
        r = engine.step(state)
        state = r.state
        for e in r.events:
            events.append(e.type)
            if e.type == "PIT_STOP" and e.driver_id:
                tyres[e.driver_id].append(str(e.payload["to_compound"]))
    return state, events, tyres


def test_a_shower_makes_the_baseline_go_to_inters_and_back(two_car_setup: RaceSetup) -> None:
    setup = _wet_race(two_car_setup)
    final, events, tyres = _run(engine.create_world(setup, seed=3, world_id="w"))
    assert events.count("RAIN_STARTED") == 1 and events.count("RAIN_STOPPED") == 1
    assert "TYRE_RULE_BREACHED" not in events  # wet tyres lift the two-dry-compound rule
    for d in (VER, HAM):
        assert "INTERMEDIATE" in tyres[d], tyres[d]
        assert tyres[d][-1] in ("SOFT", "MEDIUM", "HARD")  # back on slicks once it dried
        assert dry_compounds_required(setup, final.car(d).compounds_used) == 0


def test_wet_laps_are_slower_and_weather_is_shared_by_forks(two_car_setup: RaceSetup) -> None:
    setup = _wet_race(two_car_setup)
    world = engine.start(engine.create_world(setup, seed=3, world_id="w"))
    a = world
    for _ in range(2400):  # 20 minutes: well into the shower
        a = engine.step(a).state
    assert a.weather.raining and a.weather.wetness > 0.5
    # a fork where HAM pushes: different decisions, identical weather
    b = engine.resume(fork_world(world, new_timeline_id="t_b"))
    b = engine.apply_action(b, SetPaceMode(driver_id=HAM, pace_mode=PaceMode.PUSH)).state
    for _ in range(2400):
        b = engine.step(b).state
    assert b.weather == a.weather


def test_user_controlled_car_is_not_switched_automatically(two_car_setup: RaceSetup) -> None:
    setup = _wet_race(two_car_setup)
    world = engine.create_world(setup, seed=3, world_id="w")
    world = engine.apply_action(
        world,
        SetStrategyController(driver_id=HAM, controller=StrategyController.USER),
    ).state
    _, _, tyres = _run(world)
    assert tyres[HAM] == [setup.driver(HAM).start_compound.value]  # never boxed on its own
    assert "INTERMEDIATE" in tyres[VER]


def test_observation_has_weather_and_never_the_future(two_car_world: WorldState) -> None:
    obs = build_observation(two_car_world, VER).features
    assert obs["weather.raining"] == 0.0 and obs["weather.wetness"] == 0.0
    assert 0.0 <= obs["weather.forecast_rain"] <= 1.0
    # change the rain far in the future: what the strategist sees now doesn't move
    far = two_car_world.setup.weather.model_copy(update={"rain_timeline": ((5000.0, True),)})
    changed = two_car_world.model_copy(
        update={"setup": two_car_world.setup.model_copy(update={"weather": far})}
    )
    assert build_observation(changed, VER).features == obs


# --- the inferred conditions --------------------------------------------------------------


def _laps(compounds: list[list[str]]) -> pd.DataFrame:
    rows = []
    for car, seq in enumerate(compounds):
        for lap, c in enumerate(seq):
            rows.append(
                {
                    "lap_start_time_s": 1000.0 + 90.0 * lap + car,
                    "lap_end_time_s": 1090.0 + 90.0 * lap + car,
                    "compound": c,
                }
            )
    return pd.DataFrame(rows)


def test_inferred_wetness_follows_the_field_and_keeps_inters_in_their_band() -> None:
    slick, inter = "MEDIUM", "INTERMEDIATE"
    field = [[slick, slick, inter, inter, inter, slick]] * 10
    timeline = revealed_wetness_timeline(_laps(field), 1000.0, W)
    by_time = dict(timeline)
    lo, hi = crossovers(W)
    assert by_time[0.0] < lo  # all on slicks: dry side of the slick/inter crossover
    assert lo < by_time[240.0] <= hi - 0.05 + 1e-9  # all on inters: inter band, not full wets
    dry = revealed_wetness_timeline(_laps([[slick] * 5] * 10), 1000.0, W)
    assert dry == ()  # a dry race has no inferred timeline


def test_weather_params_round_trip() -> None:
    assert WeatherParams.model_validate_json(W.model_dump_json()) == W
