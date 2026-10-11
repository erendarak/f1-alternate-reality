from f1ar.domain.driver_profiles import PROFILE_FEATURE_NAMES
from f1ar.domain.race_state import WorldState
from f1ar.policy.observation import (
    OBSERVATION_FEATURE_NAMES,
    STATE_FEATURE_NAMES,
    build_observation,
)
from f1ar.simulation import lightweight_engine as engine
from tests.conftest import HAM, VER, run_ticks


def test_observation_schema_includes_profile_block() -> None:
    """M0 gate: DriverProfile fields are part of the observation schema."""
    assert OBSERVATION_FEATURE_NAMES == STATE_FEATURE_NAMES + PROFILE_FEATURE_NAMES
    assert len(set(OBSERVATION_FEATURE_NAMES)) == len(OBSERVATION_FEATURE_NAMES)


def test_policy_observation_contains_profile(two_car_world: WorldState) -> None:
    obs = build_observation(two_car_world, VER)
    profile = two_car_world.setup.driver(VER).profile
    assert {k: obs.features[k] for k in PROFILE_FEATURE_NAMES} == profile.features()
    assert len(obs.vector()) == len(OBSERVATION_FEATURE_NAMES)


def test_different_profiles_give_different_observations(two_car_world: WorldState) -> None:
    ver = build_observation(two_car_world, VER).features
    ham = build_observation(two_car_world, HAM).features
    assert ver["profile.risk_appetite"] != ham["profile.risk_appetite"]


def test_observation_hides_presampled_lap_time(two_car_world: WorldState) -> None:
    """No future leakage: the hidden duration of the current lap must not be observable."""
    state = run_ticks(two_car_world, 50)
    car = state.car(VER)
    tampered = state.model_copy(
        update={
            "cars": tuple(
                c.model_copy(update={"lap_target_s": c.lap_target_s + 5.0})
                if c.driver_id == VER
                else c
                for c in state.cars
            )
        }
    )
    assert tampered.car(VER).lap_target_s != car.lap_target_s
    assert build_observation(tampered, VER) == build_observation(state, VER)


def test_observation_tracks_race_progress(two_car_world: WorldState) -> None:
    start = build_observation(two_car_world, VER).features
    assert start["race.lap_number"] == 1.0
    assert start["car.has_last_lap"] == 0.0
    assert start["car.compound.medium"] == 1.0 and start["car.pace_mode.balanced"] == 1.0

    state = two_car_world
    while state.car(VER).laps_completed == 0:
        state = engine.step(state).state
    later = build_observation(state, VER).features
    assert later["race.lap_number"] == 2.0
    assert later["car.tyre_age_laps"] == 1.0
    assert later["car.has_last_lap"] == 1.0 and later["car.last_lap_time_s"] > 0
    assert later["car.position"] == engine.standings(state).index(VER) + 1


def test_observation_hides_true_tyre_wear(two_car_world: WorldState) -> None:
    """Partial observability: wear is inferred from lap times, not read off the car."""
    state = run_ticks(two_car_world, 400)
    tampered = state.model_copy(
        update={"cars": tuple(c.model_copy(update={"tyre_wear_s": 9.9}) for c in state.cars)}
    )
    assert build_observation(tampered, VER) == build_observation(state, VER)
