import pytest

from f1ar.domain.race_state import PaceMode, SetPaceMode, WorldState, WorldStatus
from f1ar.policy.observation import build_observation
from f1ar.simulation import lightweight_engine as engine
from tests.conftest import HAM, VER, run_ticks, run_to_end


def _push(driver_id: str = HAM) -> SetPaceMode:
    return SetPaceMode(driver_id=driver_id, pace_mode=PaceMode.PUSH)


def test_set_pace_mode_changes_car_and_emits_event(two_car_world: WorldState) -> None:
    state = run_ticks(two_car_world, 10)
    result = engine.apply_action(state, _push())
    assert result.state.car(HAM).pace_mode is PaceMode.PUSH
    assert result.state.car(VER).pace_mode is PaceMode.BALANCED
    (event,) = result.events
    assert event.type == "PACE_MODE_CHANGED" and event.driver_id == HAM
    assert event.seq == state.event_seq and event.tick == state.sim_tick
    assert event.payload == {
        "from": "BALANCED",
        "to": "PUSH",
        "effective_from_lap": 2,
        "actor_id": "USER",
    }
    assert result.state.event_seq == state.event_seq + 1
    assert result.state.sim_tick == state.sim_tick  # actions don't advance time


def test_action_does_not_touch_the_lap_in_progress(two_car_world: WorldState) -> None:
    state = run_ticks(two_car_world, 10)
    after = engine.apply_action(state, _push()).state
    assert after.car(HAM).lap_target_s == state.car(HAM).lap_target_s


def test_push_changes_the_next_lap(two_car_world: WorldState) -> None:
    """Same seed, same noise: the only difference on lap 2 is the pace mode."""
    base = two_car_world
    pushed = engine.apply_action(two_car_world, _push()).state
    while base.car(HAM).laps_completed == 0:
        base = engine.step(base).state
        pushed = engine.step(pushed).state
    assert pushed.car(HAM).lap_target_s < base.car(HAM).lap_target_s
    assert pushed.car(VER).lap_target_s == base.car(VER).lap_target_s


def test_same_mode_is_an_accepted_no_op(two_car_world: WorldState) -> None:
    result = engine.apply_action(
        two_car_world, SetPaceMode(driver_id=HAM, pace_mode=PaceMode.BALANCED)
    )
    assert result.state == two_car_world and result.events == ()


def test_actions_allowed_while_paused(two_car_world: WorldState) -> None:
    paused = engine.pause(two_car_world)
    after = engine.apply_action(paused, _push()).state
    assert after.world_status is WorldStatus.PAUSED and after.car(HAM).pace_mode is PaceMode.PUSH


def test_illegal_actions_are_rejected(two_car_world: WorldState) -> None:
    with pytest.raises(engine.ActionRejectedError, match="unknown driver"):
        engine.apply_action(two_car_world, _push("nobody_2024"))
    finished = run_to_end(two_car_world)
    with pytest.raises(engine.ActionRejectedError, match="race is over"):
        engine.apply_action(finished, _push())


def test_pace_mode_is_visible_in_the_observation(two_car_world: WorldState) -> None:
    after = engine.apply_action(two_car_world, _push()).state
    features = build_observation(after, HAM).features
    assert features["car.pace_mode.push"] == 1.0 and features["car.pace_mode.balanced"] == 0.0
