"""The RaceEngine port (plan §7.3): contract, bounded step, and action masks that agree with
``apply_action`` (what M5's masked RL will rely on)."""

import pytest

from f1ar.domain.race_state import (
    CancelPitStop,
    Compound,
    DriverAction,
    PaceMode,
    RaceSetup,
    RequestPitStop,
    SetPaceMode,
    SetStrategyController,
    StrategyController,
    WorldState,
    WorldStatus,
)
from f1ar.simulation import lightweight_engine as engine
from f1ar.simulation.engine_port import LightweightRaceEngine, RaceEngine
from tests.conftest import HAM, VER


def _all_actions(driver_id: str) -> list[DriverAction]:
    return [
        *(SetPaceMode(driver_id=driver_id, pace_mode=m) for m in PaceMode),
        *(RequestPitStop(driver_id=driver_id, compound=c) for c in Compound),
        CancelPitStop(driver_id=driver_id),
        *(SetStrategyController(driver_id=driver_id, controller=c) for c in StrategyController),
    ]


def _listed(mask: engine.ValidActions, action: DriverAction) -> bool:
    match action:
        case SetPaceMode():
            return action.pace_mode in mask.pace_modes
        case RequestPitStop():
            return action.compound in mask.pit_compounds
        case CancelPitStop():
            return mask.any  # always legal while racing (a no-op if nothing is pending)
        case SetStrategyController():
            return action.controller in mask.controllers
    raise AssertionError(action)


def test_the_adapter_satisfies_the_protocol(two_car_setup: RaceSetup) -> None:
    port: RaceEngine = LightweightRaceEngine()
    state = port.initialize(two_car_setup, seed=1, world_id="w")
    assert state.world_status is WorldStatus.RUNNING
    assert port.get_observation(HAM).driver_id == HAM
    assert [d for d, _, _ in port.get_positions()] == list(engine.standings(state))
    assert not port.is_terminal()


def test_step_is_bounded_to_whole_ticks(two_car_setup: RaceSetup) -> None:
    port = LightweightRaceEngine()
    port.initialize(two_car_setup, seed=1, world_id="w")
    port.step(1.24)  # 2 ticks of 0.5 s, never 3
    assert port.get_state().sim_tick == 2
    port.step(0.1)  # less than a tick still makes progress
    assert port.get_state().sim_tick == 3
    with pytest.raises(ValueError):
        port.step(0)
    while not port.is_terminal():
        port.step(600.0)
    assert port.get_state().world_status is WorldStatus.COMPLETED
    seqs = [e.seq for e in port.get_events()]
    assert seqs == list(range(len(seqs)))
    assert all(e.seq >= 10 for e in port.get_events(since_seq=10))


def test_set_state_round_trips_and_refuses_foreign_states(two_car_world: WorldState) -> None:
    port = LightweightRaceEngine()
    port.set_state(two_car_world)
    port.step(30.0)
    later = port.get_state()
    other = LightweightRaceEngine(two_car_world)
    other.step(30.0)
    assert other.get_state() == later  # same state + same steps -> same world
    with pytest.raises(engine.EngineStateError, match="engine"):
        port.set_state(two_car_world.model_copy(update={"engine_version": "old_engine"}))


def test_action_mask_agrees_with_apply_action(two_car_world: WorldState) -> None:
    """Over a whole race, at many states: every listed action is accepted and every unlisted
    action is rejected."""
    state = two_car_world
    checked = 0
    while state.world_status is WorldStatus.RUNNING:
        if state.sim_tick % 450 == 0:
            for driver_id in (VER, HAM):
                mask = engine.valid_actions(state, driver_id)
                for action in _all_actions(driver_id):
                    if _listed(mask, action):
                        engine.apply_action(state, action)  # must not raise
                    else:
                        with pytest.raises(engine.ActionRejectedError):
                            engine.apply_action(state, action)
                    checked += 1
        state = engine.step(state).state
    assert checked > 150
    final_mask = engine.valid_actions(state, HAM)
    assert not final_mask.any  # nothing is legal once the race is over


def test_no_pit_on_the_final_lap_and_cancel_only_when_pending(two_car_world: WorldState) -> None:
    state = two_car_world
    total = state.setup.track.total_laps
    while state.car(HAM).laps_completed < total - 1:
        state = engine.step(state).state
    mask = engine.valid_actions(state, HAM)
    assert mask.pit_compounds == () and mask.pace_modes
    early = engine.valid_actions(two_car_world, HAM)
    assert early.cancel_pit is (engine.pending_stop(two_car_world.car(HAM)) is not None)
