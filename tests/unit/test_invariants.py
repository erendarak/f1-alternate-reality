"""World invariants on every tick (plan §18.2): full races with everything switched on."""

import pytest

from f1ar.domain.driver_profiles import ProfileSet
from f1ar.domain.identities import RaceRoster
from f1ar.domain.race_state import CarStatus, RacePhase, WorldState, WorldStatus
from f1ar.simulation import lightweight_engine as engine
from f1ar.simulation.invariants import state_violations, step_violations
from f1ar.simulation.scenarios import build_setup
from f1ar.simulation.world_runner import Command, CommandType, WorldRunner
from tests.conftest import HAM


@pytest.mark.parametrize("seed", [3, 21])
def test_every_tick_of_a_full_race_is_valid(
    monza_roster: RaceRoster, monza_profiles: ProfileSet, seed: int
) -> None:
    """20 cars with pit stops, traffic, incidents, plus a race-director VSC and safety car."""
    setup = build_setup(monza_roster, monza_profiles, [e.driver_id for e in monza_roster.entries])
    state = engine.start(engine.create_world(setup, seed=seed, world_id="w"))
    problems: list[str] = []
    while state.world_status is WorldStatus.RUNNING:
        if state.sim_tick == 168 * 12 and state.race_control.phase is RacePhase.GREEN:
            state = engine.deploy_neutralisation(state, RacePhase.VSC, vsc_s=60).state
        if state.sim_tick == 168 * 30 and state.race_control.phase is not RacePhase.SC:
            state = engine.deploy_neutralisation(state, RacePhase.SC, sc_laps=3).state
        result = engine.step(state)
        problems += step_violations(state, result.state, result.events)
        state = result.state
        if len(problems) > 20:
            break
    assert problems == []
    assert state.race_control.sc_count >= 1 and state.race_control.vsc_count >= 1


def test_the_checker_catches_broken_states(two_car_world: WorldState) -> None:
    state = engine.step(two_car_world).state
    assert state_violations(state) == []
    ham = state.car(HAM)
    broken = state.model_copy(
        update={
            "cars": tuple(
                c.model_copy(update={"distance_m": 1e9, "status": CarStatus.FINISHED})
                if c.driver_id == HAM
                else c
                for c in state.cars
            )
        }
    )
    found = " | ".join(state_violations(broken))
    assert "distance" in found and "finished" in found
    backwards = state.model_copy(update={"sim_tick": state.sim_tick})
    assert any("tick went" in v for v in step_violations(state, backwards, ()))
    assert ham.laps_completed == 0


def test_runner_debug_mode_checks_every_tick(isolated_setup: object) -> None:
    from f1ar.domain.race_state import RaceSetup

    assert isinstance(isolated_setup, RaceSetup)
    runner = WorldRunner(
        engine.create_world(isolated_setup, seed=2, world_id="w"),
        clock=lambda: 0.0,
        check_invariants=True,
    )
    for key, type_ in (("s", CommandType.START), ("p", CommandType.PAUSE)):
        runner.submit(Command(idempotency_key=key, type=type_))
    runner.pump(0.0)
    done = runner.submit(Command(idempotency_key="go", type=CommandType.STEP, ticks=10_000))
    runner.pump(0.0)
    assert done.result().accepted
    assert runner.state.world_status is WorldStatus.COMPLETED
