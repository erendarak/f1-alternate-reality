"""WorldRunner with a fake clock: exact, instant tests of pause/resume semantics (plan §4.7)."""

from pathlib import Path
from typing import Any

import pytest

from f1ar.domain.race_state import RaceEvent, RaceSetup, WorldState, WorldStatus
from f1ar.persistence.sqlite_store import SnapshotStore
from f1ar.simulation import lightweight_engine as engine
from f1ar.simulation.world_runner import Command, CommandAck, CommandType, WorldRunner
from tests.conftest import run_to_end


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


_counter = 0


def cmd(type_: CommandType, **kw: Any) -> Command:
    global _counter
    _counter += 1
    return Command(idempotency_key=kw.pop("key", f"k{_counter}"), type=type_, **kw)


def send(runner: WorldRunner, command: Command, now: float) -> CommandAck:
    future = runner.submit(command)
    runner.pump(now)
    return future.result(timeout=0)


def frozen_view(state: WorldState) -> dict[str, Any]:
    """Everything except the status flag, which pausing is supposed to change."""
    return state.model_dump(mode="json", exclude={"world_status"})


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def runner(two_car_setup: RaceSetup, clock: FakeClock) -> WorldRunner:
    world = engine.create_world(two_car_setup, seed=42, world_id="world_test")
    return WorldRunner(world, clock=clock)


def started(runner: WorldRunner, now: float = 0.0) -> WorldRunner:
    assert send(runner, cmd(CommandType.START), now).accepted
    return runner


# --- pacing -------------------------------------------------------------------------------


def test_created_world_does_not_move_until_started(runner: WorldRunner) -> None:
    runner.pump(100.0)
    assert runner.state.sim_tick == 0 and runner.state.world_status is WorldStatus.CREATED


def test_real_time_pacing_at_x1_and_x5(runner: WorldRunner) -> None:
    started(runner)
    runner.pump(10.0)  # 10 real s at x1, 0.5 s ticks
    assert runner.state.sim_tick == 20
    send(runner, cmd(CommandType.SET_SPEED, speed=5), 10.0)
    runner.pump(11.0)  # +1 real s at x5 = 5 sim s = 10 ticks
    assert runner.state.sim_tick == 30


def test_invalid_speed_is_rejected(runner: WorldRunner) -> None:
    ack = send(runner, cmd(CommandType.SET_SPEED, speed=3), 0.0)
    assert not ack.accepted and "speed" in (ack.reason or "")
    assert runner.speed == 1


def test_catchup_is_capped_and_never_jumps(two_car_setup: RaceSetup, clock: FakeClock) -> None:
    world = engine.create_world(two_car_setup, seed=42, world_id="w")
    runner = started(WorldRunner(world, clock=clock, max_catchup_ticks=10))
    runner.pump(100.0)  # 200 ticks due, cap is 10
    assert runner.state.sim_tick == 10
    runner.pump(100.0)  # re-anchored: nothing more is "owed"
    assert runner.state.sim_tick == 10


# --- pause / resume -------------------------------------------------------------------------


def test_pause_freezes_everything(runner: WorldRunner) -> None:
    """Plan §4.7 test 3: tick, tyres, clocks, events and car positions don't change."""
    started(runner)
    runner.pump(50.0)
    ack = send(runner, cmd(CommandType.PAUSE), 50.0)
    assert ack.accepted and ack.world_status is WorldStatus.PAUSED
    assert ack.applied_tick == runner.state.sim_tick == 100

    frozen, n_events = frozen_view(runner.state), len(runner.events)
    for t in (51.0, 500.0, 10_000.0):
        runner.pump(t)
        assert frozen_view(runner.state) == frozen
        assert len(runner.events) == n_events


def test_paused_wall_time_is_not_caught_up(runner: WorldRunner) -> None:
    """Plan §4.7 test 4: race time only advances after resume."""
    started(runner)
    runner.pump(10.0)
    send(runner, cmd(CommandType.PAUSE), 10.0)
    runner.pump(15.0)  # 5 real seconds paused
    assert runner.state.sim_tick == 20
    send(runner, cmd(CommandType.RESUME), 15.0)
    runner.pump(16.0)  # 1 real second after resume
    assert runner.state.sim_tick == 22  # not 32


def test_pause_mid_burst_lands_on_the_next_tick_boundary(runner: WorldRunner) -> None:
    """A pause arriving while a burst of ticks is running applies before the next tick."""
    started(runner)

    def pause_at_50(state: WorldState, _: tuple[RaceEvent, ...]) -> None:
        if state.sim_tick == 50:
            runner.submit(cmd(CommandType.PAUSE))

    runner.add_listener(pause_at_50)
    runner.pump(1000.0)  # 2000 ticks due
    assert runner.state.world_status is WorldStatus.PAUSED
    assert runner.state.sim_tick == 50


def test_pause_exactly_on_a_lap_event_is_consistent(runner: WorldRunner) -> None:
    """Plan §4.7 test 10: pausing at an event boundary leaves state and event log in agreement."""
    started(runner)

    def pause_on_first_lap(_: WorldState, events: tuple[RaceEvent, ...]) -> None:
        if any(e.type == "LAP_COMPLETED" for e in events):
            runner.submit(cmd(CommandType.PAUSE))

    runner.add_listener(pause_on_first_lap)
    runner.pump(1000.0)
    state = runner.state
    assert state.world_status is WorldStatus.PAUSED
    laps_logged = [e for e in runner.events if e.type == "LAP_COMPLETED"]
    assert len(laps_logged) == sum(c.laps_completed for c in state.cars) >= 1
    assert state.event_seq == len(runner.events)
    assert all(e.tick <= state.sim_tick for e in runner.events)


def test_resume_requires_paused_and_pause_requires_running(runner: WorldRunner) -> None:
    assert not send(runner, cmd(CommandType.PAUSE), 0.0).accepted  # CREATED
    assert not send(runner, cmd(CommandType.RESUME), 0.0).accepted  # CREATED
    started(runner)
    assert not send(runner, cmd(CommandType.RESUME), 0.0).accepted  # RUNNING
    log = runner.command_log
    assert [entry.ack.accepted for entry in log] == [False, False, True, False]


# --- manual stepping, idempotency ----------------------------------------------------------


def test_step_while_paused_stays_paused(runner: WorldRunner) -> None:
    started(runner)
    runner.pump(5.0)
    send(runner, cmd(CommandType.PAUSE), 5.0)
    ack = send(runner, cmd(CommandType.STEP, ticks=7), 5.0)
    assert ack.accepted and ack.applied_tick == 17
    assert runner.state.world_status is WorldStatus.PAUSED
    runner.pump(500.0)
    assert runner.state.sim_tick == 17


def test_step_while_running_is_rejected(runner: WorldRunner) -> None:
    started(runner)
    ack = send(runner, cmd(CommandType.STEP, ticks=3), 0.0)
    assert not ack.accepted and "PAUSED" in (ack.reason or "")


def test_duplicate_command_is_applied_once(runner: WorldRunner) -> None:
    """Plan §4.7 test 9: same idempotency key twice -> second time is not applied."""
    started(runner)
    send(runner, cmd(CommandType.PAUSE), 0.0)
    first = send(runner, cmd(CommandType.STEP, ticks=5, key="double-click"), 0.0)
    second = send(runner, cmd(CommandType.STEP, ticks=5, key="double-click"), 0.0)
    assert runner.state.sim_tick == 5
    assert second.duplicate and not first.duplicate
    assert second.applied_tick == first.applied_tick
    assert len(runner.command_log) == 3  # START, PAUSE, STEP: the duplicate isn't logged again


# --- saving ------------------------------------------------------------------------------------


def test_save_while_paused_and_running(
    two_car_setup: RaceSetup, clock: FakeClock, tmp_path: Path
) -> None:
    with SnapshotStore(tmp_path / "w.sqlite") as store:
        world = engine.create_world(two_car_setup, seed=42, world_id="w")
        runner = started(WorldRunner(world, clock=clock, store=store))
        runner.pump(30.0)

        running_ack = send(runner, cmd(CommandType.SAVE, label="while running"), 30.0)
        assert running_ack.accepted and running_ack.snapshot_id is not None
        saved = store.load(running_ack.snapshot_id)
        assert saved.sim_tick == running_ack.applied_tick == 60
        assert saved.world_status is WorldStatus.RUNNING

        send(runner, cmd(CommandType.PAUSE), 30.0)
        paused_ack = send(runner, cmd(CommandType.SAVE), 30.0)
        assert paused_ack.snapshot_id is not None
        assert store.load(paused_ack.snapshot_id) == runner.state
        assert store.get_meta(paused_ack.snapshot_id).label == "tick 60"


def test_save_without_store_is_rejected(runner: WorldRunner) -> None:
    ack = send(runner, cmd(CommandType.SAVE), 0.0)
    assert not ack.accepted and ack.snapshot_id is None


def test_failed_save_never_reports_success(
    two_car_setup: RaceSetup, clock: FakeClock, tmp_path: Path
) -> None:
    """Plan US-09: a failing save is reported as failed and the world keeps going."""
    store = SnapshotStore(tmp_path / "w.sqlite")
    world = engine.create_world(two_car_setup, seed=42, world_id="w")
    runner = started(WorldRunner(world, clock=clock, store=store))
    store.close()  # simulate the database becoming unavailable
    ack = send(runner, cmd(CommandType.SAVE), 1.0)
    assert not ack.accepted and ack.snapshot_id is None
    assert "save failed" in (ack.reason or "")
    runner.pump(2.0)
    assert runner.state.sim_tick == 4


# --- determinism is independent of pacing ----------------------------------------------------


def test_pacing_and_pauses_never_change_the_race(
    two_car_setup: RaceSetup, clock: FakeClock
) -> None:
    """Same commands at different speeds and wall times -> same canonical race."""
    reference = run_to_end(engine.start(engine.create_world(two_car_setup, seed=42, world_id="w")))

    world = engine.create_world(two_car_setup, seed=42, world_id="w")
    runner = started(WorldRunner(world, clock=clock, max_catchup_ticks=100_000))
    t = 0.0
    for i in range(40):
        t += 7.3
        runner.pump(t)
        if i % 3 == 0:
            send(runner, cmd(CommandType.PAUSE), t)
            t += 123.0  # long real-time pause
            send(runner, cmd(CommandType.RESUME), t)
        if i % 5 == 0:
            send(runner, cmd(CommandType.SET_SPEED, speed=[1, 5, 25, 2][i % 4]), t)
    while runner.state.world_status is not WorldStatus.COMPLETED:
        t += 60.0
        runner.pump(t)

    assert runner.state.canonical_hash() == reference.canonical_hash()
