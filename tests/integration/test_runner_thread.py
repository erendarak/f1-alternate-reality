"""WorldRunner on a real background thread with the real clock (plan §4.2 latency target)."""

import time

from f1ar.domain.race_state import RaceSetup, WorldStatus
from f1ar.simulation import lightweight_engine as engine
from f1ar.simulation.world_runner import Command, CommandType, WorldRunner

PAUSE_ACK_BUDGET_S = 0.25  # plan §4.2: < 250 ms UI acknowledgement


def test_live_pause_latency_freeze_and_resume(two_car_setup: RaceSetup) -> None:
    world = engine.create_world(two_car_setup, seed=42, world_id="world_live")
    runner = WorldRunner(world, speed=25)  # 0.5 s ticks at x25 -> 50 ticks per real second
    runner.start_background()
    try:
        assert (
            runner.submit(Command(idempotency_key="s", type=CommandType.START)).result(1).accepted
        )
        time.sleep(0.4)
        assert runner.state.sim_tick > 0

        t0 = time.perf_counter()
        ack = runner.submit(Command(idempotency_key="p", type=CommandType.PAUSE)).result(1)
        latency = time.perf_counter() - t0
        assert ack.accepted and latency < PAUSE_ACK_BUDGET_S
        assert runner.state.sim_tick == ack.applied_tick

        frozen = runner.state
        time.sleep(1.0)  # real time passes while paused
        assert runner.state == frozen

        runner.submit(Command(idempotency_key="r", type=CommandType.RESUME)).result(1)
        time.sleep(0.3)
        gained = runner.state.sim_tick - frozen.sim_tick
        # ~15 ticks expected for 0.3 s at 50 ticks/s; the paused second (~50) is NOT caught up.
        assert 0 < gained < 40
        assert runner.state.world_status is WorldStatus.RUNNING
    finally:
        runner.stop_background()
