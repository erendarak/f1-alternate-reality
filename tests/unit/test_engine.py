import pytest

from f1ar.domain.race_state import CarStatus, RaceEvent, RaceSetup, WorldState, WorldStatus
from f1ar.simulation import lightweight_engine as engine
from tests.conftest import HAM, VER, run_ticks, run_to_end


def test_new_world_is_created_on_the_line(two_car_setup: RaceSetup) -> None:
    world = engine.create_world(two_car_setup, seed=42, world_id="w")
    assert world.world_status is WorldStatus.CREATED
    assert world.sim_tick == 0 and world.sim_clock_s == 0.0
    assert all(c.distance_m == 0.0 and c.laps_completed == 0 for c in world.cars)
    # Cars are level at the start, so grid order (HAM P6, VER P7) decides the standings.
    assert engine.standings(world) == (HAM, VER)


def test_cannot_step_before_start_or_start_twice(two_car_setup: RaceSetup) -> None:
    world = engine.create_world(two_car_setup, seed=42, world_id="w")
    with pytest.raises(engine.EngineStateError):
        engine.step(world)
    with pytest.raises(engine.EngineStateError):
        engine.start(engine.start(world))


def test_one_tick_advances_time_and_cars(two_car_world: WorldState) -> None:
    result = engine.step(two_car_world)
    s = result.state
    assert s.sim_tick == 1
    assert s.sim_clock_s == two_car_world.setup.tick_seconds
    for before, after in zip(two_car_world.cars, s.cars, strict=True):
        assert after.distance_m > before.distance_m
        assert after.lap_elapsed_s == pytest.approx(s.setup.tick_seconds)
    assert result.events == ()


def test_step_does_not_mutate_its_input(two_car_world: WorldState) -> None:
    before = two_car_world.canonical_hash()
    engine.step(two_car_world)
    assert two_car_world.canonical_hash() == before


def test_lap_completion_emits_ordered_events(two_car_world: WorldState) -> None:
    state: WorldState = two_car_world
    events: list[RaceEvent] = []
    while not any(e.type == "LAP_COMPLETED" for e in events):
        result = engine.step(state)
        state, events = result.state, events + list(result.events)
    lap_events = [e for e in events if e.type == "LAP_COMPLETED"]
    assert all(e.payload["lap"] == 1 for e in lap_events)
    assert [e.seq for e in events] == list(range(len(events)))
    assert state.event_seq == len(events)
    for e in lap_events:
        assert e.driver_id is not None
        car = state.car(e.driver_id)
        assert car.laps_completed == 1 and car.tyre_age_laps == 1
        assert car.last_lap_time_s == e.payload["lap_time_s"]


def test_full_race_invariants_hold_every_tick(two_car_world: WorldState) -> None:
    """Plan §18.2 invariants, checked on every tick of a full 53-lap race."""
    state = two_car_world
    length = state.setup.track.length_m
    total = state.setup.track.total_laps
    seqs: list[int] = []
    while state.world_status is WorldStatus.RUNNING:
        result = engine.step(state)
        prev, state = state, result.state
        assert state.sim_tick == prev.sim_tick + 1
        assert state.sim_clock_s == state.sim_tick * state.setup.tick_seconds
        for c, p in zip(state.cars, prev.cars, strict=True):
            assert 0.0 <= c.distance_m < length
            assert 0 <= c.laps_completed <= total
            assert c.laps_completed >= p.laps_completed
            assert c.tyre_age_laps == c.laps_completed  # no pit stops yet
            if c.status is CarStatus.FINISHED:
                assert c.laps_completed == total and c.finish_time_s is not None
        seqs += [e.seq for e in result.events]
        # Round-trips through JSON (the future save format) without loss.
        if state.sim_tick % 1000 == 0:
            again = WorldState.model_validate_json(state.model_dump_json())
            assert again.canonical_hash() == state.canonical_hash()

    assert state.world_status is WorldStatus.COMPLETED
    assert seqs == list(range(len(seqs)))
    finish = {c.driver_id: c.finish_time_s for c in state.cars}
    assert engine.standings(state) == tuple(sorted(finish, key=lambda d: finish[d] or 0.0))
    with pytest.raises(engine.EngineStateError):
        engine.step(state)


def test_race_duration_is_plausible_for_monza(two_car_world: WorldState) -> None:
    """Sanity bound only; the coefficients are uncalibrated placeholders."""
    final = run_to_end(two_car_world)
    for c in final.cars:
        assert c.finish_time_s is not None
        assert 60 * 60 < c.finish_time_s < 90 * 60


def test_run_ticks_helper_matches_manual_stepping(two_car_world: WorldState) -> None:
    manual = two_car_world
    for _ in range(25):
        manual = engine.step(manual).state
    assert run_ticks(two_car_world, 25) == manual
