import pytest

from f1ar.domain.race_state import WorldState, WorldStatus
from f1ar.simulation import lightweight_engine as engine
from f1ar.simulation.timeline import SeedMode, branch_seed, fork_world, new_timeline_id
from tests.conftest import run_ticks


def test_fork_copies_state_under_a_new_timeline(two_car_world: WorldState) -> None:
    source = engine.pause(run_ticks(two_car_world, 300))
    branch = fork_world(source, new_timeline_id="timeline_b")
    assert branch.timeline_id == "timeline_b"
    assert branch.parent_timeline_id == source.timeline_id
    assert branch.world_id == source.world_id
    assert branch.world_status is WorldStatus.PAUSED
    assert (branch.cars, branch.sim_tick, branch.rng, branch.setup) == (
        source.cars,
        source.sim_tick,
        source.rng,
        source.setup,
    )


def test_running_source_is_forked_as_paused(two_car_world: WorldState) -> None:
    branch = fork_world(run_ticks(two_car_world, 5), new_timeline_id="timeline_b")
    assert branch.world_status is WorldStatus.PAUSED


def test_fork_needs_a_new_id(two_car_world: WorldState) -> None:
    with pytest.raises(ValueError, match="new timeline id"):
        fork_world(two_car_world, new_timeline_id=two_car_world.timeline_id)


def test_independent_seed_is_new_but_reproducible(two_car_world: WorldState) -> None:
    a = fork_world(two_car_world, new_timeline_id="t_x", seed_mode=SeedMode.INDEPENDENT)
    b = fork_world(two_car_world, new_timeline_id="t_x", seed_mode=SeedMode.INDEPENDENT)
    c = fork_world(two_car_world, new_timeline_id="t_y", seed_mode=SeedMode.INDEPENDENT)
    assert a.rng.seed == b.rng.seed == branch_seed(two_car_world.rng.seed, "t_x")
    assert a.rng.seed not in (two_car_world.rng.seed, c.rng.seed)
    assert a.rng.seed >= 0


def test_new_timeline_ids_are_unique() -> None:
    assert len({new_timeline_id() for _ in range(1000)}) == 1000
