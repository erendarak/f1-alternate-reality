"""Timeline forks through the store (plan §4.7 tests 5-6, US-04)."""

from collections.abc import Iterator
from pathlib import Path

import pytest

from f1ar.domain.race_state import PaceMode, SetPaceMode, WorldState, WorldStatus
from f1ar.persistence.sqlite_store import SnapshotStore
from f1ar.simulation import lightweight_engine as engine
from f1ar.simulation.timeline import SeedMode, fork_world
from tests.conftest import HAM, VER, run_ticks, run_to_end

LAP_18_TICKS = 18 * 168


@pytest.fixture
def store(tmp_path: Path) -> Iterator[SnapshotStore]:
    s = SnapshotStore(tmp_path / "worlds.sqlite")
    yield s
    s.close()


def _finish_with(branch: WorldState, mode: PaceMode | None) -> WorldState:
    state = engine.resume(branch)
    if mode is not None:
        state = engine.apply_action(state, SetPaceMode(driver_id=HAM, pace_mode=mode)).state
    return run_to_end(state)


def test_fork_with_different_decision_leaves_source_untouched(
    store: SnapshotStore, two_car_world: WorldState
) -> None:
    """Plan §4.7 test 5: fork A->B, B takes another decision, A is unchanged and B differs."""
    reference_final = run_to_end(two_car_world)
    a = engine.pause(run_ticks(two_car_world, LAP_18_TICKS))
    a_meta = store.save(a, label="lap 18")

    b, b_meta = store.fork(a_meta.snapshot_id, label="HAM pushes from lap 18")
    assert b.timeline_id != a.timeline_id and b.parent_timeline_id == a.timeline_id
    b_final = _finish_with(b, PaceMode.PUSH)

    # A: snapshot bytes identical, and continuing A still gives the original race.
    assert store.load(a_meta.snapshot_id) == a
    assert store.get_meta(a_meta.snapshot_id).payload_sha256 == a_meta.payload_sha256
    a_final = run_to_end(engine.resume(store.load(a_meta.snapshot_id)))
    assert a_final.cars == reference_final.cars

    # B: Hamilton's race changed; Verstappen's did not (no interaction model, common noise).
    assert b_final.car(HAM).finish_time_s != a_final.car(HAM).finish_time_s
    assert b_final.car(VER).finish_time_s == a_final.car(VER).finish_time_s
    assert b_meta.timeline_id == b.timeline_id


def test_counterfactual_is_reproducible(store: SnapshotStore, two_car_world: WorldState) -> None:
    """Same snapshot + same decision + same seed mode -> same alternate history."""
    meta = store.save(engine.pause(run_ticks(two_car_world, LAP_18_TICKS)), label="lap 18")
    b1, _ = store.fork(meta.snapshot_id, label="push 1")
    b2, _ = store.fork(meta.snapshot_id, label="push 2")
    assert b1.timeline_id != b2.timeline_id
    final_1 = _finish_with(b1, PaceMode.PUSH)
    final_2 = _finish_with(b2, PaceMode.PUSH)
    assert final_1.cars == final_2.cars


def test_independent_seed_branch_gets_its_own_luck(
    store: SnapshotStore, two_car_world: WorldState
) -> None:
    meta = store.save(engine.pause(run_ticks(two_car_world, LAP_18_TICKS)), label="lap 18")
    common, _ = store.fork(meta.snapshot_id, label="same luck")
    indep, _ = store.fork(meta.snapshot_id, label="new luck", seed_mode=SeedMode.INDEPENDENT)
    # No decision changed, so only the dice differ.
    assert (
        _finish_with(common, None).car(VER).finish_time_s
        != _finish_with(indep, None).car(VER).finish_time_s
    )


def test_deleting_the_source_snapshot_does_not_break_branches(
    store: SnapshotStore, two_car_world: WorldState
) -> None:
    """Plan §4.7 test 6: branches hold full copies, not references."""
    a_meta = store.save(engine.pause(run_ticks(two_car_world, LAP_18_TICKS)), label="lap 18")
    b, b_meta = store.fork(a_meta.snapshot_id, label="branch")
    store.delete_snapshot(a_meta.snapshot_id)

    reloaded = store.load(b_meta.snapshot_id)
    assert reloaded == b
    assert run_to_end(engine.resume(reloaded)).world_status is WorldStatus.COMPLETED
    lineage = {t.timeline_id: t for t in store.list_timelines(b.world_id)}
    assert lineage[b.timeline_id].fork_snapshot_id == a_meta.snapshot_id  # history survives
    assert lineage[b.timeline_id].parent_timeline_id == two_car_world.timeline_id


def test_timeline_listing_shows_the_tree(store: SnapshotStore, two_car_world: WorldState) -> None:
    root = store.save(engine.pause(run_ticks(two_car_world, 100)), label="t100")
    b, _ = store.fork(root.snapshot_id, label="B")
    b_later = store.save(run_ticks(engine.resume(b), 50), label="B t150")
    c, _ = store.fork(b_later.snapshot_id, label="C from B")

    timelines = {t.label: t for t in store.list_timelines(two_car_world.world_id)}
    assert set(timelines) == {"main", "B", "C from B"}
    assert timelines["main"].parent_timeline_id is None
    assert timelines["B"].parent_timeline_id == timelines["main"].timeline_id
    assert timelines["C from B"].parent_timeline_id == timelines["B"].timeline_id
    assert timelines["B"].snapshot_count == 2  # branch point + t150
    assert c.sim_tick == 150


def test_saving_an_in_memory_fork_keeps_lineage(
    store: SnapshotStore, two_car_world: WorldState
) -> None:
    """A fork whose parent was never saved still records where it came from."""
    branch = fork_world(run_ticks(two_car_world, 10), new_timeline_id="timeline_mem")
    store.save(branch, label="in-memory fork")
    timelines = {t.timeline_id: t for t in store.list_timelines(branch.world_id)}
    assert timelines["timeline_mem"].parent_timeline_id == two_car_world.timeline_id
    assert timelines[two_car_world.timeline_id].label == "(unsaved)"


def test_late_push_is_faster_on_the_next_lap(
    store: SnapshotStore, two_car_world: WorldState
) -> None:
    """Regression (engine v0.1.0 bug): pushing on old tyres must not add retroactive wear.

    With common random numbers, HAM's first pushed lap must beat the same lap in main.
    """
    meta = store.save(engine.pause(run_ticks(two_car_world, 43 * 168)), label="lap 43")
    main, _ = store.fork(meta.snapshot_id, label="stay")
    push, _ = store.fork(meta.snapshot_id, label="push")
    push = engine.apply_action(push, SetPaceMode(driver_id=HAM, pace_mode=PaceMode.PUSH)).state

    def next_lap_time(state: WorldState) -> float:
        state = engine.resume(state)
        start_lap = state.car(HAM).laps_completed
        while state.car(HAM).laps_completed < start_lap + 2:  # finish current, then one pushed lap
            state = engine.step(state).state
        time = state.car(HAM).last_lap_time_s
        assert time is not None
        return time

    assert next_lap_time(push) < next_lap_time(main) - 0.3
