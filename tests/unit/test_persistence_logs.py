"""Event/command logs, resume-as-branch, autosave and shutdown (store v3, M2 slice 4)."""

from collections.abc import Iterator
from pathlib import Path

import pytest

from f1ar.domain.race_state import Compound, RaceEvent, WorldState, WorldStatus
from f1ar.persistence.sqlite_store import (
    CommandRecord,
    SnapshotIntegrityError,
    SnapshotStore,
    UnsupportedSnapshotError,
)
from f1ar.simulation import lightweight_engine as engine
from f1ar.simulation.world_runner import (
    AutosavePolicy,
    Command,
    CommandAck,
    WorldRunner,
)
from tests.conftest import HAM, run_ticks


@pytest.fixture
def store(tmp_path: Path) -> Iterator[SnapshotStore]:
    s = SnapshotStore(tmp_path / "logs.sqlite")
    yield s
    s.close()


def _events_of(world: WorldState, ticks: int) -> tuple[WorldState, list[RaceEvent]]:
    events: list[RaceEvent] = []
    for _ in range(ticks):
        result = engine.step(world)
        world = result.state
        events += result.events
    return world, events


def _send(runner: WorldRunner, key: str, **fields: object) -> CommandAck:
    future = runner.submit(Command.model_validate({"idempotency_key": key, **fields}))
    runner.pump(0.0)
    return future.result(timeout=5)


# --- the store ---------------------------------------------------------------------------------


def test_snapshot_and_its_events_are_saved_together(
    store: SnapshotStore, two_car_world: WorldState
) -> None:
    state, events = _events_of(two_car_world, 400)
    meta = store.save(state, label="lap 2", events=events)
    assert meta.event_seq == state.event_seq == len(events)
    assert store.events(state.timeline_id) == events
    store.append_log(state, events=events)  # idempotent: nothing duplicated
    assert len(store.events(state.timeline_id)) == len(events)


def test_branch_history_is_parent_prefix_plus_own_events(
    store: SnapshotStore, two_car_world: WorldState
) -> None:
    a, a_events = _events_of(two_car_world, 600)
    meta = store.save(engine.pause(a), label="branch point", events=a_events)
    a_more, more = _events_of(a, 300)  # the main timeline carries on...
    store.append_log(a_more, events=more)

    b, _ = store.fork(meta.snapshot_id, label="other future")
    assert b.branch_event_seq == len(a_events)
    b_run, b_events = _events_of(engine.resume(b), 300)
    store.append_log(b_run, events=b_events)

    history = store.events(b.timeline_id)
    assert history == a_events + b_events  # not the main timeline's later events
    assert [e.seq for e in history] == list(range(len(history)))
    assert store.events(a.timeline_id) == a_events + more  # main untouched


def test_resuming_an_old_save_never_rewrites_history(
    store: SnapshotStore, two_car_world: WorldState
) -> None:
    early, early_events = _events_of(two_car_world, 300)
    first = store.save(early, label="early", events=early_events)
    late, late_events = _events_of(early, 300)
    last = store.save(late, label="late", events=late_events)

    same = store.resume(last.snapshot_id)  # the newest point: just continue
    assert same.timeline_id == late.timeline_id
    branch = store.resume(first.snapshot_id)  # an older point: a new branch
    assert branch.timeline_id != early.timeline_id
    assert branch.parent_timeline_id == early.timeline_id
    assert store.events(branch.timeline_id) == early_events
    assert store.events(early.timeline_id) == early_events + late_events


def test_commands_are_logged_with_their_cursor(
    store: SnapshotStore, two_car_world: WorldState
) -> None:
    record = CommandRecord(
        idempotency_key="k1",
        log_seq=0,
        type="PIT_REQUEST",
        actor_id="USER",
        applied_tick=0,
        accepted=True,
        command_json="{}",
        ack_json="{}",
    )
    meta = store.save(two_car_world, label="x", commands=[record])
    assert meta.command_seq == 1
    assert store.commands(two_car_world.timeline_id) == [record]


def test_a_save_made_with_other_policies_is_refused(
    store: SnapshotStore, two_car_world: WorldState
) -> None:
    other = two_car_world.setup.model_copy(update={"policy_manifest": {"strategy": "rl_v9"}})
    meta = store.save(two_car_world.model_copy(update={"setup": other}), label="foreign")
    assert meta.policy_manifest == '{"strategy": "rl_v9"}'
    with pytest.raises(UnsupportedSnapshotError, match="policies"):
        store.load(meta.snapshot_id)
    with pytest.raises(engine.EngineStateError, match="pins policies"):
        engine.create_world(other, seed=1, world_id="w")


# --- the runner --------------------------------------------------------------------------------


def test_runner_logs_everything_and_idempotency_survives_a_restart(
    store: SnapshotStore, isolated_world: WorldState
) -> None:
    created = engine.create_world(isolated_world.setup, seed=3, world_id="w_idem")
    runner = WorldRunner(created, clock=lambda: 0.0, store=store)
    _send(runner, "start", type="START")
    _send(runner, "pause", type="PAUSE")  # pausing writes the log
    _send(runner, "step", type="STEP", ticks=200)
    box = _send(runner, "box-ham", type="PIT_REQUEST", driver_id=HAM, compound="HARD")
    assert box.accepted
    saved = _send(runner, "save", type="SAVE", label="mine")
    assert saved.snapshot_id is not None
    assert store.events(created.timeline_id) == list(runner.events)
    assert [c.idempotency_key for c in store.commands(created.timeline_id)] == [
        "start",
        "pause",
        "step",
        "box-ham",
    ]  # the SAVE's own record is written with the next flush

    # "Restart the app": a new runner on the saved world. The repeated pit request is a
    # duplicate (not applied twice), a new command is applied.
    restored = WorldRunner(store.resume(saved.snapshot_id), clock=lambda: 0.0, store=store)
    again = _send(restored, "box-ham", type="PIT_REQUEST", driver_id=HAM, compound="SOFT")
    assert again.duplicate and again.applied_tick == box.applied_tick
    assert restored.state.car(HAM).pit_request is Compound.HARD
    assert _send(restored, "new", type="STEP", ticks=1).accepted


def test_autosave_every_laps_keeps_the_newest(
    store: SnapshotStore, isolated_world: WorldState
) -> None:
    created = engine.create_world(isolated_world.setup, seed=4, world_id="w_auto")
    policy = AutosavePolicy(every_laps=5, keep=3)
    runner = WorldRunner(created, clock=lambda: 0.0, store=store, autosave=policy)
    _send(runner, "start", type="START")
    _send(runner, "pause", type="PAUSE")
    manual = _send(runner, "save", type="SAVE", label="before")
    _send(runner, "step", type="STEP", ticks=10_000)  # to the flag
    assert runner.state.world_status is WorldStatus.COMPLETED

    saves = store.list_snapshots("w_auto")
    autos = [m for m in saves if m.kind == "AUTOSAVE"]
    assert len(autos) == policy.keep  # pruned to the newest 3
    assert autos[0].label == "autosave: race finished"
    assert {m.label for m in autos[1:]} <= {f"autosave: lap {n}" for n in range(5, 54, 5)}
    assert manual.snapshot_id in {m.snapshot_id for m in saves}  # manual saves are never pruned
    assert store.events(created.timeline_id) == list(runner.events)


def test_critical_commands_are_autosaved(store: SnapshotStore, isolated_world: WorldState) -> None:
    created = engine.create_world(isolated_world.setup, seed=5, world_id="w_crit")
    runner = WorldRunner(
        created, clock=lambda: 0.0, store=store, autosave=AutosavePolicy(every_laps=50)
    )
    _send(runner, "start", type="START")
    _send(runner, "sc", type="DEPLOY_SC", laps=2)
    newest = store.list_snapshots("w_crit")[0]
    assert newest.label == "autosave: after DEPLOY_SC" and newest.command_seq == 2
    assert runner.persistence["last_autosave"] == newest.snapshot_id


def test_a_failing_store_never_stops_the_race(tmp_path: Path, isolated_world: WorldState) -> None:
    broken = SnapshotStore(tmp_path / "broken.sqlite")
    created = engine.create_world(isolated_world.setup, seed=6, world_id="w_broken")
    runner = WorldRunner(
        created, clock=lambda: 0.0, store=broken, autosave=AutosavePolicy(every_laps=1)
    )
    _send(runner, "start", type="START")
    _send(runner, "pause", type="PAUSE")
    broken.close()  # the disk "goes away"
    step = _send(runner, "step", type="STEP", ticks=600)
    assert step.accepted and runner.state.sim_tick == 600  # the race went on
    assert str(runner.persistence["error"]).startswith(("autosave failed", "log write failed"))
    failed = _send(runner, "save", type="SAVE")
    assert not failed.accepted and (failed.reason or "").startswith("save failed")


def test_shutdown_leaves_a_paused_saved_world(
    store: SnapshotStore, isolated_world: WorldState
) -> None:
    created = engine.create_world(isolated_world.setup, seed=8, world_id="w_shut")
    runner = WorldRunner(
        created, clock=lambda: 0.0, store=store, autosave=AutosavePolicy(every_laps=50)
    )
    _send(runner, "start", type="START")
    runner.pump(30.0)  # 60 ticks of real-time racing
    assert runner.state.world_status is WorldStatus.RUNNING
    runner.shutdown()
    latest = store.latest_valid("w_shut")
    assert latest is not None and latest.label == "autosave: shutdown"
    restored = store.load(latest.snapshot_id)
    assert restored.world_status is WorldStatus.PAUSED and restored == runner.state
    assert runner.persistence["unsaved_events"] == runner.persistence["unsaved_commands"] == 0


def test_autosave_needs_a_store(isolated_world: WorldState) -> None:
    with pytest.raises(ValueError, match="needs a store"):
        WorldRunner(isolated_world, autosave=AutosavePolicy())


def test_run_ticks_helper_still_works(two_car_world: WorldState) -> None:
    assert run_ticks(two_car_world, 3).sim_tick == 3


def test_two_histories_under_one_id_are_refused(
    store: SnapshotStore, two_car_world: WorldState
) -> None:
    """Regression: world/timeline ids were reused (every world's main timeline was
    "timeline_main", the viewer restarted at "viewer_0"), which would have mixed histories."""
    a, a_events = _events_of(two_car_world, 300)
    store.save(a, label="first run", events=a_events)
    other_seed = engine.start(engine.create_world(a.setup, seed=999, world_id=a.world_id))
    b, b_events = _events_of(other_seed, 300)
    with pytest.raises(SnapshotIntegrityError, match="world id reused"):
        store.append_log(b, events=b_events)
    assert store.events(a.timeline_id) == a_events  # the first history is untouched

    foreign = engine.create_world(a.setup, seed=1, world_id="another_world")
    with pytest.raises(SnapshotIntegrityError, match="belongs to world"):
        store.save(foreign.model_copy(update={"timeline_id": a.timeline_id}), label="x")


def test_default_ids_are_unique_per_world(two_car_world: WorldState) -> None:
    from f1ar.simulation.timeline import main_timeline_id, new_world_id

    assert two_car_world.timeline_id == main_timeline_id(two_car_world.world_id)
    assert new_world_id("viewer") != new_world_id("viewer")
