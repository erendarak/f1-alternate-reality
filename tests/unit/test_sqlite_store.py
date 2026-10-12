import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from f1ar.domain.race_state import WorldState
from f1ar.persistence.snapshot_codec import canonical_json
from f1ar.persistence.sqlite_store import (
    STORE_SCHEMA_VERSION,
    SnapshotIntegrityError,
    SnapshotNotFoundError,
    SnapshotStore,
    UnsupportedSnapshotError,
)
from f1ar.simulation import lightweight_engine as engine
from tests.conftest import run_ticks


@pytest.fixture
def store(tmp_path: Path) -> Iterator[SnapshotStore]:
    with SnapshotStore(tmp_path / "worlds.sqlite") as s:
        yield s


def _raw(path: Path, sql: str, params: tuple[object, ...] = ()) -> None:
    db = sqlite3.connect(path)
    with db:
        db.execute(sql, params)
    db.close()


def test_save_then_load_is_identical(store: SnapshotStore, two_car_world: WorldState) -> None:
    state = run_ticks(two_car_world, 400)
    meta = store.save(state, label="lap 2-ish")
    loaded = store.load(meta.snapshot_id)
    assert loaded == state
    assert loaded.canonical_hash() == state.canonical_hash() == meta.payload_sha256
    assert (meta.tick, meta.world_id, meta.label, meta.kind) == (
        400,
        state.world_id,
        "lap 2-ish",
        "MANUAL",
    )


def test_snapshots_survive_reopening_the_file(tmp_path: Path, two_car_world: WorldState) -> None:
    path = tmp_path / "worlds.sqlite"
    with SnapshotStore(path) as s:
        snapshot_id = s.save(two_car_world, label="a").snapshot_id
    with SnapshotStore(path) as s:
        assert s.load(snapshot_id) == two_car_world


def test_list_is_newest_first_and_filters_by_world(
    store: SnapshotStore, two_car_world: WorldState
) -> None:
    a = store.save(two_car_world, label="first")
    b = store.save(run_ticks(two_car_world, 10), label="second")
    other = two_car_world.model_copy(
        update={"world_id": "other_world", "timeline_id": "other_world.main"}
    )
    store.save(other, label="other")
    assert [m.snapshot_id for m in store.list_snapshots(two_car_world.world_id)] == [
        b.snapshot_id,
        a.snapshot_id,
    ]
    assert len(store.list_snapshots()) == 3
    assert store.get_meta(a.snapshot_id) == a


def test_saving_does_not_modify_the_snapshot_later(
    store: SnapshotStore, two_car_world: WorldState
) -> None:
    """Snapshots are immutable: continuing the race changes nothing that was saved."""
    meta = store.save(two_car_world, label="start")
    run_ticks(two_car_world, 500)
    assert store.load(meta.snapshot_id) == two_car_world


def test_unknown_snapshot(store: SnapshotStore) -> None:
    with pytest.raises(SnapshotNotFoundError):
        store.load("snap_missing")


def _flip_one_byte(path: Path, snapshot_id: str, offset: int = 200) -> None:
    db = sqlite3.connect(path)
    with db:
        (payload,) = db.execute(
            "SELECT payload FROM snapshots WHERE id = ?", (snapshot_id,)
        ).fetchone()
        corrupted = bytearray(payload)
        corrupted[offset] ^= 0x01
        db.execute("UPDATE snapshots SET payload = ? WHERE id = ?", (bytes(corrupted), snapshot_id))
    db.close()


def test_corrupted_payload_is_rejected_and_last_good_offered(
    store: SnapshotStore, two_car_world: WorldState
) -> None:
    """Plan §4.7 test 7: bad checksum is refused; the last sound save is suggested."""
    good = store.save(run_ticks(two_car_world, 100), label="good")
    bad = store.save(run_ticks(two_car_world, 200), label="bad")
    _flip_one_byte(store.path, bad.snapshot_id)
    with pytest.raises(SnapshotIntegrityError, match="checksum"):
        store.load(bad.snapshot_id)
    recovered = store.latest_valid(two_car_world.world_id)
    assert recovered is not None and recovered.snapshot_id == good.snapshot_id


def test_payload_with_wrong_storage_type_is_rejected(
    store: SnapshotStore, two_car_world: WorldState
) -> None:
    """SQLite's || turns a BLOB into TEXT; that kind of damage must fail cleanly too."""
    meta = store.save(two_car_world, label="x")
    _raw(
        store.path,
        "UPDATE snapshots SET payload = payload || ' ' WHERE id = ?",
        (meta.snapshot_id,),
    )
    with pytest.raises(SnapshotIntegrityError, match="storage type"):
        store.load(meta.snapshot_id)


def test_tampered_checksum_column_is_detected(
    store: SnapshotStore, two_car_world: WorldState
) -> None:
    meta = store.save(two_car_world, label="x")
    _raw(
        store.path,
        "UPDATE snapshots SET payload_sha256 = 'sha256:00' WHERE id = ?",
        (meta.snapshot_id,),
    )
    with pytest.raises(SnapshotIntegrityError):
        store.load(meta.snapshot_id)
    assert store.latest_valid(two_car_world.world_id) is None


def test_snapshot_from_another_engine_version_is_refused(
    store: SnapshotStore, two_car_world: WorldState
) -> None:
    old = two_car_world.model_copy(update={"engine_version": "lightweight_engine_v0.0.9"})
    meta = store.save(old, label="old engine")
    with pytest.raises(UnsupportedSnapshotError, match="different engine"):
        store.load(meta.snapshot_id)


def test_store_from_a_newer_schema_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "future.sqlite"
    SnapshotStore(path).close()
    _raw(path, "UPDATE store_meta SET value = '99' WHERE key = 'schema_version'")
    with pytest.raises(UnsupportedSnapshotError, match="migration"):
        SnapshotStore(path)


_V1_DDL = """
CREATE TABLE store_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE worlds (id TEXT PRIMARY KEY, event_id TEXT NOT NULL, created_utc TEXT NOT NULL);
CREATE TABLE timelines (
    id TEXT PRIMARY KEY, world_id TEXT NOT NULL REFERENCES worlds(id),
    parent_timeline_id TEXT REFERENCES timelines(id), created_utc TEXT NOT NULL);
CREATE TABLE snapshots (
    id TEXT PRIMARY KEY, world_id TEXT NOT NULL REFERENCES worlds(id),
    timeline_id TEXT NOT NULL REFERENCES timelines(id), tick INTEGER NOT NULL,
    sim_clock_s REAL NOT NULL, world_status TEXT NOT NULL, label TEXT NOT NULL,
    kind TEXT NOT NULL, schema_version INTEGER NOT NULL, codec_version INTEGER NOT NULL,
    engine_version TEXT NOT NULL, profile_set_hash TEXT NOT NULL, payload_sha256 TEXT NOT NULL,
    payload BLOB NOT NULL, created_utc TEXT NOT NULL, seq INTEGER NOT NULL UNIQUE);
INSERT INTO store_meta VALUES ('schema_version', '1');
"""


def test_v1_store_is_migrated_in_place(tmp_path: Path, two_car_world: WorldState) -> None:
    """A save file from step 5 (store schema v1) migrates v1 -> ... -> v4, loads and forks."""
    path = tmp_path / "v1.sqlite"
    state = run_ticks(two_car_world, 50)
    payload = canonical_json(state.model_dump(mode="json"))
    db = sqlite3.connect(path)
    db.executescript(_V1_DDL)
    with db:
        db.execute("INSERT INTO worlds VALUES (?, ?, 't')", (state.world_id, state.setup.event_id))
        db.execute(
            "INSERT INTO timelines VALUES (?, ?, NULL, 't')", (state.timeline_id, state.world_id)
        )
        db.execute(
            "INSERT INTO snapshots VALUES ('snap_v1', ?, ?, 50, 25.0, 'RUNNING', 'old', 'MANUAL',"
            " ?, 1, ?, ?, ?, ?, 't', 1)",
            (
                state.world_id,
                state.timeline_id,
                state.schema_version,
                state.engine_version,
                state.setup.profile_set_manifest_hash,
                state.canonical_hash(),
                payload,
            ),
        )
    db.close()

    with SnapshotStore(path) as store:
        assert store.schema_version == STORE_SCHEMA_VERSION == 4
        assert store.list_worlds()[0].active_timeline_id is None  # v4 column, not yet set
        assert store.load("snap_v1") == state
        branch, _ = store.fork("snap_v1", label="after migration")
        assert {t.timeline_id for t in store.list_timelines(state.world_id)} == {
            state.timeline_id,
            branch.timeline_id,
        }


def test_continue_point_follows_the_active_timeline(
    store: SnapshotStore, two_car_world: WorldState
) -> None:
    """ "Continue" resumes the timeline last made live, not whichever timeline saved last."""
    main_save = store.save(run_ticks(two_car_world, 20), label="main")
    branch, _ = store.fork(main_save.snapshot_id, label="branch")
    branch_save = store.save(run_ticks(engine.resume(branch), 30), label="branch later")
    world = two_car_world.world_id
    assert store.continue_point(world) == store.get_meta(branch_save.snapshot_id)  # newest
    store.set_active_timeline(world, two_car_world.timeline_id)
    assert store.continue_point(world) == main_save
    listed = store.list_worlds()[0]
    assert listed.active_timeline_id == two_car_world.timeline_id
    assert listed.latest == main_save
    with pytest.raises(SnapshotIntegrityError):
        store.set_active_timeline(world, "timeline_of_someone_else")


def test_delete_snapshot(store: SnapshotStore, two_car_world: WorldState) -> None:
    meta = store.save(two_car_world, label="x")
    store.delete_snapshot(meta.snapshot_id)
    with pytest.raises(SnapshotNotFoundError):
        store.load(meta.snapshot_id)
    with pytest.raises(SnapshotNotFoundError):
        store.delete_snapshot(meta.snapshot_id)
