"""SQLite snapshot store (plan §4.5, §4.6, §13.1).

A snapshot is the canonical JSON encoding of a complete ``WorldState`` plus its SHA-256,
written together with its metadata in ONE transaction. A crash can't leave a half-written
snapshot that looks valid. Every load re-verifies the checksum, the schema version and the
engine version; nothing is silently migrated or loaded with a different engine.

Snapshots are immutable: they are only ever inserted, never updated. Timelines record
their lineage (parent timeline + the snapshot they were forked from), so the history of
alternate realities can be shown even after a source snapshot is deleted.

Store schema history (``STORE_MIGRATIONS``): v1 = worlds/timelines/snapshots;
v2 = timelines gain ``label`` and ``fork_snapshot_id``.
"""

import hashlib
import sqlite3
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, ValidationError

from f1ar.domain.race_state import WORLD_SCHEMA_VERSION, WorldState, WorldStatus
from f1ar.persistence.snapshot_codec import CODEC_VERSION, HASH_PREFIX, canonical_json
from f1ar.simulation.lightweight_engine import ENGINE_VERSION
from f1ar.simulation.timeline import SeedMode, fork_world, new_timeline_id

STORE_SCHEMA_VERSION = 2

STORE_MIGRATIONS: dict[int, tuple[str, ...]] = {
    # from version N to N+1
    1: (
        "ALTER TABLE timelines ADD COLUMN label TEXT NOT NULL DEFAULT ''",
        "ALTER TABLE timelines ADD COLUMN fork_snapshot_id TEXT",
    ),
}

SnapshotKind = Literal["MANUAL", "AUTOSAVE"]

_DDL = """
CREATE TABLE IF NOT EXISTS store_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS worlds (
    id          TEXT PRIMARY KEY,
    event_id    TEXT NOT NULL,
    created_utc TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS timelines (
    id                 TEXT PRIMARY KEY,
    world_id           TEXT NOT NULL REFERENCES worlds(id),
    parent_timeline_id TEXT REFERENCES timelines(id),
    created_utc        TEXT NOT NULL,
    label              TEXT NOT NULL DEFAULT '',
    fork_snapshot_id   TEXT  -- provenance only, deliberately not a foreign key
);
CREATE TABLE IF NOT EXISTS snapshots (
    id                TEXT PRIMARY KEY,
    world_id          TEXT NOT NULL REFERENCES worlds(id),
    timeline_id       TEXT NOT NULL REFERENCES timelines(id),
    tick              INTEGER NOT NULL,
    sim_clock_s       REAL NOT NULL,
    world_status      TEXT NOT NULL,
    label             TEXT NOT NULL,
    kind              TEXT NOT NULL,
    schema_version    INTEGER NOT NULL,
    codec_version     INTEGER NOT NULL,
    engine_version    TEXT NOT NULL,
    profile_set_hash  TEXT NOT NULL,
    payload_sha256    TEXT NOT NULL,
    payload           BLOB NOT NULL,
    created_utc       TEXT NOT NULL,
    seq               INTEGER NOT NULL UNIQUE
);
CREATE INDEX IF NOT EXISTS snapshots_by_world ON snapshots(world_id, seq);
"""


class SnapshotError(Exception):
    """Base class for snapshot problems the user must be told about."""


class SnapshotNotFoundError(SnapshotError):
    pass


class SnapshotIntegrityError(SnapshotError):
    """Stored bytes don't match their checksum, or don't decode to a valid world."""


class UnsupportedSnapshotError(SnapshotError):
    """Snapshot was written by a different schema/codec/engine version than this build."""


class SnapshotMeta(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    snapshot_id: str
    world_id: str
    timeline_id: str
    tick: int
    sim_clock_s: float
    world_status: WorldStatus
    label: str
    kind: SnapshotKind
    schema_version: int
    codec_version: int
    engine_version: str
    profile_set_hash: str
    payload_sha256: str
    payload_bytes: int
    created_utc: str


class TimelineMeta(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    timeline_id: str
    world_id: str
    parent_timeline_id: str | None
    label: str
    fork_snapshot_id: str | None
    created_utc: str
    snapshot_count: int


_META_COLUMNS = (
    "id, world_id, timeline_id, tick, sim_clock_s, world_status, label, kind, schema_version, "
    "codec_version, engine_version, profile_set_hash, payload_sha256, length(payload), created_utc"
)


def _utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _sha256(payload: bytes) -> str:
    return HASH_PREFIX + hashlib.sha256(payload).hexdigest()


class SnapshotStore:
    """One SQLite file holding worlds, timelines and immutable snapshots. Thread-safe."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._lock = threading.Lock()
        # The runner thread saves while other threads read; all access goes through _lock.
        self._db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA foreign_keys=ON")
        self._db.execute("PRAGMA synchronous=FULL")
        with self._lock:
            self._db.executescript(_DDL)
            row = self._db.execute(
                "SELECT value FROM store_meta WHERE key='schema_version'"
            ).fetchone()
            if row is None:
                self._db.execute(
                    "INSERT INTO store_meta(key, value) VALUES ('schema_version', ?)",
                    (str(STORE_SCHEMA_VERSION),),
                )
            elif int(row[0]) > STORE_SCHEMA_VERSION:
                self._db.close()
                raise UnsupportedSnapshotError(
                    f"{path.name} uses store schema {row[0]}, this build supports "
                    f"{STORE_SCHEMA_VERSION}; refusing to open without a migration"
                )
            else:
                self._migrate(int(row[0]))

    def _migrate(self, version: int) -> None:
        """Upgrade an older store in place, one version per transaction."""
        while version < STORE_SCHEMA_VERSION:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                for sql in STORE_MIGRATIONS[version]:
                    self._db.execute(sql)
                version += 1
                self._db.execute(
                    "UPDATE store_meta SET value = ? WHERE key = 'schema_version'", (str(version),)
                )
                self._db.execute("COMMIT")
            except BaseException:
                self._db.execute("ROLLBACK")
                raise

    @property
    def schema_version(self) -> int:
        with self._lock:
            row = self._db.execute(
                "SELECT value FROM store_meta WHERE key='schema_version'"
            ).fetchone()
        return int(row[0])

    # --- lifecycle --------------------------------------------------------------------------

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    # --- writing ----------------------------------------------------------------------------

    def save(self, state: WorldState, *, label: str, kind: SnapshotKind = "MANUAL") -> SnapshotMeta:
        """Persist ``state`` atomically. Returns only after the transaction has committed."""
        return self._save(state, label=label, kind=kind, timeline_label="", fork_snapshot_id=None)

    def fork(
        self,
        snapshot_id: str,
        *,
        label: str,
        seed_mode: SeedMode = SeedMode.COMMON_EXOGENOUS_RANDOMNESS,
        timeline_id: str | None = None,
    ) -> tuple[WorldState, SnapshotMeta]:
        """Branch a saved snapshot into a new timeline and save the branch point.

        The source snapshot is only read (and verified), never modified.
        """
        source = self.load(snapshot_id)
        branch = fork_world(
            source, new_timeline_id=timeline_id or new_timeline_id(), seed_mode=seed_mode
        )
        meta = self._save(
            branch,
            label=f"branch point: {label}",
            kind="MANUAL",
            timeline_label=label,
            fork_snapshot_id=snapshot_id,
        )
        return branch, meta

    def _save(
        self,
        state: WorldState,
        *,
        label: str,
        kind: SnapshotKind,
        timeline_label: str,
        fork_snapshot_id: str | None,
    ) -> SnapshotMeta:
        payload = canonical_json(state.model_dump(mode="json"))
        meta = SnapshotMeta(
            snapshot_id=f"snap_{uuid.uuid4().hex}",
            world_id=state.world_id,
            timeline_id=state.timeline_id,
            tick=state.sim_tick,
            sim_clock_s=state.sim_clock_s,
            world_status=state.world_status,
            label=label,
            kind=kind,
            schema_version=state.schema_version,
            codec_version=CODEC_VERSION,
            engine_version=state.engine_version,
            profile_set_hash=state.setup.profile_set_manifest_hash,
            payload_sha256=_sha256(payload),
            payload_bytes=len(payload),
            created_utc=_utc_now(),
        )
        with self._lock:
            try:
                self._db.execute("BEGIN IMMEDIATE")
                self._db.execute(
                    "INSERT OR IGNORE INTO worlds(id, event_id, created_utc) VALUES (?, ?, ?)",
                    (state.world_id, state.setup.event_id, meta.created_utc),
                )
                if state.parent_timeline_id is not None:
                    # Parent forked in memory but never saved: keep the lineage anyway.
                    self._db.execute(
                        "INSERT OR IGNORE INTO timelines(id, world_id, parent_timeline_id, label,"
                        " fork_snapshot_id, created_utc) VALUES (?, ?, NULL, '(unsaved)', NULL, ?)",
                        (state.parent_timeline_id, state.world_id, meta.created_utc),
                    )
                self._db.execute(
                    "INSERT OR IGNORE INTO timelines(id, world_id, parent_timeline_id, label,"
                    " fork_snapshot_id, created_utc) VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        state.timeline_id,
                        state.world_id,
                        state.parent_timeline_id,
                        timeline_label or ("main" if state.parent_timeline_id is None else ""),
                        fork_snapshot_id,
                        meta.created_utc,
                    ),
                )
                next_seq = self._db.execute(
                    "SELECT COALESCE(MAX(seq), 0) + 1 FROM snapshots"
                ).fetchone()[0]
                self._db.execute(
                    "INSERT INTO snapshots(id, world_id, timeline_id, tick, sim_clock_s, "
                    "world_status, label, kind, schema_version, codec_version, engine_version, "
                    "profile_set_hash, payload_sha256, payload, created_utc, seq) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        meta.snapshot_id,
                        meta.world_id,
                        meta.timeline_id,
                        meta.tick,
                        meta.sim_clock_s,
                        meta.world_status.value,
                        meta.label,
                        meta.kind,
                        meta.schema_version,
                        meta.codec_version,
                        meta.engine_version,
                        meta.profile_set_hash,
                        meta.payload_sha256,
                        payload,
                        meta.created_utc,
                        next_seq,
                    ),
                )
                self._db.execute("COMMIT")
            except BaseException:
                if self._db.in_transaction:
                    self._db.execute("ROLLBACK")
                raise
        return meta

    # --- reading ----------------------------------------------------------------------------

    def load(self, snapshot_id: str) -> WorldState:
        """Load and fully verify a snapshot. Raises a ``SnapshotError`` subclass on any problem."""
        with self._lock:
            row = self._db.execute(
                "SELECT schema_version, codec_version, engine_version, payload_sha256, payload "
                "FROM snapshots WHERE id = ?",
                (snapshot_id,),
            ).fetchone()
        if row is None:
            raise SnapshotNotFoundError(snapshot_id)
        schema_version, codec_version, engine_version, expected_sha, payload = row

        if (schema_version, codec_version, engine_version) != (
            WORLD_SCHEMA_VERSION,
            CODEC_VERSION,
            ENGINE_VERSION,
        ):
            raise UnsupportedSnapshotError(
                f"{snapshot_id} was written by schema {schema_version} / codec {codec_version} / "
                f"{engine_version}; this build is schema {WORLD_SCHEMA_VERSION} / codec "
                f"{CODEC_VERSION} / {ENGINE_VERSION}. Refusing to load with a different engine."
            )
        if not isinstance(payload, bytes):
            raise SnapshotIntegrityError(
                f"{snapshot_id}: payload has storage type {type(payload).__name__}, not bytes "
                "(corrupted)"
            )
        if _sha256(payload) != expected_sha:
            raise SnapshotIntegrityError(f"{snapshot_id}: payload checksum mismatch (corrupted)")
        try:
            state = WorldState.model_validate_json(payload)
        except ValidationError as exc:
            raise SnapshotIntegrityError(f"{snapshot_id}: payload is not a valid world") from exc
        if state.canonical_hash() != expected_sha:
            raise SnapshotIntegrityError(f"{snapshot_id}: decoded state does not re-encode exactly")
        return state

    def get_meta(self, snapshot_id: str) -> SnapshotMeta:
        with self._lock:
            row = self._db.execute(
                f"SELECT {_META_COLUMNS} FROM snapshots WHERE id = ?", (snapshot_id,)
            ).fetchone()
        if row is None:
            raise SnapshotNotFoundError(snapshot_id)
        return self._meta(row)

    def list_snapshots(self, world_id: str | None = None) -> list[SnapshotMeta]:
        """Snapshots newest first, optionally for one world."""
        query = f"SELECT {_META_COLUMNS} FROM snapshots"
        params: tuple[str, ...] = ()
        if world_id is not None:
            query += " WHERE world_id = ?"
            params = (world_id,)
        with self._lock:
            rows = self._db.execute(query + " ORDER BY seq DESC", params).fetchall()
        return [self._meta(r) for r in rows]

    def latest_valid(self, world_id: str) -> SnapshotMeta | None:
        """Newest fully verified snapshot of ``world_id`` (recovery after corruption)."""
        for meta in self.list_snapshots(world_id):
            try:
                self.load(meta.snapshot_id)
            except SnapshotError:
                continue
            return meta
        return None

    def list_timelines(self, world_id: str) -> list[TimelineMeta]:
        """All timelines of a world, oldest first, with lineage and snapshot counts."""
        with self._lock:
            rows = self._db.execute(
                "SELECT t.id, t.world_id, t.parent_timeline_id, t.label, t.fork_snapshot_id, "
                "t.created_utc, (SELECT COUNT(*) FROM snapshots s WHERE s.timeline_id = t.id) "
                "FROM timelines t WHERE t.world_id = ? ORDER BY t.created_utc, t.rowid",
                (world_id,),
            ).fetchall()
        keys = TimelineMeta.model_fields
        return [TimelineMeta.model_validate(dict(zip(keys, r, strict=True))) for r in rows]

    def delete_snapshot(self, snapshot_id: str) -> None:
        """Delete one snapshot. Branches forked from it keep working: they hold full copies."""
        with self._lock:
            cursor = self._db.execute("DELETE FROM snapshots WHERE id = ?", (snapshot_id,))
        if cursor.rowcount == 0:
            raise SnapshotNotFoundError(snapshot_id)

    @staticmethod
    def _meta(row: tuple[object, ...]) -> SnapshotMeta:
        keys = SnapshotMeta.model_fields
        return SnapshotMeta.model_validate(dict(zip(keys, row, strict=True)))
