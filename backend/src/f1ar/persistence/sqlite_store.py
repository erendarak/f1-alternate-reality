"""SQLite snapshot store (plan §4.5, §4.6, §13.1).

A snapshot is the canonical JSON encoding of a complete ``WorldState`` plus its SHA-256,
written together with its metadata in ONE transaction. A crash can't leave a half-written
snapshot that looks valid. Every load re-verifies the checksum, the schema version and the
engine version; nothing is silently migrated or loaded with a different engine.

Snapshots are immutable: they are only ever inserted, never updated. Timelines record
their lineage (parent timeline + the snapshot they were forked from), so the history of
alternate realities can be shown even after a source snapshot is deleted.

**Event and command logs** (store v3, plan §4.5, §13.1): every race event and every command
(with its acknowledgement) is stored per timeline. A save writes the snapshot *and* the log
records up to it in the same transaction, so a snapshot's ``event_seq`` is always backed by its
events. A branch's full history is its parent's events before the branch point plus its own.
Stored command keys keep commands idempotent across restarts (plan §4.7 test 9).

**History is never rewritten:** resuming from a snapshot that isn't the latest point of its
timeline creates a new branch instead of continuing (and overwriting) that timeline's future.

Store schema history (``STORE_MIGRATIONS``): v1 = worlds/timelines/snapshots;
v2 = timelines gain ``label`` and ``fork_snapshot_id``; v3 = events and commands tables,
snapshots gain ``event_seq``, ``policy_manifest`` and ``dataset_hash``, timelines gain
``fork_event_seq``.
"""

import hashlib
import json
import sqlite3
import threading
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, ValidationError

from f1ar.domain.race_state import WORLD_SCHEMA_VERSION, RaceEvent, WorldState, WorldStatus
from f1ar.persistence.snapshot_codec import CODEC_VERSION, HASH_PREFIX, canonical_json
from f1ar.simulation.lightweight_engine import ENGINE_VERSION, POLICY_MANIFEST
from f1ar.simulation.timeline import SeedMode, fork_world, new_timeline_id

STORE_SCHEMA_VERSION = 3

STORE_MIGRATIONS: dict[int, tuple[str, ...]] = {
    # from version N to N+1
    1: (
        "ALTER TABLE timelines ADD COLUMN label TEXT NOT NULL DEFAULT ''",
        "ALTER TABLE timelines ADD COLUMN fork_snapshot_id TEXT",
    ),
    2: (
        "ALTER TABLE timelines ADD COLUMN fork_event_seq INTEGER",
        "ALTER TABLE snapshots ADD COLUMN event_seq INTEGER NOT NULL DEFAULT 0",
        "ALTER TABLE snapshots ADD COLUMN policy_manifest TEXT NOT NULL DEFAULT '{}'",
        "ALTER TABLE snapshots ADD COLUMN dataset_hash TEXT NOT NULL DEFAULT ''",
        "ALTER TABLE snapshots ADD COLUMN command_seq INTEGER NOT NULL DEFAULT 0",
        "ALTER TABLE timelines ADD COLUMN fork_command_seq INTEGER",
        # events / commands tables are created by the DDL (CREATE TABLE IF NOT EXISTS)
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
    fork_snapshot_id   TEXT,  -- provenance only, deliberately not a foreign key
    fork_event_seq     INTEGER,  -- parent events before this belong to this branch's history
    fork_command_seq   INTEGER   -- parent commands before this were applied before the branch
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
    seq               INTEGER NOT NULL UNIQUE,
    event_seq         INTEGER NOT NULL DEFAULT 0,
    policy_manifest   TEXT NOT NULL DEFAULT '{}',
    dataset_hash      TEXT NOT NULL DEFAULT '',
    command_seq       INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS snapshots_by_world ON snapshots(world_id, seq);
CREATE TABLE IF NOT EXISTS events (
    timeline_id  TEXT NOT NULL REFERENCES timelines(id),
    seq          INTEGER NOT NULL,
    tick         INTEGER NOT NULL,
    type         TEXT NOT NULL,
    driver_id    TEXT,
    payload_json TEXT NOT NULL,
    PRIMARY KEY (timeline_id, seq)
);
CREATE TABLE IF NOT EXISTS commands (
    timeline_id     TEXT NOT NULL REFERENCES timelines(id),
    idempotency_key TEXT NOT NULL,
    log_seq         INTEGER NOT NULL,
    type            TEXT NOT NULL,
    actor_id        TEXT NOT NULL,
    applied_tick    INTEGER NOT NULL,
    accepted        INTEGER NOT NULL,
    command_json    TEXT NOT NULL,
    ack_json        TEXT NOT NULL,
    created_utc     TEXT NOT NULL,
    PRIMARY KEY (timeline_id, idempotency_key)
);
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
    event_seq: int
    policy_manifest: str
    dataset_hash: str
    command_seq: int
    """Command-log cursor: commands of this timeline with ``log_seq`` below it were applied
    before this snapshot (plan §4.4 ``command_cursor``)."""


@dataclass(frozen=True)
class CommandRecord:
    """One command and the world's answer, as stored in the command log."""

    idempotency_key: str
    log_seq: int
    """Position in the timeline's command log (order of arrival)."""
    type: str
    actor_id: str
    applied_tick: int
    accepted: bool
    command_json: str
    ack_json: str


class TimelineMeta(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    timeline_id: str
    world_id: str
    parent_timeline_id: str | None
    label: str
    fork_snapshot_id: str | None
    created_utc: str
    snapshot_count: int
    fork_event_seq: int | None
    event_count: int
    command_count: int


_META_COLUMNS = (
    "id, world_id, timeline_id, tick, sim_clock_s, world_status, label, kind, schema_version, "
    "codec_version, engine_version, profile_set_hash, payload_sha256, length(payload), "
    "created_utc, "
    "event_seq, policy_manifest, dataset_hash, command_seq"
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

    def save(
        self,
        state: WorldState,
        *,
        label: str,
        kind: SnapshotKind = "MANUAL",
        events: Sequence[RaceEvent] = (),
        commands: Sequence[CommandRecord] = (),
    ) -> SnapshotMeta:
        """Persist ``state`` atomically, together with any not-yet-stored log records up to it.

        Returns only after the transaction has committed.
        """
        return self._save(
            state,
            label=label,
            kind=kind,
            timeline_label="",
            fork_snapshot_id=None,
            events=events,
            commands=commands,
        )

    def append_log(
        self,
        state: WorldState,
        *,
        events: Sequence[RaceEvent] = (),
        commands: Sequence[CommandRecord] = (),
    ) -> None:
        """Store log records of ``state``'s timeline (idempotent: known records are skipped)."""
        if not events and not commands:
            return
        with self._lock:
            try:
                self._db.execute("BEGIN IMMEDIATE")
                self._ensure_timeline(state, created=_utc_now(), label="", fork_snapshot_id=None)
                self._insert_log(state.timeline_id, events, commands)
                self._db.execute("COMMIT")
            except BaseException:
                if self._db.in_transaction:
                    self._db.execute("ROLLBACK")
                raise

    def _ensure_timeline(
        self,
        state: WorldState,
        *,
        created: str,
        label: str,
        fork_snapshot_id: str | None,
        fork_command_seq: int | None = None,
    ) -> None:
        self._db.execute(
            "INSERT OR IGNORE INTO worlds(id, event_id, created_utc) VALUES (?, ?, ?)",
            (state.world_id, state.setup.event_id, created),
        )
        if state.parent_timeline_id is not None:
            # Parent forked in memory but never saved: keep the lineage anyway.
            self._db.execute(
                "INSERT OR IGNORE INTO timelines(id, world_id, parent_timeline_id, label,"
                " fork_snapshot_id, created_utc) VALUES (?, ?, NULL, '(unsaved)', NULL, ?)",
                (state.parent_timeline_id, state.world_id, created),
            )
        owner = self._db.execute(
            "SELECT world_id FROM timelines WHERE id = ?", (state.timeline_id,)
        ).fetchone()
        if owner is not None and owner[0] != state.world_id:
            raise SnapshotIntegrityError(
                f"timeline {state.timeline_id} belongs to world {owner[0]}, not {state.world_id}"
            )
        self._db.execute(
            "INSERT OR IGNORE INTO timelines(id, world_id, parent_timeline_id, label,"
            " fork_snapshot_id, fork_event_seq, fork_command_seq, created_utc)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                state.timeline_id,
                state.world_id,
                state.parent_timeline_id,
                label or ("main" if state.parent_timeline_id is None else ""),
                fork_snapshot_id,
                state.branch_event_seq,
                fork_command_seq,
                created,
            ),
        )

    def _insert_log(
        self,
        timeline_id: str,
        events: Sequence[RaceEvent],
        commands: Sequence[CommandRecord],
    ) -> None:
        rows = [
            (timeline_id, e.seq, e.tick, e.type, e.driver_id, json.dumps(e.payload)) for e in events
        ]
        if rows:
            # Writing the same record twice is a no-op; a *different* record under an existing
            # (timeline, seq) means two histories are being mixed (e.g. a reused world id).
            existing = {
                r[0]: r[1:]
                for r in self._db.execute(
                    "SELECT seq, tick, type, driver_id, payload_json FROM events "
                    "WHERE timeline_id = ? AND seq BETWEEN ? AND ?",
                    (timeline_id, min(r[1] for r in rows), max(r[1] for r in rows)),
                ).fetchall()
            }
            for row in rows:
                if row[1] in existing and tuple(existing[row[1]]) != row[2:]:
                    raise SnapshotIntegrityError(
                        f"event {row[1]} of {timeline_id} is already stored with different "
                        "content: two histories under one timeline id (world id reused?)"
                    )
            self._db.executemany(
                "INSERT OR IGNORE INTO events(timeline_id, seq, tick, type, driver_id, "
                "payload_json) VALUES (?, ?, ?, ?, ?, ?)",
                rows,
            )
        now = _utc_now()
        self._db.executemany(
            "INSERT OR IGNORE INTO commands(timeline_id, idempotency_key, log_seq, type, actor_id,"
            " applied_tick, accepted, command_json, ack_json, created_utc)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    timeline_id,
                    c.idempotency_key,
                    c.log_seq,
                    c.type,
                    c.actor_id,
                    c.applied_tick,
                    int(c.accepted),
                    c.command_json,
                    c.ack_json,
                    now,
                )
                for c in commands
            ],
        )

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
        source_meta = self.get_meta(snapshot_id)
        branch = fork_world(
            source, new_timeline_id=timeline_id or new_timeline_id(), seed_mode=seed_mode
        )
        meta = self._save(
            branch,
            label=f"branch point: {label}",
            kind="MANUAL",
            timeline_label=label,
            fork_snapshot_id=snapshot_id,
            fork_command_seq=source_meta.command_seq,
        )
        return branch, meta

    def resume(self, snapshot_id: str) -> WorldState:
        """The world to continue from ``snapshot_id`` without ever rewriting history.

        If the snapshot is the latest point of its timeline (no later snapshot, no later event),
        that timeline simply continues. Otherwise a new branch is created at the snapshot
        (same luck), so the timeline's recorded future stays intact.
        """
        state = self.load(snapshot_id)
        meta = self.get_meta(snapshot_id)
        with self._lock:
            later_events = self._db.execute(
                "SELECT COUNT(*) FROM events WHERE timeline_id = ? AND seq >= ?",
                (state.timeline_id, state.event_seq),
            ).fetchone()[0]
            later_snapshots = self._db.execute(
                "SELECT COUNT(*) FROM snapshots WHERE timeline_id = ? AND tick > ?",
                (state.timeline_id, state.sim_tick),
            ).fetchone()[0]
            later_commands = self._db.execute(
                "SELECT COUNT(*) FROM commands WHERE timeline_id = ? AND log_seq >= ?",
                (state.timeline_id, meta.command_seq),
            ).fetchone()[0]
        if later_events == 0 and later_snapshots == 0 and later_commands == 0:
            return state
        branch, _ = self.fork(snapshot_id, label=f"resumed from: {meta.label}")
        return branch

    def _save(
        self,
        state: WorldState,
        *,
        label: str,
        kind: SnapshotKind,
        timeline_label: str,
        fork_snapshot_id: str | None,
        events: Sequence[RaceEvent] = (),
        commands: Sequence[CommandRecord] = (),
        fork_command_seq: int | None = None,
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
            event_seq=state.event_seq,
            policy_manifest=json.dumps(state.setup.policy_manifest, sort_keys=True),
            dataset_hash=state.setup.inputs.get("dataset_sha256", ""),
            command_seq=0,  # set inside the transaction, once the commands are stored
        )
        with self._lock:
            try:
                self._db.execute("BEGIN IMMEDIATE")
                self._ensure_timeline(
                    state,
                    created=meta.created_utc,
                    label=timeline_label,
                    fork_snapshot_id=fork_snapshot_id,
                    fork_command_seq=fork_command_seq,
                )
                self._insert_log(state.timeline_id, events, commands)
                command_seq = self._db.execute(
                    "SELECT COALESCE(MAX(log_seq) + 1, 0) FROM commands WHERE timeline_id = ?",
                    (state.timeline_id,),
                ).fetchone()[0]
                meta = meta.model_copy(update={"command_seq": command_seq})
                next_seq = self._db.execute(
                    "SELECT COALESCE(MAX(seq), 0) + 1 FROM snapshots"
                ).fetchone()[0]
                self._db.execute(
                    "INSERT INTO snapshots(id, world_id, timeline_id, tick, sim_clock_s, "
                    "world_status, label, kind, schema_version, codec_version, engine_version, "
                    "profile_set_hash, payload_sha256, payload, created_utc, seq, event_seq, "
                    "policy_manifest, dataset_hash, command_seq) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
                        meta.event_seq,
                        meta.policy_manifest,
                        meta.dataset_hash,
                        meta.command_seq,
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
        if state.setup.policy_manifest != POLICY_MANIFEST:
            raise UnsupportedSnapshotError(
                f"{snapshot_id} was decided by policies {state.setup.policy_manifest}; this build "
                f"runs {POLICY_MANIFEST}. Refusing to continue it with different decision-makers."
            )
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

    def events(self, timeline_id: str) -> list[RaceEvent]:
        """The full event history of a timeline, in order: its ancestors' events before each
        branch point, then its own events."""
        segments: list[tuple[str, int | None]] = []  # (timeline, events strictly before this)
        with self._lock:
            current: str | None = timeline_id
            limit: int | None = None
            while current is not None:
                row = self._db.execute(
                    "SELECT parent_timeline_id, fork_event_seq FROM timelines WHERE id = ?",
                    (current,),
                ).fetchone()
                segments.append((current, limit))
                if row is None or row[0] is None or row[1] is None:
                    break
                current, limit = row[0], row[1]
            rows: list[tuple[int, int, str, str | None, str]] = []
            for tid, before in segments:
                query = (
                    "SELECT seq, tick, type, driver_id, payload_json FROM events "
                    "WHERE timeline_id = ?"
                )
                params: tuple[str | int, ...] = (tid,)
                if before is not None:
                    query += " AND seq < ?"
                    params = (tid, before)
                rows += self._db.execute(query, params).fetchall()
        rows.sort()
        return [
            RaceEvent.model_validate(
                {
                    "seq": seq,
                    "tick": tick,
                    "type": type_,
                    "driver_id": driver_id,
                    "payload": json.loads(payload),
                }
            )
            for seq, tick, type_, driver_id, payload in rows
        ]

    def commands(self, timeline_id: str, *, with_ancestors: bool = False) -> list[CommandRecord]:
        """The command log of one timeline, in arrival order. With ``with_ancestors``, first the
        commands its ancestors applied before each branch point (for idempotency after a resume
        into a branch: a re-sent command that already shaped this history isn't applied again)."""
        segments: list[tuple[str, int | None]] = [(timeline_id, None)]
        with self._lock:
            if with_ancestors:
                current = timeline_id
                while True:
                    row = self._db.execute(
                        "SELECT parent_timeline_id, fork_command_seq FROM timelines WHERE id = ?",
                        (current,),
                    ).fetchone()
                    if row is None or row[0] is None or row[1] is None:
                        break
                    current = row[0]
                    segments.insert(0, (current, row[1]))
            rows = []
            for tid, before in segments:
                query = (
                    "SELECT idempotency_key, log_seq, type, actor_id, applied_tick, accepted, "
                    "command_json, ack_json FROM commands WHERE timeline_id = ?"
                )
                params: tuple[str | int, ...] = (tid,)
                if before is not None:
                    query += " AND log_seq < ?"
                    params = (tid, before)
                rows += self._db.execute(query + " ORDER BY log_seq", params).fetchall()
        return [
            CommandRecord(
                idempotency_key=r[0],
                log_seq=r[1],
                type=r[2],
                actor_id=r[3],
                applied_tick=r[4],
                accepted=bool(r[5]),
                command_json=r[6],
                ack_json=r[7],
            )
            for r in rows
        ]

    def prune_autosaves(self, timeline_id: str, *, keep: int) -> int:
        """Delete all but the newest ``keep`` AUTOSAVE snapshots of a timeline (manual saves are
        never pruned). Returns how many were deleted."""
        with self._lock:
            cursor = self._db.execute(
                "DELETE FROM snapshots WHERE id IN (SELECT id FROM snapshots WHERE timeline_id = ?"
                " AND kind = 'AUTOSAVE' ORDER BY seq DESC LIMIT -1 OFFSET ?)",
                (timeline_id, keep),
            )
        return cursor.rowcount

    def list_timelines(self, world_id: str) -> list[TimelineMeta]:
        """All timelines of a world, oldest first, with lineage, snapshot and log counts."""
        with self._lock:
            rows = self._db.execute(
                "SELECT t.id, t.world_id, t.parent_timeline_id, t.label, t.fork_snapshot_id, "
                "t.created_utc, (SELECT COUNT(*) FROM snapshots s WHERE s.timeline_id = t.id), "
                "t.fork_event_seq, "
                "(SELECT COUNT(*) FROM events e WHERE e.timeline_id = t.id), "
                "(SELECT COUNT(*) FROM commands c WHERE c.timeline_id = t.id) "
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
