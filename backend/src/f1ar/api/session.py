"""The API's live session: at most one loaded world, driven by one ``WorldRunner`` (D-027).

F1AR is a local single-user app, so the backend keeps exactly one *live* world in memory.
Every other world lives only in the save file. Making another world live (create, load, fork)
first shuts the current runner down the safe way (pause, autosave, write logs; plan §4.7
test 8), so two runners never write to the same timeline.

- **Create**: a new world (new id, US-01) is saved once right away ("created"), so it shows
  up in the library even before the first manual save.
- **Continue** (load without a snapshot): the newest valid snapshot of the timeline that was
  last live (``SnapshotStore.continue_point``), so a crash doesn't drop the user on another
  branch.
- **Load**: ``SnapshotStore.resume`` decides whether the timeline continues or a branch is
  created (history is never rewritten, D-025). A loaded world is always PAUSED: if the save
  was taken while running, a logged ``SYSTEM`` PAUSE is applied.
- **Fork**: from the live world, a SAVE command first fixes a canonical tick boundary; from a
  stored world, the given (or newest valid) snapshot is used. ``SnapshotStore.fork`` makes the
  branch, which becomes live; decision overrides are then applied as logged commands.
- **Auto-pause**: when the last stream client of a running live world disconnects, a logged
  ``SYSTEM`` PAUSE is applied (plan §13.4 MVP policy).
"""

import threading
import time
import uuid
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass

from f1ar.api.catalog import RaceCatalog, RaceSpec
from f1ar.domain.race_state import WorldState, WorldStatus
from f1ar.persistence.sqlite_store import (
    SnapshotMeta,
    SnapshotNotFoundError,
    SnapshotStore,
)
from f1ar.simulation import lightweight_engine as engine
from f1ar.simulation.timeline import SeedMode, new_world_id
from f1ar.simulation.world_runner import (
    ALLOWED_SPEEDS,
    AutosavePolicy,
    Command,
    CommandAck,
    CommandType,
    WorldRunner,
)

COMMAND_TIMEOUT_S = 5.0
SYSTEM_ACTOR = "SYSTEM"


class UnknownWorldError(KeyError):
    pass


class WorldNotLiveError(Exception):
    def __init__(self, world_id: str, live_world_id: str | None) -> None:
        super().__init__(f"world {world_id} is not loaded (live: {live_world_id})")
        self.world_id, self.live_world_id = world_id, live_world_id


class CommandTimeoutError(Exception):
    pass


class CommandRejectedError(Exception):
    def __init__(self, ack: CommandAck) -> None:
        super().__init__(ack.reason or "rejected")
        self.ack = ack


@dataclass(frozen=True)
class LiveWorld:
    runner: WorldRunner
    race: RaceSpec
    generation: int
    """Increases every time a runner is replaced (create/load/fork); streams use it."""

    @property
    def world_id(self) -> str:
        return self.runner.state.world_id


@dataclass(frozen=True)
class ForkResult:
    state: WorldState
    branch_point: SnapshotMeta
    source_snapshot_id: str
    acks: tuple[CommandAck, ...]


class LiveSession:
    def __init__(
        self,
        store: SnapshotStore,
        catalog: RaceCatalog,
        *,
        run_background: bool = True,
        clock: Callable[[], float] = time.monotonic,
        autosave: AutosavePolicy | None = None,
        auto_pause_on_disconnect: bool = True,
    ) -> None:
        self.store, self.catalog = store, catalog
        self._background = run_background
        self._clock = clock
        self._autosave = autosave or AutosavePolicy()
        self._auto_pause = auto_pause_on_disconnect
        self._lock = threading.RLock()
        self._live: LiveWorld | None = None
        self._generation = 0
        self._clients: dict[str, int] = {}

    # --- reading ------------------------------------------------------------------------

    @property
    def live(self) -> LiveWorld | None:
        return self._live

    def live_world(self, world_id: str) -> LiveWorld | None:
        live = self._live
        return live if live is not None and live.world_id == world_id else None

    def require_live(self, world_id: str) -> LiveWorld:
        live = self.live_world(world_id)
        if live is None:
            self.require_known(world_id)
            raise WorldNotLiveError(world_id, None if self._live is None else self._live.world_id)
        return live

    def require_known(self, world_id: str) -> None:
        if self.live_world(world_id) is None and not self.store.list_snapshots(world_id):
            raise UnknownWorldError(world_id)

    def current_state(self, world_id: str) -> WorldState:
        """The live state, or the newest valid saved state of a stored world."""
        live = self.live_world(world_id)
        if live is not None:
            return live.runner.state
        meta = self.store.continue_point(world_id)
        if meta is None:
            raise UnknownWorldError(world_id)
        return self.store.load(meta.snapshot_id)

    # --- lifecycle ----------------------------------------------------------------------

    def create(
        self,
        race_id: str,
        *,
        seed: int,
        codes: Sequence[str] | None = None,
        synthetic: bool = False,
        speed: int = 1,
    ) -> WorldState:
        race = self.catalog.get(race_id)
        setup = race.setup(codes, synthetic=synthetic)
        with self._lock, self._replacing_live():
            state = engine.create_world(setup, seed=seed, world_id=new_world_id("world"))
            self.store.save(state, label="created")
            self._activate(state, race, speed)
            return state

    def load(
        self, world_id: str, snapshot_id: str | None = None, *, speed: int | None = None
    ) -> tuple[WorldState, SnapshotMeta]:
        """Make a saved world live. Returns (live state, loaded snapshot)."""
        with self._lock:
            previous_speed = self._live.runner.speed if self._live is not None else 1
            if snapshot_id is not None:  # verify before touching the live world
                self._verified(world_id, snapshot_id)
            else:
                self.require_known(world_id)
            with self._replacing_live():
                # "Continue" (no id): the newest point *after* the live runner's shutdown
                # autosave, so continuing a live world doesn't create a needless branch.
                meta, race = self._verified(world_id, snapshot_id)
                state = self.store.resume(meta.snapshot_id)
                live = self._activate(state, race, speed or previous_speed)
            if state.world_status is WorldStatus.RUNNING:
                self._submit(live, self._system(CommandType.PAUSE, "load-pause"))
            return live.runner.state, meta

    def fork(
        self,
        world_id: str,
        *,
        name: str,
        source_snapshot_id: str | None = None,
        seed_mode: SeedMode = SeedMode.COMMON_EXOGENOUS_RANDOMNESS,
        overrides: Sequence[Command] = (),
        speed: int | None = None,
    ) -> ForkResult:
        with self._lock:
            live = self.live_world(world_id)
            if source_snapshot_id is None and live is not None:
                ack = self._submit(
                    live,
                    Command(
                        idempotency_key=f"fork-point-{uuid.uuid4().hex}",
                        type=CommandType.SAVE,
                        label=f"fork point: {name}",
                    ),
                )
                if not ack.accepted or ack.snapshot_id is None:
                    raise CommandRejectedError(ack)
                source_snapshot_id = ack.snapshot_id
            source, race = self._verified(world_id, source_snapshot_id)
            previous_speed = self._live.runner.speed if self._live is not None else 1
            with self._replacing_live():
                branch, branch_meta = self.store.fork(
                    source.snapshot_id, label=name, seed_mode=seed_mode
                )
                new_live = self._activate(branch, race, speed or previous_speed)
            acks = tuple(self._submit(new_live, c) for c in overrides)
            return ForkResult(new_live.runner.state, branch_meta, source.snapshot_id, acks)

    def command(self, world_id: str, command: Command) -> CommandAck:
        with self._lock:
            live = self.require_live(world_id)
        return self._submit(live, command)

    def shutdown(self) -> None:
        with self._lock:
            if self._live is not None:
                self._live.runner.shutdown()
                self._live = None

    # --- stream clients -------------------------------------------------------------------

    def client_connected(self, world_id: str) -> None:
        with self._lock:
            self._clients[world_id] = self._clients.get(world_id, 0) + 1

    def client_disconnected(self, world_id: str) -> CommandAck | None:
        """Returns the auto-pause ack if the disconnect paused a running world."""
        with self._lock:
            left = max(0, self._clients.get(world_id, 0) - 1)
            self._clients[world_id] = left
            live = self.live_world(world_id)
            if (
                left > 0
                or not self._auto_pause
                or live is None
                or live.runner.state.world_status is not WorldStatus.RUNNING
            ):
                return None
        return self._submit(live, self._system(CommandType.PAUSE, "auto-pause"))

    def clients(self, world_id: str) -> int:
        return self._clients.get(world_id, 0)

    # --- internals ----------------------------------------------------------------------

    @staticmethod
    def _system(kind: CommandType, reason: str) -> Command:
        return Command(
            idempotency_key=f"{reason}-{uuid.uuid4().hex}",
            type=kind,
            actor_id=SYSTEM_ACTOR,
            label=reason,
        )

    def _verified(self, world_id: str, snapshot_id: str | None) -> tuple[SnapshotMeta, RaceSpec]:
        """The snapshot (or the newest valid one) of ``world_id``, fully verified, and its race."""
        meta = self._snapshot_of(world_id, snapshot_id)
        return meta, self.catalog.get(self.store.load(meta.snapshot_id).setup.event_id)

    def _snapshot_of(self, world_id: str, snapshot_id: str | None) -> SnapshotMeta:
        if snapshot_id is None:
            meta = self.store.continue_point(world_id)
            if meta is None:
                raise UnknownWorldError(world_id)
            return meta
        meta = self.store.get_meta(snapshot_id)
        if meta.world_id != world_id:
            raise SnapshotNotFoundError(f"{snapshot_id} does not belong to world {world_id}")
        return meta

    def _activate(self, state: WorldState, race: RaceSpec, speed: int) -> LiveWorld:
        if speed not in ALLOWED_SPEEDS:
            raise ValueError(f"speed must be one of {ALLOWED_SPEEDS}")
        runner = WorldRunner(
            state, speed=speed, clock=self._clock, store=self.store, autosave=self._autosave
        )
        if self._background:
            runner.start_background()
        # "continue" must resume where the user is, even after a crash (store v4)
        self.store.set_active_timeline(state.world_id, state.timeline_id)
        self._generation += 1
        self._live = LiveWorld(runner, race, self._generation)
        return self._live

    @contextmanager
    def _replacing_live(self) -> Iterator[None]:
        """Shut the live runner down (pause, autosave, logs) but keep it visible, read-only,
        until the body activates its replacement, so a stream never sees "no live world" in
        the middle of a switch. If the switch fails, there is no live world."""
        old = self._live
        if old is not None:
            old.runner.shutdown()
        try:
            yield
        except BaseException:
            if self._live is old:
                self._live = None
            raise

    def _submit(self, live: LiveWorld, command: Command) -> CommandAck:
        future = live.runner.submit(command)
        if not self._background:
            live.runner.pump()  # tests / no thread: apply right away
        try:
            return future.result(timeout=COMMAND_TIMEOUT_S)
        except TimeoutError as exc:
            raise CommandTimeoutError(f"runner did not acknowledge {command.type}") from exc
