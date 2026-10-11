"""WorldRunner: owns one live world, applies commands at tick boundaries, paces ticks (plan §12.4).

Two concerns are kept strictly apart:

- **What happens** is decided only by the pure engine and the ordered commands. Every
  command is applied *between* ticks, never inside one, so a paused or saved world is always
  a consistent canonical state.
- **When it happens** is decided by wall-clock pacing. Real time only decides *how many*
  ticks are due (``speed`` x real seconds / tick length). It never changes what a tick
  computes, so the same command sequence gives the same states at any speed.

Pausing freezes the world completely. On resume the pacing anchor is reset, so time spent
paused is never "caught up" (plan §4.7 test 4).

``pump(now)`` is the deterministic scheduling core and can be driven by a fake clock in
tests. ``start_background()`` runs it in a thread against ``time.monotonic``.
"""

import math
import queue
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from f1ar.domain.race_state import RaceEvent, WorldState, WorldStatus
from f1ar.simulation import lightweight_engine as engine

ALLOWED_SPEEDS: tuple[int, ...] = (1, 2, 5, 10, 25)
MAX_STEP_TICKS = 10_000


class CommandType(StrEnum):
    START = "START"
    PAUSE = "PAUSE"
    RESUME = "RESUME"
    SET_SPEED = "SET_SPEED"
    STEP = "STEP"
    """Advance N ticks manually; only while PAUSED."""


class Command(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    idempotency_key: str = Field(min_length=1)
    """Client-chosen unique key; a repeated key is acknowledged again but never re-applied."""
    type: CommandType
    actor_id: str = "USER"
    speed: int | None = None
    ticks: int = Field(default=1, ge=1, le=MAX_STEP_TICKS)


class CommandAck(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    idempotency_key: str
    type: CommandType
    accepted: bool
    reason: str | None
    applied_tick: int
    """Canonical tick at which the command took effect (or was rejected)."""
    world_status: WorldStatus
    speed: int
    duplicate: bool = False


class AppliedCommand(BaseModel):
    """Command log record: what was asked, and what the world did with it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    command: Command
    ack: CommandAck


Listener = Callable[[WorldState, tuple[RaceEvent, ...]], None]


class _Rejected(Exception):
    pass


class WorldRunner:
    def __init__(
        self,
        state: WorldState,
        *,
        speed: int = 1,
        clock: Callable[[], float] = time.monotonic,
        max_catchup_ticks: int = 1_000,
    ) -> None:
        if speed not in ALLOWED_SPEEDS:
            raise ValueError(f"speed must be one of {ALLOWED_SPEEDS}")
        self._state = state
        self._speed = speed
        self._clock = clock
        self._max_catchup = max_catchup_ticks

        self._lock = threading.RLock()
        self._queue: queue.SimpleQueue[tuple[Command, Future[CommandAck]]] = queue.SimpleQueue()
        self._acks: dict[str, CommandAck] = {}
        self._log: list[AppliedCommand] = []
        self._events: list[RaceEvent] = []
        self._listeners: list[Listener] = []

        # Pacing anchor: tick ``_anchor_tick`` corresponds to wall time ``_anchor_wall``.
        self._anchor_wall = 0.0
        self._anchor_tick = state.sim_tick
        if state.world_status is WorldStatus.RUNNING:
            self._anchor_wall = clock()

        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # --- read-only views (states and events are immutable, so sharing references is safe) ---

    @property
    def state(self) -> WorldState:
        with self._lock:
            return self._state

    @property
    def speed(self) -> int:
        with self._lock:
            return self._speed

    @property
    def events(self) -> tuple[RaceEvent, ...]:
        with self._lock:
            return tuple(self._events)

    @property
    def command_log(self) -> tuple[AppliedCommand, ...]:
        with self._lock:
            return tuple(self._log)

    def add_listener(self, listener: Listener) -> None:
        """Called after every tick with (new_state, events), on the runner's thread.

        Listeners must be quick and must not block waiting on a command result.
        """
        with self._lock:
            self._listeners.append(listener)

    # --- commands ---------------------------------------------------------------------------

    def submit(self, command: Command) -> Future[CommandAck]:
        """Queue a command; it is applied at the next tick boundary. Thread-safe."""
        future: Future[CommandAck] = Future()
        self._queue.put((command, future))
        self._wake.set()
        return future

    # --- deterministic scheduling core ------------------------------------------------------

    def pump(self, now: float | None = None) -> None:
        """Apply queued commands, then run every tick that is due at wall time ``now``.

        Commands that arrive while ticks are running are applied between ticks, so a pause
        takes effect within one tick.
        """
        now = self._clock() if now is None else now
        with self._lock:
            self._drain(now)
            ran = 0
            while (
                self._state.world_status is WorldStatus.RUNNING
                and self._state.sim_tick < self._due_tick(now)
            ):
                if ran >= self._max_catchup:
                    # Can't keep up: slow the simulation down instead of jumping ahead.
                    self._reanchor(now)
                    break
                self._advance(allow_paused=False)
                ran += 1
                self._drain(now)

    def _due_tick(self, now: float) -> int:
        elapsed_sim = (now - self._anchor_wall) * self._speed
        return self._anchor_tick + math.floor(elapsed_sim / self._state.setup.tick_seconds + 1e-9)

    def _reanchor(self, now: float) -> None:
        self._anchor_wall = now
        self._anchor_tick = self._state.sim_tick

    def _advance(self, *, allow_paused: bool) -> None:
        result = engine.step(self._state, allow_paused=allow_paused)
        self._state = result.state
        self._events.extend(result.events)
        for listener in self._listeners:
            listener(result.state, result.events)

    def _drain(self, now: float) -> None:
        while True:
            try:
                command, future = self._queue.get_nowait()
            except queue.Empty:
                return
            future.set_result(self._apply(command, now))

    def _apply(self, command: Command, now: float) -> CommandAck:
        previous = self._acks.get(command.idempotency_key)
        if previous is not None:
            return previous.model_copy(update={"duplicate": True})

        try:
            self._execute(command, now)
            accepted, reason = True, None
        except (_Rejected, engine.EngineStateError) as exc:
            accepted, reason = False, str(exc)

        ack = CommandAck(
            idempotency_key=command.idempotency_key,
            type=command.type,
            accepted=accepted,
            reason=reason,
            applied_tick=self._state.sim_tick,
            world_status=self._state.world_status,
            speed=self._speed,
        )
        self._acks[command.idempotency_key] = ack
        self._log.append(AppliedCommand(command=command, ack=ack))
        return ack

    def _execute(self, command: Command, now: float) -> None:
        match command.type:
            case CommandType.START:
                self._state = engine.start(self._state)
                self._reanchor(now)
            case CommandType.PAUSE:
                self._state = engine.pause(self._state)
            case CommandType.RESUME:
                self._state = engine.resume(self._state)
                self._reanchor(now)  # paused wall time is never caught up
            case CommandType.SET_SPEED:
                if command.speed not in ALLOWED_SPEEDS:
                    raise _Rejected(f"speed must be one of {ALLOWED_SPEEDS}, got {command.speed}")
                self._speed = command.speed
                self._reanchor(now)
            case CommandType.STEP:
                if self._state.world_status is not WorldStatus.PAUSED:
                    raise _Rejected("manual stepping requires a PAUSED world")
                for _ in range(command.ticks):
                    if self._state.world_status is not WorldStatus.PAUSED:
                        break  # race completed during stepping
                    self._advance(allow_paused=True)

    # --- background thread --------------------------------------------------------------------

    def start_background(self) -> None:
        if self._thread is not None:
            raise RuntimeError("runner thread already started")
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="world-runner", daemon=True)
        self._thread.start()

    def stop_background(self, timeout: float = 5.0) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout)
            self._thread = None

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._wake.clear()
            self.pump()
            self._wake.wait(self._seconds_until_next_tick())

    def _seconds_until_next_tick(self) -> float | None:
        with self._lock:
            if self._state.world_status is not WorldStatus.RUNNING:
                return None  # sleep until a command arrives
            next_tick = self._state.sim_tick + 1
            due_at = (
                self._anchor_wall
                + (next_tick - self._anchor_tick) * self._state.setup.tick_seconds / self._speed
            )
            return max(0.0, due_at - self._clock())
