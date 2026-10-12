"""The ``RaceEngine`` port (plan §7.3) and its implementation on the lightweight engine.

The world runner, the debug viewer and the tests use the pure functions of
``lightweight_engine`` directly. This port is the stable contract for everything that wants an
engine as an object: the M5 Gymnasium environment (observation, action mask, bounded step) and
any future alternative engine (plan §7.2: an engine that can't satisfy this contract, including
``get_state``/``set_state``, is not an option).

``step(max_sim_delta_seconds)`` is bounded: it advances whole ticks only, never more than the
requested sim time (at least one tick), and stops at the end of the race.
"""

import math
from collections.abc import Sequence
from typing import Protocol

from f1ar.domain.race_state import DriverAction, RaceEvent, RaceSetup, WorldState, WorldStatus
from f1ar.policy.observation import Observation, build_observation
from f1ar.simulation import lightweight_engine as engine


class RaceEngine(Protocol):
    def initialize(self, setup: RaceSetup, *, seed: int, world_id: str) -> WorldState: ...
    def get_state(self) -> WorldState: ...
    def set_state(self, state: WorldState) -> None: ...
    def get_observation(self, driver_id: str) -> Observation: ...
    def get_valid_actions(self, driver_id: str) -> engine.ValidActions: ...
    def apply_commands(self, actions: Sequence[DriverAction]) -> tuple[RaceEvent, ...]: ...
    def step(self, max_sim_delta_seconds: float) -> tuple[RaceEvent, ...]: ...
    def get_events(self, since_seq: int = 0) -> tuple[RaceEvent, ...]: ...
    def get_positions(self) -> tuple[tuple[str, int, float], ...]: ...
    def is_terminal(self) -> bool: ...


class LightweightRaceEngine:
    """``RaceEngine`` over the pure lightweight engine. Holds one current state and the events
    emitted since it was initialized or set (it adds no behaviour of its own)."""

    def __init__(self, state: WorldState | None = None) -> None:
        self._state = state
        self._events: list[RaceEvent] = []

    def _current(self) -> WorldState:
        if self._state is None:
            raise engine.EngineStateError("initialize() or set_state() first")
        return self._state

    def initialize(self, setup: RaceSetup, *, seed: int, world_id: str) -> WorldState:
        self._state = engine.start(engine.create_world(setup, seed=seed, world_id=world_id))
        self._events = []
        return self._state

    def get_state(self) -> WorldState:
        return self._current()

    def set_state(self, state: WorldState) -> None:
        if state.engine_version != engine.ENGINE_VERSION:
            raise engine.EngineStateError(
                f"state from {state.engine_version}, this is {engine.ENGINE_VERSION}"
            )
        if state.setup.policy_manifest != engine.POLICY_MANIFEST:
            raise engine.EngineStateError(f"state pins policies {state.setup.policy_manifest}")
        self._state = state
        self._events = []

    def get_observation(self, driver_id: str) -> Observation:
        return build_observation(self._current(), driver_id)

    def get_valid_actions(self, driver_id: str) -> engine.ValidActions:
        return engine.valid_actions(self._current(), driver_id)

    def apply_commands(self, actions: Sequence[DriverAction]) -> tuple[RaceEvent, ...]:
        """Apply actions in order; an illegal one raises ``ActionRejectedError`` and leaves the
        state as it was after the previous (legal) ones."""
        out: list[RaceEvent] = []
        for action in actions:
            result = engine.apply_action(self._current(), action)
            self._state = result.state
            out += result.events
        self._events += out
        return tuple(out)

    def step(self, max_sim_delta_seconds: float) -> tuple[RaceEvent, ...]:
        state = self._current()
        if max_sim_delta_seconds <= 0:
            raise ValueError("max_sim_delta_seconds must be positive")
        ticks = max(1, math.floor(max_sim_delta_seconds / state.setup.tick_seconds + 1e-9))
        out: list[RaceEvent] = []
        for _ in range(ticks):
            if state.world_status is not WorldStatus.RUNNING:
                break
            result = engine.step(state)
            state = result.state
            out += result.events
        self._state = state
        self._events += out
        return tuple(out)

    def get_events(self, since_seq: int = 0) -> tuple[RaceEvent, ...]:
        return tuple(e for e in self._events if e.seq >= since_seq)

    def get_positions(self) -> tuple[tuple[str, int, float], ...]:
        """(driver, completed laps, distance into the lap) in race order."""
        state = self._current()
        return tuple(
            (d, state.car(d).laps_completed, state.car(d).distance_m)
            for d in engine.standings(state)
        )

    def is_terminal(self) -> bool:
        return self._current().world_status is WorldStatus.COMPLETED
