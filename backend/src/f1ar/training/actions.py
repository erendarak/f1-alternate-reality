"""The strategist's 54 actions and the engine commands that carry them out (M5, D-038, D-048).

An action is a pit choice ``[stay out, S, M, H, I, W]`` x a pace mode
``[conserve, balanced, push]`` x a racecraft intent ``[defend, normal, attack]`` (slice 4);
``encode``/``decode`` map it to and from ``0..53`` as ``(pit * 3 + pace) * 3 + intent``. Use
``pit_of`` / ``pace_of`` / ``intent_of`` / ``is_stop`` instead of arithmetic on action numbers
(slice 3's 18-action layout is gone: its policies are refused by their action count).
"""

import numpy as np

from f1ar.domain.race_state import (
    CancelPitStop,
    CarState,
    Compound,
    PaceMode,
    RacecraftIntent,
    RequestPitStop,
    SetPaceMode,
    SetRacecraftIntent,
)
from f1ar.simulation import lightweight_engine as engine

PIT_OPTIONS: tuple[Compound | None, ...] = (
    None,
    Compound.SOFT,
    Compound.MEDIUM,
    Compound.HARD,
    Compound.INTERMEDIATE,
    Compound.WET,
)
PACE_OPTIONS: tuple[PaceMode, ...] = (PaceMode.CONSERVE, PaceMode.BALANCED, PaceMode.PUSH)
INTENT_OPTIONS: tuple[RacecraftIntent, ...] = (
    RacecraftIntent.DEFEND,
    RacecraftIntent.NORMAL,
    RacecraftIntent.ATTACK,
)
N_PACE = len(PACE_OPTIONS)
N_INTENT = len(INTENT_OPTIONS)
PER_PIT = N_PACE * N_INTENT
"""Actions per pit choice (the stay-out block is ``0 .. PER_PIT - 1``)."""
N_ACTIONS = len(PIT_OPTIONS) * PER_PIT
LEGACY_N_ACTIONS = len(PIT_OPTIONS) * N_PACE
"""Slices 1-3b: pit x pace only (18 actions). Such a policy always raced NORMAL."""

Command = RequestPitStop | CancelPitStop | SetPaceMode | SetRacecraftIntent


def decode(action: int) -> tuple[Compound | None, PaceMode, RacecraftIntent]:
    rest, intent = divmod(int(action), N_INTENT)
    pit, pace = divmod(rest, N_PACE)
    return PIT_OPTIONS[pit], PACE_OPTIONS[pace], INTENT_OPTIONS[intent]


def encode(
    pit: Compound | None, pace: PaceMode, intent: RacecraftIntent = RacecraftIntent.NORMAL
) -> int:
    return (
        PIT_OPTIONS.index(pit) * N_PACE + PACE_OPTIONS.index(pace)
    ) * N_INTENT + INTENT_OPTIONS.index(intent)


def legacy_to_current(legacy_action: int) -> int:
    """An 18-action policy's action in the current layout (intent NORMAL)."""
    pit, pace = divmod(int(legacy_action), N_PACE)
    return encode(PIT_OPTIONS[pit], PACE_OPTIONS[pace])


def pit_of(action: int) -> Compound | None:
    return PIT_OPTIONS[int(action) // PER_PIT]


def pace_of(action: int) -> PaceMode:
    return decode(action)[1]


def intent_of(action: int) -> RacecraftIntent:
    return INTENT_OPTIONS[int(action) % N_INTENT]


def is_stop(actions: "np.ndarray | int") -> "np.ndarray | bool":
    """Whether an action (or an array of them) asks for a pit stop."""
    if isinstance(actions, np.ndarray):
        return actions >= PER_PIT
    return int(actions) >= PER_PIT


def agent_commands(car: CarState, action: int, actor_id: str) -> list[Command]:
    """The engine commands that carry out ``action`` for ``car`` (only what changes)."""
    pit, pace, intent = decode(action)
    out: list[Command] = []
    if pit is None:
        if engine.pending_stop(car) is not None:
            out.append(CancelPitStop(driver_id=car.driver_id, actor_id=actor_id))
    elif car.pit_request is not pit:
        out.append(RequestPitStop(driver_id=car.driver_id, compound=pit, actor_id=actor_id))
    if car.pace_mode is not pace:
        out.append(SetPaceMode(driver_id=car.driver_id, pace_mode=pace, actor_id=actor_id))
    if car.intent is not intent:
        out.append(SetRacecraftIntent(driver_id=car.driver_id, intent=intent, actor_id=actor_id))
    return out
