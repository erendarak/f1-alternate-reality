"""The strategist's 18 actions and the engine commands that carry them out (M5, D-038).

An action is a pit choice ``[stay out, S, M, H, I, W]`` x a pace mode
``[conserve, balanced, push]``; ``encode``/``decode`` map it to and from ``0..17``.
"""

from f1ar.domain.race_state import (
    CancelPitStop,
    CarState,
    Compound,
    PaceMode,
    RequestPitStop,
    SetPaceMode,
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
N_ACTIONS = len(PIT_OPTIONS) * len(PACE_OPTIONS)

Command = RequestPitStop | CancelPitStop | SetPaceMode


def decode(action: int) -> tuple[Compound | None, PaceMode]:
    pit, pace = divmod(int(action), len(PACE_OPTIONS))
    return PIT_OPTIONS[pit], PACE_OPTIONS[pace]


def encode(pit: Compound | None, pace: PaceMode) -> int:
    return PIT_OPTIONS.index(pit) * len(PACE_OPTIONS) + PACE_OPTIONS.index(pace)


def agent_commands(car: CarState, action: int, actor_id: str) -> list[Command]:
    """The engine commands that carry out ``action`` for ``car`` (only what changes)."""
    pit, pace = decode(action)
    out: list[Command] = []
    if pit is None:
        if engine.pending_stop(car) is not None:
            out.append(CancelPitStop(driver_id=car.driver_id, actor_id=actor_id))
    elif car.pit_request is not pit:
        out.append(RequestPitStop(driver_id=car.driver_id, compound=pit, actor_id=actor_id))
    if car.pace_mode is not pace:
        out.append(SetPaceMode(driver_id=car.driver_id, pace_mode=pace, actor_id=actor_id))
    return out
