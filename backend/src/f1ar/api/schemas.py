"""Request bodies of the M3 API (plan §13.2-13.3; documented in docs/API_CONTRACTS.md).

Responses are plain JSON built from existing models (``CommandAck``, ``SnapshotMeta``, the view
models), so they aren't duplicated here. Every mutating request may carry a client-chosen
``idempotency_key``: repeating it returns the original acknowledgement (``duplicate: true``)
and never applies the command twice.
"""

import uuid
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from f1ar.domain.race_state import Compound, PaceMode, StrategyController
from f1ar.simulation.timeline import SeedMode
from f1ar.simulation.world_runner import MAX_STEP_TICKS, Command, CommandType

IdempotencyKey = Field(default=None, min_length=1, max_length=128)
ActorId = Field(default="USER", pattern=r"^[A-Za-z0-9_.-]{1,64}$")


class _Request(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _key(key: str | None) -> str:
    return key or f"api-{uuid.uuid4().hex}"


class CreateWorldRequest(_Request):
    race_id: str
    seed: int | None = Field(default=None, ge=0, le=2**62)
    """None = a fresh random seed."""
    drivers: list[str] | None = Field(default=None, min_length=1)
    """3-letter codes; None = the full grid."""
    synthetic: bool = False
    """Ignore data-derived inputs (equal cars, all on MEDIUM)."""
    speed: int = 1


class ControlRequest(_Request):
    """Body of start / pause / resume (optional)."""

    idempotency_key: str | None = IdempotencyKey


class SpeedRequest(_Request):
    speed: int
    idempotency_key: str | None = IdempotencyKey


class StepRequest(_Request):
    ticks: int = Field(default=1, ge=1, le=MAX_STEP_TICKS)
    idempotency_key: str | None = IdempotencyKey


class SaveRequest(_Request):
    label: str | None = Field(default=None, min_length=1, max_length=120)
    idempotency_key: str | None = IdempotencyKey


class ActionType(StrEnum):
    PIT_REQUEST = "PIT_REQUEST"
    PIT_CANCEL = "PIT_CANCEL"
    SET_PACE_MODE = "SET_PACE_MODE"
    SET_CONTROLLER = "SET_CONTROLLER"
    DEPLOY_SC = "DEPLOY_SC"
    DEPLOY_VSC = "DEPLOY_VSC"


class ActionRequest(_Request):
    """A driver/team decision or a race-director action (plan §13.3)."""

    action_type: ActionType
    driver_id: str | None = None
    compound: Compound | None = None
    pace_mode: PaceMode | None = None
    controller: StrategyController | None = None
    laps: int = Field(default=4, ge=1, le=20)
    """DEPLOY_SC: safety-car laps."""
    duration_s: float = Field(default=90.0, gt=0, le=600)
    """DEPLOY_VSC: VSC length in sim seconds."""
    actor_id: str = ActorId
    idempotency_key: str | None = IdempotencyKey

    def to_command(self) -> Command:
        return Command(
            idempotency_key=_key(self.idempotency_key),
            type=CommandType(self.action_type.value),
            actor_id=self.actor_id,
            driver_id=self.driver_id,
            compound=self.compound,
            pace_mode=self.pace_mode,
            controller=self.controller,
            laps=self.laps,
            duration_s=self.duration_s,
        )


class LoadRequest(_Request):
    snapshot_id: str | None = None
    """None = continue from the world's newest valid snapshot."""
    speed: int | None = None


class ForkRequest(_Request):
    name: str = Field(min_length=1, max_length=120)
    source_snapshot_id: str | None = None
    """None = the live world's current tick (saved first), or a stored world's newest save."""
    seed_mode: SeedMode = SeedMode.COMMON_EXOGENOUS_RANDOMNESS
    action_overrides: list[ActionRequest] = Field(default_factory=list, max_length=20)
    """Applied on the branch, in order, as logged commands right after the fork."""
    speed: int | None = None


def control(kind: CommandType, body: ControlRequest | None) -> Command:
    return Command(idempotency_key=_key(body.idempotency_key if body else None), type=kind)


def speed_command(body: SpeedRequest) -> Command:
    return Command(
        idempotency_key=_key(body.idempotency_key), type=CommandType.SET_SPEED, speed=body.speed
    )


def step_command(body: StepRequest) -> Command:
    return Command(
        idempotency_key=_key(body.idempotency_key), type=CommandType.STEP, ticks=body.ticks
    )


def save_command(body: SaveRequest | None) -> Command:
    return Command(
        idempotency_key=_key(body.idempotency_key if body else None),
        type=CommandType.SAVE,
        label=body.label if body else None,
    )
