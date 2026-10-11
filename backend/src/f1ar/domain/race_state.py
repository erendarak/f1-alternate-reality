"""Authoritative, immutable world state (plan §4.4).

A ``WorldState`` contains everything that influences the future of the world: race setup
(track, model parameters, embedded driver profiles), per-car state, clocks and RNG spec.
Nothing outside it may affect ``step()``. That is what makes save, load, replay and fork
exact. States are frozen; the engine returns a new state per tick and never mutates one.
"""

from enum import StrEnum
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from f1ar.domain.driver_profiles import DriverProfile
from f1ar.domain.provenance import SourceType
from f1ar.persistence.snapshot_codec import state_hash
from f1ar.simulation.rng_streams import RngSpec

WORLD_SCHEMA_VERSION = 1


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class WorldStatus(StrEnum):
    CREATED = "CREATED"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    COMPLETED = "COMPLETED"


class CarStatus(StrEnum):
    RUNNING = "RUNNING"
    FINISHED = "FINISHED"


class Compound(StrEnum):
    SOFT = "SOFT"
    MEDIUM = "MEDIUM"
    HARD = "HARD"


class PaceMode(StrEnum):
    CONSERVE = "CONSERVE"
    BALANCED = "BALANCED"
    PUSH = "PUSH"


# --- setup: fixed for the lifetime of a world ---------------------------------------------


class TrackSpec(_Frozen):
    track_id: str
    length_m: float = Field(gt=0)
    total_laps: int = Field(gt=0)
    source_note: str


class CompoundParams(_Frozen):
    offset_s: float
    """Lap-time offset of a fresh tyre vs. the reference compound."""
    deg_s_per_lap: float = Field(ge=0)
    """Added lap time per lap of tyre age for a neutral driver in BALANCED mode."""


class PaceModeParams(_Frozen):
    offset_s: float
    wear_multiplier: float = Field(gt=0)
    """Scales how fast tyre age turns into lap time (push wears tyres faster)."""


class LapModelParams(_Frozen):
    """Coefficients of the minimal lap-time model. Synthetic placeholders until calibration (M4)."""

    source_type: Literal[SourceType.EDITORIAL_ASSUMPTION]
    note: str
    base_lap_s: float = Field(gt=0)
    skill_gap_s: float = Field(ge=0)
    """Lap time lost by a driver with race_pace_skill=0 vs. 1."""
    fuel_s_per_lap_remaining: float = Field(ge=0)
    tyre_mgmt_wear_max: float = Field(gt=0)
    """Wear factor at tire_management=0; it falls linearly to ``tyre_mgmt_wear_min`` at 1."""
    tyre_mgmt_wear_min: float = Field(gt=0)
    noise_floor_s: float = Field(ge=0)
    noise_inconsistency_s: float = Field(ge=0)
    """Extra lap-time std-dev for a driver with consistency=0 vs. 1."""
    min_lap_fraction: float = Field(gt=0, lt=1)
    """Sampled lap times are clamped to at least this fraction of ``base_lap_s``."""
    compounds: dict[Compound, CompoundParams]
    pace_modes: dict[PaceMode, PaceModeParams]

    @model_validator(mode="after")
    def _complete(self) -> Self:
        if set(self.compounds) != set(Compound) or set(self.pace_modes) != set(PaceMode):
            raise ValueError("lap model must define every compound and pace mode")
        return self


class DriverEntry(_Frozen):
    driver_id: str
    code: str
    team_id: str
    grid_position: int = Field(ge=0)
    start_compound: Compound
    profile: DriverProfile


class RaceSetup(_Frozen):
    event_id: str
    track: TrackSpec
    lap_model: LapModelParams
    tick_seconds: float = Field(gt=0)
    profile_set_manifest_hash: str
    """Hash of the full ProfileSet the embedded profiles were taken from."""
    drivers: tuple[DriverEntry, ...]

    @model_validator(mode="after")
    def _unique(self) -> Self:
        ids = [d.driver_id for d in self.drivers]
        if len(set(ids)) != len(ids) or not ids:
            raise ValueError("race setup needs at least one driver and unique driver ids")
        return self

    def driver(self, driver_id: str) -> DriverEntry:
        for d in self.drivers:
            if d.driver_id == driver_id:
                return d
        raise KeyError(driver_id)


# --- dynamic state ------------------------------------------------------------------------


class CarState(_Frozen):
    driver_id: str
    status: CarStatus
    laps_completed: int = Field(ge=0)
    distance_m: float = Field(ge=0)
    """Distance into the current lap, in [0, track length)."""
    lap_elapsed_s: float = Field(ge=0)
    lap_target_s: float = Field(gt=0)
    """Pre-sampled duration of the current lap. HIDDEN: never part of an observation."""
    compound: Compound
    tyre_age_laps: int = Field(ge=0)
    pace_mode: PaceMode
    last_lap_time_s: float | None
    finish_time_s: float | None


class RaceEvent(_Frozen):
    seq: int = Field(ge=0)
    tick: int = Field(ge=0)
    type: Literal["LAP_COMPLETED", "CAR_FINISHED", "RACE_COMPLETED"]
    driver_id: str | None
    payload: dict[str, float | int | str]


class WorldState(_Frozen):
    schema_version: Literal[1]
    world_id: str
    timeline_id: str
    parent_timeline_id: str | None
    world_status: WorldStatus
    mode: Literal["FREE_SIMULATION"]
    engine_version: str
    rng: RngSpec
    setup: RaceSetup
    sim_tick: int = Field(ge=0)
    sim_clock_s: float = Field(ge=0)
    cars: tuple[CarState, ...]
    event_seq: int = Field(ge=0)
    """Number of events emitted so far; the next event gets this sequence number."""

    @model_validator(mode="after")
    def _cars_match_setup(self) -> Self:
        if tuple(c.driver_id for c in self.cars) != tuple(d.driver_id for d in self.setup.drivers):
            raise ValueError("cars must match setup drivers one-to-one, in the same order")
        return self

    def car(self, driver_id: str) -> CarState:
        for c in self.cars:
            if c.driver_id == driver_id:
                return c
        raise KeyError(driver_id)

    def canonical_hash(self) -> str:
        return state_hash(self.model_dump(mode="json"))
