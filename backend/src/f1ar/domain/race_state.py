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

WORLD_SCHEMA_VERSION = 4


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


class StrategyController(StrEnum):
    """Who decides a car's pit stops."""

    BASELINE = "BASELINE"
    """The built-in baseline team strategy (``f1ar.policy.baseline_strategy``)."""
    USER = "USER"
    """A human (or external agent) via pit actions; the baseline stays out of it."""


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


class PitParams(_Frozen):
    """Pit-stop time model (plan §7.5). Standstill shape credited to TUM FTM race-simulation
    (Heilmeier et al.), which models standstill times with a log-logistic (fisk) distribution."""

    source_type: Literal[SourceType.EDITORIAL_ASSUMPTION, SourceType.ESTIMATED_FROM_DATA]
    note: str
    lane_loss_s: float = Field(ge=0)
    """Time lost driving pit entry + pit lane + pit exit instead of the racing line."""
    standstill_median_s: float = Field(gt=0)
    standstill_shape: float = Field(gt=1)
    """Log-logistic shape: higher = fewer slow stops."""
    standstill_max_s: float = Field(gt=0)
    """Draws are capped here (a stuck wheel, not a lost race)."""
    min_distinct_dry_compounds: int = Field(ge=0, le=3)
    """Dry-race rule: compounds a finisher must have used (2 in 2024; 0 disables)."""


class DriverEntry(_Frozen):
    driver_id: str
    code: str
    team_id: str
    grid_position: int = Field(ge=0)
    start_compound: Compound
    car_pace_offset_s: float = Field(default=0.0, ge=0.0, le=10.0)
    """Car performance: lap time lost vs. the fastest car in the field (0 = fastest car).
    Separate from driver skill (plan §5.2); see ``RaceSetup.inputs`` for its provenance."""
    profile: DriverProfile


class RaceSetup(_Frozen):
    event_id: str
    track: TrackSpec
    lap_model: LapModelParams
    pit: PitParams
    tick_seconds: float = Field(gt=0)
    profile_set_manifest_hash: str
    """Hash of the full ProfileSet the embedded profiles were taken from."""
    drivers: tuple[DriverEntry, ...]
    inputs: dict[str, str] = Field(default_factory=dict)
    """Provenance of data-derived inputs, e.g. ``{"start_compounds": "historical_verified: ..."}``.
    Empty = synthetic defaults (all MEDIUM, equal cars)."""

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


class PlannedStop(_Frozen):
    lap: int = Field(ge=1)
    """Pit at the end of this lap (the car crosses the line in the pit lane)."""
    compound: Compound


class CarState(_Frozen):
    driver_id: str
    status: CarStatus
    laps_completed: int = Field(ge=0)
    distance_m: float = Field(ge=0)
    """Distance into the current lap, in [0, track length). Derived from ``lap_elapsed_s``."""
    lap_elapsed_s: float = Field(ge=0)
    lap_target_s: float = Field(gt=0)
    """Pre-sampled duration of the current lap, including ``pit_hold_s``. HIDDEN: never part
    of an observation."""
    pit_hold_s: float = Field(ge=0)
    """Out-lap only: pit-lane loss + standstill, spent at the line before driving the lap."""
    compound: Compound
    tyre_age_laps: int = Field(ge=0)
    tyre_wear_s: float = Field(ge=0)
    """Accumulated degradation of the current tyre set, as lap time added. HIDDEN: a real
    team only infers this from lap times, so it is not part of an observation."""
    pace_mode: PaceMode
    """Driver's intended mode; applies from the next lap start."""
    lap_pace_mode: PaceMode
    """Mode the lap in progress is being driven in (locked when the lap started)."""
    pit_request: Compound | None
    """Pending stop: the car pits for this compound at its next line crossing."""
    pit_stops: int = Field(ge=0)
    compounds_used: tuple[Compound, ...]
    """Distinct compounds raced so far, in first-use order (for the dry-tyre rule)."""
    controller: StrategyController
    planned_stops: tuple[PlannedStop, ...]
    """Baseline strategy's remaining plan (empty under USER control)."""
    last_lap_time_s: float | None
    finish_time_s: float | None


EventType = Literal[
    "LAP_COMPLETED",
    "CAR_FINISHED",
    "RACE_COMPLETED",
    "PACE_MODE_CHANGED",
    "PIT_REQUESTED",
    "PIT_CANCELLED",
    "PIT_STOP",
    "STRATEGY_PLANNED",
    "CONTROLLER_CHANGED",
    "CHEQUERED_FLAG",
    "TYRE_RULE_BREACHED",
]


class RaceEvent(_Frozen):
    seq: int = Field(ge=0)
    tick: int = Field(ge=0)
    type: EventType
    driver_id: str | None
    payload: dict[str, float | int | str]


# --- actions: decisions that change the future ----------------------------------------------


class SetPaceMode(_Frozen):
    """Driver pace intent (plan §3.6, §9.3). Takes effect from the driver's NEXT lap.

    The duration of the lap in progress was sampled when it started, so a mid-lap change
    can't retroactively change it. This also keeps decisions lap-granular like the engine.
    """

    type: Literal["SET_PACE_MODE"] = "SET_PACE_MODE"
    driver_id: str
    pace_mode: PaceMode
    actor_id: str = "USER"


class RequestPitStop(_Frozen):
    """Box at the end of the current lap for ``compound`` (plan §7.5: one command per stop).

    Replaces any pending request. Takes the car's strategy over (controller -> USER).
    """

    type: Literal["REQUEST_PIT_STOP"] = "REQUEST_PIT_STOP"
    driver_id: str
    compound: Compound
    actor_id: str = "USER"


class CancelPitStop(_Frozen):
    """Withdraw a pending request (stay out). Takes the car's strategy over (-> USER)."""

    type: Literal["CANCEL_PIT_STOP"] = "CANCEL_PIT_STOP"
    driver_id: str
    actor_id: str = "USER"


class SetStrategyController(_Frozen):
    """Hand a car's pit decisions to the baseline (it replans from now) or to the user."""

    type: Literal["SET_STRATEGY_CONTROLLER"] = "SET_STRATEGY_CONTROLLER"
    driver_id: str
    controller: StrategyController
    actor_id: str = "USER"


DriverAction = SetPaceMode | RequestPitStop | CancelPitStop | SetStrategyController
"""Union of all driver/team actions."""


class WorldState(_Frozen):
    schema_version: Literal[4]
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
    chequered_flag_s: float | None
    """Sim time the leader took the flag; every later line crossing finishes that car."""
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
