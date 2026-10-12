"""What a strategy policy may see at a decision point (plan §9.2, §9.4).

An observation = race/car state the driver could know + the driver's full profile vector.
Hidden simulation internals (notably ``CarState.lap_target_s``, the pre-sampled duration
of the lap in progress, and the accumulated ``tyre_wear_s``) are never exposed: that would
leak the future. The pit stop the team has called for this lap is known, so it's included.
"""

from pydantic import BaseModel, ConfigDict

from f1ar.domain.driver_profiles import PROFILE_FEATURE_NAMES
from f1ar.domain.race_state import Compound, PaceMode, WorldState
from f1ar.simulation.lightweight_engine import pending_stop, standings

OBSERVATION_SCHEMA_VERSION = 2

STATE_FEATURE_NAMES: tuple[str, ...] = (
    "race.lap_number",
    "race.laps_remaining",
    "race.lap_fraction",
    "car.position",
    "car.tyre_age_laps",
    *(f"car.compound.{c.value.lower()}" for c in Compound),
    *(f"car.pace_mode.{m.value.lower()}" for m in PaceMode),
    "car.last_lap_time_s",
    "car.has_last_lap",
    "car.pit_stops",
    "car.boxing_this_lap",
    "car.distinct_compounds_used",
)

OBSERVATION_FEATURE_NAMES: tuple[str, ...] = STATE_FEATURE_NAMES + PROFILE_FEATURE_NAMES


class Observation(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = OBSERVATION_SCHEMA_VERSION
    driver_id: str
    tick: int
    features: dict[str, float]

    def vector(self) -> tuple[float, ...]:
        return tuple(self.features[name] for name in OBSERVATION_FEATURE_NAMES)


def build_observation(state: WorldState, driver_id: str) -> Observation:
    car = state.car(driver_id)
    total_laps = state.setup.track.total_laps
    profile = state.setup.driver(driver_id).profile

    features: dict[str, float] = {
        "race.lap_number": float(min(car.laps_completed + 1, total_laps)),
        "race.laps_remaining": float(total_laps - car.laps_completed),
        "race.lap_fraction": car.distance_m / state.setup.track.length_m,
        "car.position": float(standings(state).index(driver_id) + 1),
        "car.tyre_age_laps": float(car.tyre_age_laps),
        **{f"car.compound.{c.value.lower()}": float(car.compound is c) for c in Compound},
        **{f"car.pace_mode.{m.value.lower()}": float(car.pace_mode is m) for m in PaceMode},
        "car.last_lap_time_s": car.last_lap_time_s or 0.0,
        "car.has_last_lap": float(car.last_lap_time_s is not None),
        "car.pit_stops": float(car.pit_stops),
        "car.boxing_this_lap": float(pending_stop(car) is not None),
        "car.distinct_compounds_used": float(len(set(car.compounds_used))),
        **profile.features(),
    }
    assert tuple(features) == OBSERVATION_FEATURE_NAMES
    return Observation(driver_id=driver_id, tick=state.sim_tick, features=features)
