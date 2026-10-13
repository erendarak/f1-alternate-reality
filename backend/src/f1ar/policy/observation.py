"""What a strategy policy may see at a decision point (plan §9.2, §9.4).

An observation = race/car state the driver could know + the driver's full profile vector.
Hidden simulation internals (notably ``CarState.lap_target_s``, the pre-sampled duration
of the lap in progress, and the accumulated ``tyre_wear_s``) are never exposed: that would
leak the future. The pit stop the team has called for this lap is known, so it's included.

Weather (M4, US-06): the current rain flag and track wetness (what the team sees) and a noisy
forecast of rain in the next 10 minutes; never the true future rain timeline.
"""

from pydantic import BaseModel, ConfigDict

from f1ar.domain.driver_profiles import PROFILE_FEATURE_NAMES
from f1ar.domain.race_state import Compound, PaceMode, RacePhase, WorldState
from f1ar.simulation.lightweight_engine import pending_stop, standings
from f1ar.simulation.weather import forecast_rain_probability

OBSERVATION_SCHEMA_VERSION = 5

STATE_FEATURE_NAMES: tuple[str, ...] = (
    "race.lap_number",
    "race.laps_remaining",
    "race.lap_fraction",
    "race.vsc",
    "race.safety_car",
    "weather.raining",
    "weather.wetness",
    "weather.forecast_rain",
    "car.position",
    "car.tyre_age_laps",
    *(f"car.compound.{c.value.lower()}" for c in Compound),
    *(f"car.pace_mode.{m.value.lower()}" for m in PaceMode),
    "car.last_lap_time_s",
    "car.has_last_lap",
    "car.pit_stops",
    "car.boxing_this_lap",
    "car.distinct_compounds_used",
    "car.last_lap_traffic_loss_s",
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
        "race.vsc": float(state.race_control.phase is RacePhase.VSC),
        "race.safety_car": float(state.race_control.phase is RacePhase.SC),
        "weather.raining": float(state.weather.raining),
        "weather.wetness": state.weather.wetness,
        "weather.forecast_rain": forecast_rain_probability(
            state.setup.weather, state.rng.seed, state.sim_clock_s, state.weather.raining
        ),
        "car.position": float(standings(state).index(driver_id) + 1),
        "car.tyre_age_laps": float(car.tyre_age_laps),
        **{f"car.compound.{c.value.lower()}": float(car.compound is c) for c in Compound},
        **{f"car.pace_mode.{m.value.lower()}": float(car.pace_mode is m) for m in PaceMode},
        "car.last_lap_time_s": car.last_lap_time_s or 0.0,
        "car.has_last_lap": float(car.last_lap_time_s is not None),
        "car.pit_stops": float(car.pit_stops),
        "car.boxing_this_lap": float(pending_stop(car) is not None),
        "car.distinct_compounds_used": float(len(set(car.compounds_used))),
        "car.last_lap_traffic_loss_s": car.last_lap_traffic_loss_s,
        **profile.features(),
    }
    assert tuple(features) == OBSERVATION_FEATURE_NAMES
    return Observation(driver_id=driver_id, tick=state.sim_tick, features=features)
