"""The strategist's observation as a fixed-length vector (M5, ``docs/M5_DESIGN.md`` §7).

``RL_OBSERVATION_SCHEMA = "rl_obs_v1"``; the names below define the order. Built from:

1. the engine's observation (schema v5: race, weather and forecast, the car, the driver's
   profile; tested to contain nothing from the future, US-06), scaled to about [0, 1];
2. the gaps to the cars directly ahead and behind (from track positions *now*);
3. the car's strength and the track (pit loss, tyre wear level, overtaking difficulty, laps);
4. the team principal's objective and the championship context.

Everything here is known at decision time. A model is pinned to this schema version; changing
the order or meaning means a new version.
"""

import numpy as np

from f1ar.domain.race_state import CarStatus, Compound, WorldState
from f1ar.policy.baseline_strategy import dry_compounds_required
from f1ar.policy.observation import OBSERVATION_FEATURE_NAMES, build_observation
from f1ar.simulation.lightweight_engine import standings
from f1ar.training.objectives import ChampionshipContext, Objective, TeamOrder

RL_OBSERVATION_SCHEMA = "rl_obs_v1"
GAP_CAP_S = 10.0
POINTS_SCALE = 25.0
SEASON_SCALE = 24.0
"""Championship gaps are scaled by a whole season of wins (600 points)."""

_SCALE = {
    "race.lap_number": 80.0,
    "race.laps_remaining": 80.0,
    "car.position": 20.0,
    "car.tyre_age_laps": 40.0,
    "car.pit_stops": 3.0,
    "car.distinct_compounds_used": 3.0,
    "car.last_lap_traffic_loss_s": 2.0,
}
_EXTRA = (
    "car.last_lap_vs_base",
    "gap.ahead_s",
    "gap.behind_s",
    "gap.has_ahead",
    "gap.has_behind",
    "car.pace_offset_s",
    "car.dry_compounds_required",
    "track.pit_loss_s",
    "track.medium_wear_s_per_lap",
    "track.overtake_threshold_s",
    "track.total_laps",
    *(f"objective.{o.value.lower()}" for o in Objective),
    "championship.has_context",
    "championship.position",
    "championship.gap_to_leader",
    "championship.gap_to_rival",
    "championship.races_left",
)
RL_FEATURE_NAMES: tuple[str, ...] = (*OBSERVATION_FEATURE_NAMES, *_EXTRA)
"""Vector order. ``car.last_lap_time_s`` is only scaled (/ 200 s); ``car.last_lap_vs_base``
(relative to the track's base lap) is the comparable one."""


def _gaps(state: WorldState, driver_id: str) -> tuple[float | None, float | None]:
    """Seconds to the running car directly ahead / behind in the order, from race progress now."""
    length = state.setup.track.length_m
    lap_s = state.setup.lap_model.base_lap_s
    running = [d for d in standings(state) if state.car(d).status is CarStatus.RUNNING]
    if driver_id not in running:
        return None, None
    i = running.index(driver_id)

    def progress(d: str) -> float:
        c = state.car(d)
        return c.laps_completed + c.distance_m / length

    me = progress(driver_id)
    ahead = (progress(running[i - 1]) - me) * lap_s if i > 0 else None
    behind = (me - progress(running[i + 1])) * lap_s if i + 1 < len(running) else None
    return ahead, behind


def rl_observation(
    state: WorldState,
    driver_id: str,
    order: TeamOrder,
    context: ChampionshipContext | None,
) -> np.ndarray:
    setup = state.setup
    base = build_observation(state, driver_id).features
    values: dict[str, float] = {
        name: base[name] / _SCALE.get(name, 1.0) for name in OBSERVATION_FEATURE_NAMES
    }
    values["car.last_lap_time_s"] = base["car.last_lap_time_s"] / 200.0
    car = state.car(driver_id)
    entry = setup.driver(driver_id)
    ahead, behind = _gaps(state, driver_id)
    last = base["car.last_lap_time_s"]
    values["car.last_lap_vs_base"] = (
        float(np.clip((last / setup.lap_model.base_lap_s - 1.0) * 10.0, -1.0, 3.0)) if last else 0.0
    )
    values["gap.ahead_s"] = min(ahead, GAP_CAP_S) / GAP_CAP_S if ahead is not None else 1.0
    values["gap.behind_s"] = min(behind, GAP_CAP_S) / GAP_CAP_S if behind is not None else 1.0
    values["gap.has_ahead"] = float(ahead is not None)
    values["gap.has_behind"] = float(behind is not None)
    values["car.pace_offset_s"] = entry.car_pace_offset_s / 2.0
    values["car.dry_compounds_required"] = dry_compounds_required(setup, car.compounds_used) / 2.0
    values["track.pit_loss_s"] = (setup.pit.lane_loss_s + setup.pit.standstill_median_s) / 30.0
    values["track.medium_wear_s_per_lap"] = (
        setup.lap_model.compounds[Compound.MEDIUM].deg_s_per_lap * 10.0
    )
    values["track.overtake_threshold_s"] = setup.traffic.overtake_threshold_s
    values["track.total_laps"] = setup.track.total_laps / 80.0
    for o in Objective:
        values[f"objective.{o.value.lower()}"] = float(order.objective is o)
    scale = POINTS_SCALE * SEASON_SCALE
    if context is not None:
        rival = order.rival
        values["championship.has_context"] = 1.0
        values["championship.position"] = context.position(driver_id) / 20.0
        values["championship.gap_to_leader"] = context.gap_to_leader(driver_id) / scale
        values["championship.gap_to_rival"] = (
            (context.points[driver_id] - context.points[rival]) / scale
            if rival is not None and rival in context.points
            else 0.0
        )
        values["championship.races_left"] = context.races_left / context.season_races
    else:
        for name in _EXTRA:
            if name.startswith("championship."):
                values[name] = 0.0
    return np.array([values[n] for n in RL_FEATURE_NAMES], dtype=np.float32)
