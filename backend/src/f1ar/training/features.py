"""The strategist's observation as a fixed-length vector (M5, ``docs/M5_DESIGN.md`` §7).

``RL_OBSERVATION_SCHEMA = "rl_obs_v3"`` (M5 slice 4, D-048); ``rl_obs_v1`` (slice 3) and
``rl_obs_v2`` (slice 3b, D-047) are still built for the policies pinned to them
(``RL_FEATURE_SETS``). The names define the order.

**rl_obs_v1** is built from:

1. the engine's observation (schema v5: race, weather and forecast, the car, the driver's
   profile; tested to contain nothing from the future, US-06), scaled to about [0, 1];
2. the gaps to the cars directly ahead and behind (from track positions *now*);
3. the car's strength and the track (pit loss, tyre wear level, overtaking difficulty, laps);
4. the team principal's objective and the championship context.

**rl_obs_v2** = v1 plus what the W1-W6 world makes decisive (EXP-009 found v1 couldn't tell
which compound suits an unseen track):

5. the track's tyres per dry compound, exactly as the rule planner reads them from the setup:
   life (laps), pace offset vs the medium, wear per lap; the fresh-tyre bonus and the cliff;
6. the car's own set: share of its life used, laps left at BALANCED pace, wear so far;
7. the rule planner's advice for this car now (``plan_for_car`` + ``rule_pace``): laps to its
   next stop, that stop's compound, stops planned, the pace it would drive; and whether a stop
   now would be a cheap one (SC/VSC with the next planned stop inside the window);
8. undercut context: tyre age and stops made by the cars directly ahead and behind (visible
   on the timing screens in real F1).

**rl_obs_v3** = v2 plus what a racecraft decision (ATTACK / NORMAL / DEFEND, slice 4) and a
goal need: the car's current intent and its own attack/defend terms (W3); for the cars directly
ahead and behind their racecraft terms and car-pace difference (who is faster) and whether
they're attacking/defending this lap (visible on track); the points at stake in the race (the
points of the current position, of one place up and one down); and the title rival in this race
(where they are relative to this car, in places and seconds).

Everything here is known at decision time. A model is pinned to its schema version; changing
the order or meaning means a new version.
"""

import numpy as np

from f1ar.domain.race_state import (
    DRY_COMPOUNDS,
    CarStatus,
    Compound,
    PaceMode,
    RacecraftIntent,
    RacePhase,
    WorldState,
)
from f1ar.policy.baseline_strategy import dry_compounds_required, plan_for_car
from f1ar.policy.observation import OBSERVATION_FEATURE_NAMES, build_observation
from f1ar.simulation.lap_model import life_use_per_lap
from f1ar.simulation.lightweight_engine import standings
from f1ar.training.objectives import POINTS_2024, ChampionshipContext, Objective, TeamOrder

RL_OBSERVATION_SCHEMA = "rl_obs_v3"
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
RL_V1_NAMES: tuple[str, ...] = (*OBSERVATION_FEATURE_NAMES, *_EXTRA)
"""v1 order. ``car.last_lap_time_s`` is only scaled (/ 200 s); ``car.last_lap_vs_base``
(relative to the track's base lap) is the comparable one."""
_DRY = tuple(c.value.lower() for c in DRY_COMPOUNDS)
_V2_EXTRA = (
    *(f"track.{c}.life_laps" for c in _DRY),
    *(f"track.{c}.offset_s" for c in _DRY),
    *(f"track.{c}.wear_s_per_lap" for c in _DRY),
    "track.fresh_tyre_bonus_s",
    "track.cliff_s_per_lap",
    "car.tyre_life_used",
    "car.tyre_laps_left",
    "car.tyre_wear_s",
    "plan.has_stop",
    "plan.laps_to_stop",
    *(f"plan.compound.{c}" for c in _DRY),
    "plan.stops",
    *(f"plan.pace.{m.value.lower()}" for m in PaceMode),
    "plan.cheap_stop_now",
    "ahead.tyre_age_laps",
    "ahead.stops_diff",
    "behind.tyre_age_laps",
    "behind.stops_diff",
)
RL_V2_NAMES: tuple[str, ...] = (*RL_V1_NAMES, *_V2_EXTRA)
_V3_EXTRA = (
    *(f"car.intent.{i.value.lower()}" for i in RacecraftIntent),
    "car.attack_s",
    "car.defend_s",
    "ahead.defend_s",
    "ahead.pace_gap_s",
    "ahead.attacking_or_defending",
    "behind.attack_s",
    "behind.pace_gap_s",
    "behind.attacking",
    "race.points_now",
    "race.points_one_up",
    "race.points_one_down",
    "rival.in_race",
    "rival.places_ahead",
    "rival.gap_s",
)
RL_V3_NAMES: tuple[str, ...] = (*RL_V2_NAMES, *_V3_EXTRA)
RL_FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "rl_obs_v1": RL_V1_NAMES,
    "rl_obs_v2": RL_V2_NAMES,
    "rl_obs_v3": RL_V3_NAMES,
}
RL_FEATURE_NAMES: tuple[str, ...] = RL_FEATURE_SETS[RL_OBSERVATION_SCHEMA]
"""The current schema's order."""
LIFE_SCALE = 40.0


def schema_for_inputs(n_inputs: int) -> str:
    """The observation schema a network with ``n_inputs`` inputs was built for."""
    for schema, names in RL_FEATURE_SETS.items():
        if len(names) == n_inputs:
            return schema
    raise ValueError(f"no observation schema has {n_inputs} features")


def _neighbours(state: WorldState, driver_id: str) -> tuple[str | None, str | None]:
    """The running cars directly ahead / behind in the order."""
    running = [d for d in standings(state) if state.car(d).status is CarStatus.RUNNING]
    if driver_id not in running:
        return None, None
    i = running.index(driver_id)
    return (
        running[i - 1] if i > 0 else None,
        running[i + 1] if i + 1 < len(running) else None,
    )


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


def _v2_values(state: WorldState, driver_id: str) -> dict[str, float]:
    from f1ar.policy.strategists import rule_pace  # import here: strategists imports training

    setup = state.setup
    params = setup.lap_model
    car = state.car(driver_id)
    driver = setup.driver(driver_id).profile.archetype
    v: dict[str, float] = {}
    for c in DRY_COMPOUNDS:
        name = c.value.lower()
        cp = params.compounds[c]
        v[f"track.{name}.life_laps"] = (cp.life_laps or LIFE_SCALE * 2) / LIFE_SCALE
        v[f"track.{name}.offset_s"] = cp.offset_s
        v[f"track.{name}.wear_s_per_lap"] = cp.deg_s_per_lap * 10.0
    v["track.fresh_tyre_bonus_s"] = params.fresh_tyre_bonus_s
    v["track.cliff_s_per_lap"] = params.cliff_s_per_lap
    v["car.tyre_life_used"] = min(car.tyre_life_used, 2.0)
    u = life_use_per_lap(params, driver, compound=car.compound, pace_mode=PaceMode.BALANCED)
    laps_left = (1.0 - car.tyre_life_used) / u if u > 0 else LIFE_SCALE
    v["car.tyre_laps_left"] = float(np.clip(laps_left, -LIFE_SCALE, 2 * LIFE_SCALE)) / LIFE_SCALE
    v["car.tyre_wear_s"] = min(car.tyre_wear_s, 20.0) / 5.0
    plan = plan_for_car(setup, car)
    nxt = plan[0] if plan else None
    v["plan.has_stop"] = float(nxt is not None)
    v["plan.laps_to_stop"] = (nxt.lap - car.laps_completed) / LIFE_SCALE if nxt else 0.0
    for c in DRY_COMPOUNDS:
        v[f"plan.compound.{c.value.lower()}"] = float(nxt is not None and nxt.compound is c)
    v["plan.stops"] = len(plan) / 3.0
    stop_now = nxt.compound if nxt is not None and nxt.lap == car.laps_completed + 1 else None
    pace = rule_pace(setup, car, plan, stop_now)
    for m in PaceMode:
        v[f"plan.pace.{m.value.lower()}"] = float(pace is m)
    window = setup.race_control.neutralised_pit_window_laps
    v["plan.cheap_stop_now"] = float(
        state.race_control.phase is not RacePhase.GREEN
        and nxt is not None
        and nxt.lap - car.laps_completed <= window
    )
    ahead, behind = _neighbours(state, driver_id)
    for side, other in (("ahead", ahead), ("behind", behind)):
        if other is None:
            v[f"{side}.tyre_age_laps"] = 0.0
            v[f"{side}.stops_diff"] = 0.0
            continue
        o = state.car(other)
        v[f"{side}.tyre_age_laps"] = o.tyre_age_laps / LIFE_SCALE
        v[f"{side}.stops_diff"] = float(np.clip(o.pit_stops - car.pit_stops, -3, 3)) / 3.0
    return v


def _progress(state: WorldState, driver_id: str) -> float:
    c = state.car(driver_id)
    return c.laps_completed + c.distance_m / state.setup.track.length_m


def _points_at(position: int) -> float:
    # the same table for every season 2010-2025 (H1); the refuelling era (H1b) will need the setup's
    return POINTS_2024[position - 1] / POINTS_2024[0] if 1 <= position <= len(POINTS_2024) else 0.0


def _v3_values(state: WorldState, driver_id: str, order: TeamOrder) -> dict[str, float]:
    setup = state.setup
    car = state.car(driver_id)
    me = setup.driver(driver_id)
    v: dict[str, float] = {}
    for i in RacecraftIntent:
        v[f"car.intent.{i.value.lower()}"] = float(car.intent is i)
    v["car.attack_s"] = me.attack_s / 0.3
    v["car.defend_s"] = me.defend_s / 0.3
    ahead, behind = _neighbours(state, driver_id)
    if ahead is not None:
        a = setup.driver(ahead)
        v["ahead.defend_s"] = a.defend_s / 0.3
        v["ahead.pace_gap_s"] = (a.car_pace_offset_s - me.car_pace_offset_s) / 2.0
        v["ahead.attacking_or_defending"] = float(
            state.car(ahead).lap_intent is not RacecraftIntent.NORMAL
        )
    else:
        v["ahead.defend_s"] = v["ahead.pace_gap_s"] = v["ahead.attacking_or_defending"] = 0.0
    if behind is not None:
        b = setup.driver(behind)
        v["behind.attack_s"] = b.attack_s / 0.3
        v["behind.pace_gap_s"] = (me.car_pace_offset_s - b.car_pace_offset_s) / 2.0
        v["behind.attacking"] = float(state.car(behind).lap_intent is RacecraftIntent.ATTACK)
    else:
        v["behind.attack_s"] = v["behind.pace_gap_s"] = v["behind.attacking"] = 0.0
    order_now = standings(state)
    position = order_now.index(driver_id) + 1
    v["race.points_now"] = _points_at(position)
    v["race.points_one_up"] = _points_at(position - 1) if position > 1 else _points_at(1)
    v["race.points_one_down"] = _points_at(position + 1)
    rival = order.rival
    if rival is not None and rival in order_now and state.car(rival).status is CarStatus.RUNNING:
        v["rival.in_race"] = 1.0
        v["rival.places_ahead"] = (position - (order_now.index(rival) + 1)) / 20.0
        gap = (_progress(state, rival) - _progress(state, driver_id)) * setup.lap_model.base_lap_s
        v["rival.gap_s"] = float(np.clip(gap, -60.0, 60.0)) / 60.0
    else:
        v["rival.in_race"] = v["rival.places_ahead"] = v["rival.gap_s"] = 0.0
    return v


def rl_observation(
    state: WorldState,
    driver_id: str,
    order: TeamOrder,
    context: ChampionshipContext | None,
    schema: str = RL_OBSERVATION_SCHEMA,
) -> np.ndarray:
    names = RL_FEATURE_SETS[schema]
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
    if schema != "rl_obs_v1":
        values.update(_v2_values(state, driver_id))
    if schema not in ("rl_obs_v1", "rl_obs_v2"):
        values.update(_v3_values(state, driver_id, order))
    return np.array([values[n] for n in names], dtype=np.float32)
