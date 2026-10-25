"""Pit-timing counterfactuals and what a pit wall sees (A2 pre-gate, EXP-032, D-063).

Every rule car is forked ``LEAD`` laps before its planned first stop (lap ``P``); in each branch
the stop is forced at ``P + d`` (``d`` in ``OFFSETS``; the car is a USER's for those laps and goes
back to the rules after the stop; ``d = 0`` uses the same hand-over, so it is the reference).
Identical luck (keyed RNG) makes the difference between branches the effect of the timing alone.

``fork_features`` is what the car's team can know at the fork, never the truth: race and car
state, the cars around it on the road, where it would rejoin after a stop, its own tyres as its
team's estimate reads them, and how sure that estimate is.
"""

from collections.abc import Iterator
from typing import Any

from f1ar.domain.race_state import (
    DRY_COMPOUNDS,
    CarState,
    CarStatus,
    Compound,
    RequestPitStop,
    SetStrategyController,
    StrategyController,
    WorldState,
    WorldStatus,
)
from f1ar.policy.baseline_strategy import team_knowledge, team_view
from f1ar.simulation import lightweight_engine as engine
from f1ar.simulation.lap_model import life_use_per_lap

OFFSETS = (-2, -1, 0, 1, 2, 3)
LEAD = 3
GAP_CAP_S = 30.0


def _distance(state: WorldState, car: CarState) -> float:
    return car.laps_completed * state.setup.track.length_m + car.distance_m


def _time_gaps(state: WorldState, car: CarState) -> list[tuple[float, CarState]]:
    """Every other running car with its time gap to ``car`` (positive = behind), from the
    distance between them at a neutral lap's speed."""
    speed = state.setup.track.length_m / state.setup.lap_model.base_lap_s
    me = _distance(state, car)
    return [
        ((me - _distance(state, o)) / speed, o)
        for o in state.cars
        if o.driver_id != car.driver_id and o.status is CarStatus.RUNNING
    ]


def _compound_index(c: Compound) -> float:
    return float(DRY_COMPOUNDS.index(c)) if c in DRY_COMPOUNDS else 3.0


def fork_features(state: WorldState, driver_id: str, planned: Compound) -> dict[str, float]:
    """What the team of ``driver_id`` sees now (no hidden truth)."""
    setup = state.setup
    car = state.car(driver_id)
    total = setup.track.total_laps
    order = engine.standings(state)
    paces = sorted({d.car_pace_offset_s for d in setup.drivers})
    gaps = _time_gaps(state, car)
    ahead = sorted(((-g, o) for g, o in gaps if g < 0), key=lambda x: x[0])
    behind = sorted(((g, o) for g, o in gaps if g > 0), key=lambda x: x[0])
    pit_loss = setup.pit.lane_loss_s + setup.pit.standstill_median_s
    seen_setup, seen = team_view(setup, car)
    driver = setup.driver(driver_id).profile.archetype
    use = life_use_per_lap(
        seen_setup.lap_model, driver, compound=car.compound, pace_mode=car.pace_mode
    )
    est, _ = team_knowledge(setup, driver_id)
    life_sd = est.life_factor_sd.get(car.compound, 0.0) if est is not None else 0.0

    def near(rows: list[tuple[float, CarState]]) -> tuple[float, float, float, float]:
        if not rows:
            return GAP_CAP_S, 0.0, 0.0, 1.0
        g, o = rows[0]
        return (
            min(g, GAP_CAP_S),
            float(o.tyre_age_laps),
            float(o.pit_stops > 0),
            _compound_index(o.compound),
        )

    ga, aa, sa, ca = near(ahead)
    gb, ab, sb, cb = near(behind)
    return {
        "laps_left": float(total - car.laps_completed),
        "race_progress": car.laps_completed / total,
        "position": float(order.index(driver_id) + 1),
        "pace_rank": float(paces.index(setup.driver(driver_id).car_pace_offset_s)),
        "gap_ahead_s": ga,
        "ahead_tyre_age": aa,
        "ahead_stopped": sa,
        "ahead_compound": ca,
        "gap_behind_s": gb,
        "behind_tyre_age": ab,
        "behind_stopped": sb,
        "behind_compound": cb,
        "cars_within_pit_loss_behind": float(sum(0 < g < pit_loss for g, _ in behind)),
        "car_at_rejoin": float(any(abs(g - pit_loss) < 2.0 for g, _ in behind)),
        "pit_loss_s": pit_loss,
        "own_tyre_age": float(car.tyre_age_laps),
        "own_compound": _compound_index(car.compound),
        "planned_compound": _compound_index(planned),
        "believed_life_used": seen.tyre_life_used,
        "believed_laps_of_life_left": (1.0 - seen.tyre_life_used) / use if use > 0 else 99.0,
        "believed_wear_s": seen.tyre_wear_s,
        "life_estimate_sd": life_sd,
        "share_stopped": sum(o.pit_stops > 0 for o in state.cars) / len(state.cars),
        "neutralisations": float(state.race_control.sc_count + state.race_control.vsc_count),
        "stuck_laps": float(car.stuck_laps),
    }


def forks(state: WorldState) -> Iterator[tuple[WorldState, str, int, Compound]]:
    """Run the race; yield (state, driver, planned lap, compound) when a rule car is ``LEAD``
    laps before its planned first stop (once per car)."""
    seen: set[str] = set()
    total = state.setup.track.total_laps
    s = state
    while s.world_status is WorldStatus.RUNNING:
        s = engine.step(s).state
        for c in s.cars:
            if c.driver_id in seen or c.status is not CarStatus.RUNNING or c.pit_stops > 0:
                continue
            if c.controller is not StrategyController.BASELINE or not c.planned_stops:
                continue
            p = c.planned_stops[0]
            if c.laps_completed == p.lap - LEAD and p.lap + max(OFFSETS) < total:
                seen.add(c.driver_id)
                yield s, c.driver_id, p.lap, p.compound


def branch(state: WorldState, driver_id: str, stop_lap: int, compound: Compound) -> WorldState:
    """The race from ``state`` with ``driver_id``'s stop forced at the end of ``stop_lap``."""
    s = engine.apply_action(
        state, SetStrategyController(driver_id=driver_id, controller=StrategyController.USER)
    ).state
    requested = handed_back = False
    while s.world_status is WorldStatus.RUNNING:
        c = s.car(driver_id)
        if not requested and c.status is CarStatus.RUNNING and c.laps_completed >= stop_lap - 1:
            s = engine.apply_action(s, RequestPitStop(driver_id=driver_id, compound=compound)).state
            requested = True
        if requested and not handed_back and c.pit_stops >= 1:
            s = engine.apply_action(
                s,
                SetStrategyController(driver_id=driver_id, controller=StrategyController.BASELINE),
            ).state
            handed_back = True
        s = engine.step(s).state
    return s


def points_score(state: WorldState, driver_id: str) -> float:
    from f1ar.training.objectives import Objective, TeamOrder, objective_value, race_results

    results = race_results(state)
    return objective_value(
        TeamOrder(Objective.POINTS), driver_id, results, len(state.setup.drivers)
    )


def timing_rows(state: WorldState) -> list[dict[str, Any]]:
    """Every fork of one race: its features and the score of each offset."""
    out = []
    for s, driver_id, lap, compound in forks(engine.start(state)):
        scores = {
            d: points_score(branch(s, driver_id, lap + d, compound), driver_id) for d in OFFSETS
        }
        out.append(
            {
                "driver_id": driver_id,
                "planned_lap": lap,
                "features": fork_features(s, driver_id, compound),
                "scores": scores,
            }
        )
    return out
