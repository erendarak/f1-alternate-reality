"""Baseline team strategy: the cheapest 0/1/2-stop plan under the expected lap model (plan §9.6).

This is the reference every learned strategist must beat. It is deliberately simple:

- It plans with the car's *expected* lap times (no noise), using the driver's own tyre wear
  (``tire_management``, the current pace mode, and the wear curve by tyre age). Two drivers in
  identical cars can therefore pit on different laps.
- Only terms that differ between plans count: compound offset, accumulated wear and pit
  loss. Fuel, skill and car pace add the same time to every plan.
- The dry-tyre rule (``PitParams.min_distinct_dry_compounds``) is a hard constraint.
- It doesn't know about traffic or rivals; safety cars only bring a planned stop forward
  (engine). It is an "empty track" optimum.

Stint costs come from wear tables by tyre age (M4, D-033: wear is no longer linear). With the
lap's wear step never negative, the cost of a fresh stint is convex in its length, so the best
split of a lap budget between two fresh stints is a min-plus convolution of two convex
sequences: merge their sorted marginal costs. That gives the best split for every budget at
once, so one plan costs O(laps x compound pairs) as before.
"""

from itertools import product

import numpy as np

from f1ar.domain.driver_profiles import DriverArchetype
from f1ar.domain.race_state import (
    DRY_COMPOUNDS,
    CarState,
    Compound,
    PaceMode,
    PlannedStop,
    RaceSetup,
)
from f1ar.simulation.lap_model import wear_per_lap_s

POLICY_ID = "baseline_strategy_v1"


def _stint_cost(offset: float, rate: float, laps: int, start_wear: float = 0.0) -> float:
    """Linear-wear stint cost (reference formula; used by tests)."""
    return laps * (offset + start_wear) + rate * laps * (laps - 1) / 2.0


def dry_compounds_required(setup: RaceSetup, used: tuple[Compound, ...]) -> int:
    """The dry-compound rule: lifted once a car has used intermediates or full wets (2024)."""
    if any(c not in DRY_COMPOUNDS for c in used):
        return 0
    return setup.pit.min_distinct_dry_compounds


def weather_stop(
    setup: RaceSetup, car: CarState, wetness: float, laps_left: int
) -> Compound | None:
    """Baseline wet strategy: change tyre type when the expected gain over the next
    ``switch_horizon_laps`` laps (conditions assumed to stay as they are) beats the pit loss.
    The tyre for a return to slicks depends on the laps left. ``None`` = no weather stop."""
    from f1ar.simulation.weather import best_tyre, driver_factor, is_slick, penalty_s

    params = setup.weather
    if laps_left < 2:
        return None
    best = best_tyre(params, wetness)
    same_type = (is_slick(car.compound) and is_slick(best)) or car.compound is best
    if same_type:
        return None
    wet_skill = setup.driver(car.driver_id).profile.archetype.wet_skill
    gain = (penalty_s(params, car.compound, wetness) - penalty_s(params, best, wetness)) * (
        driver_factor(params, wet_skill)
    )
    pit_cost = setup.pit.lane_loss_s + setup.pit.standstill_median_s
    if gain * min(params.switch_horizon_laps, laps_left - 1) <= pit_cost:
        return None
    if is_slick(best):
        return (
            Compound.HARD
            if laps_left > 30
            else Compound.MEDIUM
            if laps_left > 15
            else Compound.SOFT
        )
    return best


def wear_table(
    setup: RaceSetup, driver: DriverArchetype, compound: Compound, pace_mode: PaceMode, laps: int
) -> np.ndarray:
    """``W[k]`` = expected wear after ``k`` laps on a fresh set (``k = 0..laps``)."""
    steps = [
        wear_per_lap_s(
            setup.lap_model, driver, compound=compound, pace_mode=pace_mode, tyre_age_laps=a
        )
        for a in range(laps)
    ]
    return np.concatenate(([0.0], np.cumsum(steps)))


def fresh_stint_costs(offset: float, wear: np.ndarray) -> np.ndarray:
    """``f[n]`` = cost of ``n`` laps on a fresh set (``n = 0..len(wear) - 1``)."""
    n = np.arange(len(wear))
    return n * offset + np.concatenate(([0.0], np.cumsum(wear[:-1])))


def best_two_stints(f2: np.ndarray, f3: np.ndarray, laps: int) -> tuple[np.ndarray, np.ndarray]:
    """For every budget ``L = 0..laps``: the cheapest ``f2[b] + f3[L - b]`` with both stints
    >= 1 lap, and the ``b`` that achieves it (``inf`` / -1 where ``L < 2``).

    Both sequences are convex, so their min-plus convolution merges the marginal costs in
    increasing order; on ties the second stint's step is taken first (shorter first stint).
    """
    cost = np.full(laps + 1, np.inf)
    first = np.full(laps + 1, -1)
    if laps < 2:
        return cost, first
    d2 = np.diff(f2[1:laps])  # step from b to b+1 laps, b = 1..laps-2
    d3 = np.diff(f3[1:laps])
    steps = sorted(
        [(float(v), 0, i) for i, v in enumerate(d3)] + [(float(v), 1, i) for i, v in enumerate(d2)]
    )
    total = float(f2[1] + f3[1])
    b = 1
    cost[2], first[2] = total, 1
    for k, (v, src, _) in enumerate(steps[: laps - 2], start=3):
        total += v
        b += src
        cost[k], first[k] = total, b
    return cost, first


def plan_stops(
    setup: RaceSetup,
    driver: DriverArchetype,
    *,
    laps_completed: int,
    compound: Compound,
    tyre_wear_s: float,
    pace_mode: PaceMode,
    compounds_used: tuple[Compound, ...],
    tyre_age_laps: int = 0,
) -> tuple[PlannedStop, ...]:
    """Cheapest legal plan for the rest of the race, starting with the lap now beginning.

    The lap now beginning (``laps_completed + 1``) is driven on the current tyres, which are
    ``tyre_age_laps`` old and carry ``tyre_wear_s``. A stop at lap ``k`` means boxing at the
    end of lap ``k``; nobody stops at the end of the final lap.
    """
    total = setup.track.total_laps
    remaining = total - laps_completed
    params = setup.lap_model
    pit_cost = setup.pit.lane_loss_s + setup.pit.standstill_median_s
    used = set(compounds_used) | {compound}
    need = dry_compounds_required(setup, tuple(used))
    if remaining <= 0 or compound not in DRY_COMPOUNDS:
        return ()  # on wet-weather tyres the weather rule decides (engine, D-034)

    def legal(*later: Compound) -> bool:
        return len(used | set(later)) >= need

    # the current set: lap k of the rest (k = 0..) carries tyre_wear_s plus the curve's growth
    # from its current age (the wear so far may come from other pace modes, so it's kept as is)
    cur = wear_table(setup, driver, compound, pace_mode, tyre_age_laps + remaining)
    growth = cur[tyre_age_laps : tyre_age_laps + remaining] - cur[tyre_age_laps]
    per_lap = params.compounds[compound].offset_s + tyre_wear_s + growth
    first = np.concatenate(([0.0], np.cumsum(per_lap)))  # first[a] = cost of a laps
    fresh = {
        c: fresh_stint_costs(
            params.compounds[c].offset_s, wear_table(setup, driver, c, pace_mode, remaining)
        )
        for c in DRY_COMPOUNDS
    }

    # (cost, number of stops, plan): ties go to fewer stops, then to the enumeration order.
    best: tuple[float, int, tuple[PlannedStop, ...]] | None = None

    def consider(cost: float, plan: tuple[PlannedStop, ...]) -> None:
        nonlocal best
        if best is None or (cost, len(plan)) < (best[0], best[1]):
            best = (cost, len(plan), plan)

    if legal():
        consider(float(first[remaining]), ())

    a = np.arange(1, remaining)
    for c2 in DRY_COMPOUNDS:
        if not legal(c2) or remaining < 2:
            continue
        costs = first[a] + pit_cost + fresh[c2][remaining - a]
        i = int(np.argmin(costs))
        consider(float(costs[i]), (PlannedStop(lap=laps_completed + int(a[i]), compound=c2),))

    if remaining >= 3:
        a2 = np.arange(1, remaining - 1)
        for c2, c3 in product(DRY_COMPOUNDS, DRY_COMPOUNDS):
            if not legal(c2, c3):
                continue
            pair, split = best_two_stints(fresh[c2], fresh[c3], remaining - 1)
            costs = first[a2] + 2 * pit_cost + pair[remaining - a2]
            i = int(np.argmin(costs))
            lap1 = int(a2[i])
            b = int(split[remaining - lap1])
            consider(
                float(costs[i]),
                (
                    PlannedStop(lap=laps_completed + lap1, compound=c2),
                    PlannedStop(lap=laps_completed + lap1 + b, compound=c3),
                ),
            )

    return () if best is None else best[2]


def plan_for_car(setup: RaceSetup, car: CarState) -> tuple[PlannedStop, ...]:
    """Plan for ``car`` as it stands at the start of its current lap."""
    return plan_stops(
        setup,
        setup.driver(car.driver_id).profile.archetype,
        laps_completed=car.laps_completed,
        compound=car.compound,
        tyre_wear_s=car.tyre_wear_s,
        pace_mode=car.pace_mode,
        compounds_used=car.compounds_used,
        tyre_age_laps=car.tyre_age_laps,
    )
