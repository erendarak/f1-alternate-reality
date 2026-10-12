"""Baseline team strategy: the cheapest 0/1/2-stop plan under the expected lap model (plan §9.6).

This is the reference every learned strategist must beat. It is deliberately simple:

- It plans with the car's *expected* lap times (no noise), using the driver's own tyre wear
  rate (``tire_management`` and the current pace mode). Two drivers in identical cars can
  therefore pit on different laps.
- Only terms that differ between plans count: compound offset, accumulated wear and pit
  loss. Fuel, skill and car pace add the same time to every plan.
- The dry-tyre rule (``PitParams.min_distinct_dry_compounds``) is a hard constraint.
- It doesn't know about traffic, safety cars or rivals. Those arrive in later M2 slices;
  until then this is an "empty track" optimum.

Stint cost with linear wear, ``n`` laps on compound ``c`` starting at accumulated wear ``w0``:
``n * (offset_c + w0) + rate_c * n (n - 1) / 2``. It's convex in ``n``, so the best split of a
fixed lap budget between two fresh stints is at the floor or ceiling of the continuous
optimum. One plan therefore costs O(laps x compound pairs), cheap enough to replan after
every stop.
"""

import math
from itertools import product

from f1ar.domain.driver_profiles import DriverArchetype
from f1ar.domain.race_state import (
    CarState,
    Compound,
    PaceMode,
    PlannedStop,
    RaceSetup,
)
from f1ar.simulation.lap_model import wear_per_lap_s

POLICY_ID = "baseline_strategy_v1"


def _stint_cost(offset: float, rate: float, laps: int, start_wear: float = 0.0) -> float:
    return laps * (offset + start_wear) + rate * laps * (laps - 1) / 2.0


def plan_stops(
    setup: RaceSetup,
    driver: DriverArchetype,
    *,
    laps_completed: int,
    compound: Compound,
    tyre_wear_s: float,
    pace_mode: PaceMode,
    compounds_used: tuple[Compound, ...],
) -> tuple[PlannedStop, ...]:
    """Cheapest legal plan for the rest of the race, starting with the lap now beginning.

    The lap now beginning (``laps_completed + 1``) is driven on the current tyres. A stop at
    lap ``k`` means boxing at the end of lap ``k``; nobody stops at the end of the final lap.
    """
    total = setup.track.total_laps
    remaining = total - laps_completed
    params = setup.lap_model
    pit_cost = setup.pit.lane_loss_s + setup.pit.standstill_median_s
    need = setup.pit.min_distinct_dry_compounds
    used = set(compounds_used) | {compound}

    def offset(c: Compound) -> float:
        return params.compounds[c].offset_s

    def rate(c: Compound) -> float:
        return wear_per_lap_s(params, driver, compound=c, pace_mode=pace_mode)

    def legal(*later: Compound) -> bool:
        return len(used | set(later)) >= need

    def first(laps: int) -> float:
        return _stint_cost(offset(compound), rate(compound), laps, tyre_wear_s)

    # (cost, number of stops, plan): ties go to fewer stops, then to the enumeration order.
    best: tuple[float, int, tuple[PlannedStop, ...]] | None = None

    def consider(cost: float, plan: tuple[PlannedStop, ...]) -> None:
        nonlocal best
        if best is None or (cost, len(plan)) < (best[0], best[1]):
            best = (cost, len(plan), plan)

    if legal():
        consider(first(remaining), ())

    for c2 in Compound:
        if not legal(c2):
            continue
        for a in range(1, remaining):
            cost = first(a) + pit_cost + _stint_cost(offset(c2), rate(c2), remaining - a)
            consider(cost, (PlannedStop(lap=laps_completed + a, compound=c2),))

    for c2, c3 in product(Compound, Compound):
        if not legal(c2, c3):
            continue
        for a in range(1, remaining - 1):
            b = _best_split(offset(c2), rate(c2), offset(c3), rate(c3), remaining - a)
            cost = (
                first(a)
                + 2 * pit_cost
                + _stint_cost(offset(c2), rate(c2), b)
                + _stint_cost(offset(c3), rate(c3), remaining - a - b)
            )
            consider(
                cost,
                (
                    PlannedStop(lap=laps_completed + a, compound=c2),
                    PlannedStop(lap=laps_completed + a + b, compound=c3),
                ),
            )

    return () if best is None else best[2]


def _best_split(off2: float, r2: float, off3: float, r3: float, laps: int) -> int:
    """Laps of the first of two fresh stints sharing ``laps`` (each >= 1), minimising cost."""

    def cost(b: int) -> float:
        return _stint_cost(off2, r2, b) + _stint_cost(off3, r3, laps - b)

    if r2 + r3 <= 0.0:
        candidates = [1, laps - 1]
    else:
        # d/db = 0  ->  off2 + r2 (b - 1/2) = off3 + r3 (laps - b - 1/2)
        b_star = (off3 - off2 + r3 * (laps - 0.5) + r2 * 0.5) / (r2 + r3)
        candidates = [math.floor(b_star), math.ceil(b_star)]
    clamped = sorted({min(max(b, 1), laps - 1) for b in candidates})
    return min(clamped, key=cost)


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
    )
