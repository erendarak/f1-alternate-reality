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

**Tyre life (W1, D-041).** When the compounds have a life (data-backed races), a stint's cost
also includes how the driver has to drive it: a stint longer than the set's life at BALANCED
pace is managed (a share of CONSERVE laps, each slower but using less life), and beyond what
full management allows the cliff is paid; a stint shorter than the life leaves life to spend
on PUSH laps (faster). ``stint_costs`` gives this in closed form per stint length; with the
pace modes' trade-offs ordered (a lap of life saved by managing costs more than a lap of life
spent pushing gains) the cost stays convex in the length, so the split below still holds.
``baseline_pace_mode`` then drives the plan lap by lap: manage when the set wouldn't last to
its planned stop at BALANCED pace, push when there's life to spare. This is what makes two-
and three-stop races (short stints, pushed) compete with one-stoppers (long, managed); each
stop must also beat ``STOP_MARGIN_S`` (track position, which the empty-track plan ignores).

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
from f1ar.simulation.lap_model import life_use_per_lap, wear_per_lap_s

POLICY_ID = "baseline_strategy_v2"
STOP_MARGIN_S = 6.5
"""W1: what each stop must save on an empty track to be worth giving up track position
(McLaren's 2024 strategy review: an extra stop must be "at least six or seven seconds
quicker" on paper). Only with tyre life (data-backed races); sourced, not fitted."""


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


def management_cost(
    setup: RaceSetup,
    driver: DriverArchetype,
    compound: Compound,
    laps: np.ndarray,
    life_left: float,
) -> np.ndarray:
    """Lap time a stint of ``laps`` laps costs (or gains, < 0) through tyre management, from a
    set with ``life_left`` of its life (W1). 0 without a life model.

    ``u`` = life one BALANCED lap uses. If ``u n`` fits in ``life_left``, the spare life is
    spent on PUSH laps (share ``(left / (u n) - 1) / (m_push - 1)``, at most all laps);
    otherwise a share ``(1 - left / (u n)) / (1 - m_conserve)`` of the laps is managed (each
    extra lap then costs ``offset / (1 - m)``); beyond full management that marginal cost goes
    on and the cliff is paid for every lap past the life."""
    params = setup.lap_model
    u = life_use_per_lap(params, driver, compound=compound, pace_mode=PaceMode.BALANCED)
    n = laps.astype(float)
    if u <= 0.0:
        return np.zeros_like(n)
    push = params.pace_modes[PaceMode.PUSH]
    cons = params.pace_modes[PaceMode.CONSERVE]
    need = u * n
    out = np.zeros_like(n)
    spare = need <= life_left
    with np.errstate(divide="ignore", invalid="ignore"):
        if push.wear_multiplier > 1.0:
            fp = np.clip((life_left / need - 1.0) / (push.wear_multiplier - 1.0), 0.0, 1.0)
            out = np.where(spare & (n > 0), fp * n * push.offset_s, out)
        # managed share: life used n u (1 - fc (1 - m)) = what's left
        fc = (1.0 - life_left / need) / (1.0 - cons.wear_multiplier)
        managed = ~spare & (fc <= 1.0) & (life_left > 0)
        out = np.where(managed, fc * n * cons.offset_s, out)
        # beyond what full management can stretch: every further lap costs at least the
        # managed stint's last marginal lap, plus the cliff for every lap past the life (this
        # keeps the cost convex; a slope that dropped here broke the planner's split, D-041)
        m = cons.wear_multiplier
        if life_left > 0:
            n_b = life_left / (u * m)  # the longest fully managed stint
            slope_b = cons.offset_s / (1.0 - m)  # the managed stint's marginal lap
            over = np.maximum(0.0, n - n_b)
            beyond = n_b * cons.offset_s + slope_b * over + params.cliff_s_per_lap * over**2 / 2.0
        else:  # already past the life: full management and the cliff from here on
            over_start = -life_left / (u * m)
            beyond = (
                n * cons.offset_s
                + params.cliff_s_per_lap * ((over_start + n) ** 2 - over_start**2) / 2.0
            )
        out = np.where(~spare & ~managed, beyond, out)
    return np.asarray(out)


def stint_costs(
    setup: RaceSetup,
    driver: DriverArchetype,
    compound: Compound,
    laps: int,
    *,
    tyre_age_laps: int = 0,
    tyre_wear_s: float = 0.0,
    life_used: float = 0.0,
    pace_mode: PaceMode = PaceMode.BALANCED,
) -> np.ndarray:
    """``f[n]`` = cost of the next ``n`` laps (``n = 0..laps``) on this set: compound offset,
    wear carried and growing from ``tyre_age_laps``, and (W1) tyre management."""
    params = setup.lap_model
    table = wear_table(setup, driver, compound, pace_mode, tyre_age_laps + laps)
    growth = table[tyre_age_laps : tyre_age_laps + laps] - table[tyre_age_laps]
    per_lap = params.compounds[compound].offset_s + tyre_wear_s + growth
    f = np.concatenate(([0.0], np.cumsum(per_lap)))
    out: np.ndarray = f + management_cost(
        setup, driver, compound, np.arange(laps + 1), 1.0 - life_used
    )
    return out


def has_tyre_life(setup: RaceSetup) -> bool:
    return any(setup.lap_model.compounds[c].life_laps is not None for c in DRY_COMPOUNDS)


def baseline_pace_mode(setup: RaceSetup, car: CarState) -> PaceMode:
    """The baseline's pace for the lap now beginning (W1): manage when the set wouldn't reach its
    planned stop (or the flag) at BALANCED pace; push when it would still get there after a
    pushed lap; else BALANCED. Without a tyre life the car keeps its mode."""
    params = setup.lap_model
    driver = setup.driver(car.driver_id).profile.archetype
    u = life_use_per_lap(params, driver, compound=car.compound, pace_mode=PaceMode.BALANCED)
    if u <= 0.0:
        return car.pace_mode
    end = car.planned_stops[0].lap if car.planned_stops else setup.track.total_laps
    laps_left = max(1, end - car.laps_completed)
    left = 1.0 - car.tyre_life_used
    if left < u * laps_left:
        return PaceMode.CONSERVE
    u_push = life_use_per_lap(params, driver, compound=car.compound, pace_mode=PaceMode.PUSH)
    if left - u_push >= u * (laps_left - 1):
        return PaceMode.PUSH
    return PaceMode.BALANCED


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
    life_used: float = 0.0,
) -> tuple[PlannedStop, ...]:
    """Cheapest legal plan for the rest of the race (see ``best_plan``)."""
    best = best_plan(
        setup,
        driver,
        laps_completed=laps_completed,
        compound=compound,
        tyre_wear_s=tyre_wear_s,
        pace_mode=pace_mode,
        compounds_used=compounds_used,
        tyre_age_laps=tyre_age_laps,
        life_used=life_used,
    )
    return () if best is None else best[1]


def best_plan(
    setup: RaceSetup,
    driver: DriverArchetype,
    *,
    laps_completed: int,
    compound: Compound,
    tyre_wear_s: float,
    pace_mode: PaceMode,
    compounds_used: tuple[Compound, ...],
    tyre_age_laps: int = 0,
    life_used: float = 0.0,
) -> tuple[float, tuple[PlannedStop, ...]] | None:
    """Cheapest legal plan for the rest of the race and its cost (time on an empty track that
    differs between plans, stop margins included), starting with the lap now beginning; None
    if no legal plan (or on wet-weather tyres).

    The lap now beginning (``laps_completed + 1``) is driven on the current tyres, which are
    ``tyre_age_laps`` old and carry ``tyre_wear_s``. A stop at lap ``k`` means boxing at the
    end of lap ``k``; nobody stops at the end of the final lap.
    """
    total = setup.track.total_laps
    remaining = total - laps_completed
    pit_cost = setup.pit.lane_loss_s + setup.pit.standstill_median_s
    used = set(compounds_used) | {compound}
    need = dry_compounds_required(setup, tuple(used))
    if remaining <= 0 or compound not in DRY_COMPOUNDS:
        return None  # on wet-weather tyres the weather rule decides (engine, D-034)

    def legal(*later: Compound) -> bool:
        return len(used | set(later)) >= need

    # W1: with tyre life the plan assumes BALANCED wear plus management (the baseline then
    # manages / pushes lap by lap); without it, the car's current mode as before
    if has_tyre_life(setup):
        pace_mode = PaceMode.BALANCED
    # the current set: lap k of the rest (k = 0..) carries tyre_wear_s plus the curve's growth
    # from its current age (the wear so far may come from other pace modes, so it's kept as is)
    first = stint_costs(  # first[a] = cost of a laps
        setup,
        driver,
        compound,
        remaining,
        tyre_age_laps=tyre_age_laps,
        tyre_wear_s=tyre_wear_s,
        life_used=life_used,
        pace_mode=pace_mode,
    )
    fresh = {
        c: stint_costs(setup, driver, c, remaining, pace_mode=pace_mode) for c in DRY_COMPOUNDS
    }

    # (cost, number of stops, plan): ties go to fewer stops, then to the enumeration order.
    best: tuple[float, int, tuple[PlannedStop, ...]] | None = None
    margin = STOP_MARGIN_S if has_tyre_life(setup) else 0.0

    def consider(cost: float, plan: tuple[PlannedStop, ...]) -> None:
        nonlocal best
        cost += margin * len(plan)
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

    return None if best is None else (best[0], best[2])


def free_change_compound(setup: RaceSetup, car: CarState, wetness: float) -> Compound:
    """W4 red flag: the compound the baseline fits when a new set is free now: the right tyre
    type for the conditions, and among dry tyres the one with the cheapest rest of the race."""
    from f1ar.simulation.weather import best_tyre, is_slick

    wanted = best_tyre(setup.weather, wetness)
    if not is_slick(wanted):
        return wanted
    driver = setup.driver(car.driver_id).profile.archetype
    options = []
    for c in DRY_COMPOUNDS:
        used = car.compounds_used if c in car.compounds_used else (*car.compounds_used, c)
        best = best_plan(
            setup,
            driver,
            laps_completed=car.laps_completed,
            compound=c,
            tyre_wear_s=0.0,
            pace_mode=PaceMode.BALANCED,
            compounds_used=used,
        )
        if best is not None:
            options.append((best[0], DRY_COMPOUNDS.index(c), c))
    return min(options)[2] if options else car.compound


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
        life_used=car.tyre_life_used,
    )
