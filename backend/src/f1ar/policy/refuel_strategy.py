"""Baseline strategy for the refuelling era, 2000-2009 (H1b, D-057).

A stop puts in the fuel for the next stint and nothing more, so the fuel a car carries follows
its stints: a stint of ``n`` laps starts with ``n`` laps of fuel. Its cost (time on an empty
track that differs between plans):

    tyres     the stint's tyre cost (``baseline_strategy.stint_costs``: compound, wear, W1 life)
    fuel      fuel_s_per_lap x (n + (n - 1) + ... + 1)  = fuel_s x n (n + 1) / 2
    the stop  lane loss + max(tyre change, refuelling n laps of fuel)   (after a stop)

Each term is convex in ``n``, so short stints are light and fast and every stop costs a lane
loss: the plan balances the two (2-3 stops in a typical race of the era). Up to ``MAX_STOPS``
stops, by dynamic programming over laps covered x dry compounds used (the 2007-2009 rule).

- **Mid-race** the stint on the current fuel is fixed: a car can't run past its fuel, and
  stopping early would have meant carrying weight for nothing (the engine still brings a stop
  forward under a safety car; it then refuels to the plan from there).
- **At a stop or at the start** the next stint's fuel is chosen (``free_first``), except when
  the race-fuel rule fixed the start fuel in qualifying (``DriverEntry.start_fuel_laps``).
- **2005** (no tyre changes): the tyre cost is the same set over the race; stops are fuel only.
"""

from dataclasses import dataclass

import numpy as np

from f1ar.domain.driver_profiles import DriverArchetype
from f1ar.domain.race_state import DRY_COMPOUNDS, Compound, PaceMode, PlannedStop, RaceSetup
from f1ar.policy.baseline_strategy import (
    STOP_MARGIN_S,
    dry_compounds_required,
    has_tyre_life,
    stint_costs,
)

MAX_STOPS = 4


@dataclass(frozen=True)
class RefuelPlan:
    cost: float
    first_stint_laps: int
    """Laps on the current fuel (at a stop or the start: the fuel to put in)."""
    stops: tuple[PlannedStop, ...]


def fuel_cost(setup: RaceSetup, n: np.ndarray) -> np.ndarray:
    return np.asarray(setup.lap_model.fuel_s_per_lap_remaining * n * (n + 1) / 2.0)


def best_plan_refuel(
    setup: RaceSetup,
    driver: DriverArchetype,
    *,
    laps_completed: int,
    compound: Compound,
    compounds_used: tuple[Compound, ...],
    fuel_laps: float | None,
    free_first: bool,
    at_stop: bool = False,
    tyre_age_laps: int = 0,
    tyre_wear_s: float = 0.0,
    life_used: float = 0.0,
) -> RefuelPlan | None:
    """The cheapest legal plan for the rest of the race; None on wet-weather tyres."""
    total = setup.track.total_laps
    remaining = total - laps_completed
    if remaining <= 0 or compound not in DRY_COMPOUNDS:
        return None
    pit, rules = setup.pit, setup.rules
    tyres_change = rules.tyre_changes_allowed
    used = set(compounds_used) | {compound}
    need = dry_compounds_required(setup, tuple(used)) if tyres_change else 0
    margin = STOP_MARGIN_S if has_tyre_life(setup) else 0.0
    laps = np.arange(remaining + 1, dtype=float)
    tyre_stand = pit.standstill_median_s if tyres_change else 0.0
    stop_time = pit.lane_loss_s + np.maximum(tyre_stand, pit.refuel_s_per_lap * laps) + margin

    pace = PaceMode.BALANCED
    current = stint_costs(
        setup,
        driver,
        compound,
        remaining,
        tyre_age_laps=tyre_age_laps,
        tyre_wear_s=tyre_wear_s,
        life_used=life_used,
        pace_mode=pace,
    )
    # first stint: on the current tyres and the current (or now chosen) fuel
    first = current + (fuel_cost(setup, laps) if free_first else 0.0)
    if at_stop and free_first:  # this stop's stationary time depends on the fuel put in
        first = first + np.maximum(tyre_stand, pit.refuel_s_per_lap * laps) - tyre_stand
    if free_first:
        first_options = range(1, remaining + 1)
    else:
        fixed = remaining if fuel_laps is None else max(1, min(remaining, int(fuel_laps)))
        first_options = range(fixed, fixed + 1)

    # later stints: g[c][n] = cost of a stint of n laps after a stop
    bits = {c: 1 << i for i, c in enumerate(DRY_COMPOUNDS)}
    if tyres_change:
        stint = {
            c: stint_costs(setup, driver, c, remaining, pace_mode=pace)
            + fuel_cost(setup, laps)
            + stop_time
            for c in DRY_COMPOUNDS
        }
    else:  # 2005: no tyre changes: later stints cost fuel and the stop only (the set's
        # cost over the whole rest of the race is counted once with the first stint)
        stint = {compound: fuel_cost(setup, laps) + stop_time}
    choices = list(stint)
    start_mask = sum(bits[c] for c in used if c in bits)
    inf = np.inf
    # dp[k][mask][L]: cheapest k later stints covering L laps, compounds used = mask
    dp = np.full((MAX_STOPS + 1, 8, remaining + 1), inf)
    back = np.full((MAX_STOPS + 1, 8, remaining + 1, 3), -1, dtype=int)  # prev mask, c, n
    dp[0, start_mask, 0] = 0.0
    for k in range(1, MAX_STOPS + 1):
        for mask in range(8):
            prev = dp[k - 1, mask]
            if not np.isfinite(prev).any():
                continue
            for ci, c in enumerate(choices):
                g = stint[c]
                new_mask = mask | bits[c]
                for n in range(1, remaining + 1):
                    cand = prev[: remaining + 1 - n] + g[n]
                    target = dp[k, new_mask, n:]
                    better = cand < target
                    if better.any():
                        target[better] = cand[better]
                        idx = np.nonzero(better)[0] + n
                        back[k, new_mask, idx] = (mask, ci, n)

    def legal(mask: int) -> bool:
        return bin(mask).count("1") >= need

    best: tuple[float, int, int, int, int] | None = None  # cost, stops, first, k, mask
    for a in first_options:
        rest = remaining - a
        # 2005: the current set runs the whole rest of the race whatever the fuel stops
        base = float(first[a] if tyres_change else first[a] - current[a] + current[remaining])
        if rest == 0:
            if legal(start_mask) and (best is None or (base, 0) < (best[0], best[1])):
                best = (base, 0, a, 0, start_mask)
            continue
        for k in range(1, MAX_STOPS + 1):
            for mask in range(8):
                if not legal(mask) or not np.isfinite(dp[k, mask, rest]):
                    continue
                cost = base + float(dp[k, mask, rest])
                if best is None or (cost, k) < (best[0], best[1]):
                    best = (cost, k, a, k, mask)
    if best is None:
        return None
    cost, _, a, k, mask = best
    stints: list[tuple[Compound, int]] = []
    rest = remaining - a
    while k > 0:
        prev_mask, ci, n = (int(x) for x in back[k, mask, rest])
        stints.append((choices[ci], n))
        mask, rest, k = prev_mask, rest - n, k - 1
    stints.reverse()
    plan, lap = [], laps_completed + a
    for c, n in stints:
        plan.append(PlannedStop(lap=lap, compound=c))
        lap += n
    return RefuelPlan(cost, a, tuple(plan))
