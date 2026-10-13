"""Baseline strategists the learned AI must beat (M5 slice 2, plan §9.6, D-039).

Every strategist decides for one AI car at its decision points (``training.session``):

- **B0 ``HistoricalPlan``**: the stops the real driver made in the real 2024 race (stop lap and
  new compound), replayed as a fixed plan. A *reference*, not a fair opponent: it was chosen for
  a different race than the simulated one.
- **B1** is the engine's rule planner itself (``baseline_strategy``): the car simply stays under
  ``StrategyController.BASELINE`` (no strategist object).
- **B2 ``Enumerator``**: the best plan for every legal compound sequence with 0-3 stops on the
  car's own expected lap model (an empty track, like B1, but 3 stops too), then the best few
  re-ranked by simulating the whole race in imagined futures (traffic, safety cars), choosing
  the best mean score. Fixed after the start, except that it keeps the plan.
- **B3 ``MonteCarlo``** (MPC): at each decision, every legal option (stay out, or stop for each
  compound) is played out in a few imagined futures with the rule planner driving afterwards;
  the best mean score wins. Pace stays BALANCED (B3 doesn't optimise pace). Pruned for speed:
  on a normal lap end it doesn't deliberate while the tyres are younger than
  ``min_tyre_age`` laps, and on a dry track it doesn't consider wet tyres.
- **``RandomStrategist``**: uniformly random legal actions (lower bound).

Look-ahead uses the engine but never the real future (``training.rollouts``).
"""

import statistics
from collections.abc import Sequence
from itertools import product
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from f1ar.domain.race_state import DRY_COMPOUNDS, Compound, PaceMode, PlannedStop, RaceSetup
from f1ar.policy.baseline_strategy import dry_compounds_required, fresh_stint_costs, wear_table
from f1ar.simulation import lightweight_engine as engine
from f1ar.training.actions import PIT_OPTIONS, encode
from f1ar.training.rollouts import imagined_seed, imagined_world, rollout_reward
from f1ar.training.session import Decision, RaceSession

STAY = encode(None, PaceMode.BALANCED)
MIN_STINT = 5
"""Enumerator: no stint shorter than this many laps (editorial)."""
DRY_TRACK = 0.3
"""MPC pruning: below this wetness and without rain, wet tyres aren't considered."""


class Strategist(Protocol):
    name: str

    def start(self, session: RaceSession, driver_id: str) -> None: ...

    def decide(self, session: RaceSession, decision: Decision) -> int: ...


def _stop_or_stay(decision: Decision, plan: Sequence[PlannedStop]) -> int:
    for stop in plan:
        if stop.lap == decision.lap:
            action = encode(stop.compound, PaceMode.BALANCED)
            return action if decision.mask[action] else STAY
    return STAY


class FixedPlan:
    """Follows a stop plan: stop at the end of each planned lap for its compound."""

    def __init__(self, plan: Sequence[PlannedStop], name: str = "fixed_plan") -> None:
        self.plan = tuple(plan)
        self.name = name

    def start(self, session: RaceSession, driver_id: str) -> None:
        pass

    def decide(self, session: RaceSession, decision: Decision) -> int:
        return _stop_or_stay(decision, self.plan)


def historical_plan(data_dir: Path, driver_code: str) -> tuple[PlannedStop, ...]:
    """The real driver's stops (pit entries outside red flags) and the compounds after them."""
    from f1ar.calibration.reference_metrics import red_flag_periods

    laps = pd.read_parquet(data_dir / "laps.parquet")
    red = red_flag_periods(pd.read_parquet(data_dir / "track_status.parquet"))
    mine = sorted(
        (r for r in laps.to_dict("records") if r["driver_code"] == driver_code),
        key=lambda r: int(r["lap_number"]),
    )
    by_lap = {int(r["lap_number"]): r for r in mine}
    plan = []
    for r in mine:
        if pd.isna(r["pit_in_time_s"]) or any(a <= float(r["pit_in_time_s"]) < b for a, b in red):
            continue
        nxt = by_lap.get(int(r["lap_number"]) + 1)
        if nxt is None or pd.isna(nxt["compound"]):
            continue
        try:
            plan.append(PlannedStop(lap=int(r["lap_number"]), compound=Compound(nxt["compound"])))
        except ValueError:
            continue  # an unknown compound label
    return tuple(plan)


# --- B2 --------------------------------------------------------------------------------------


def candidate_plans(
    setup: RaceSetup, driver_id: str, *, max_stops: int = 3
) -> list[tuple[float, tuple[PlannedStop, ...]]]:
    """The cheapest plan for every legal dry compound sequence (0..max_stops stops), with its
    expected cost on an empty track, cheapest first."""
    entry = setup.driver(driver_id)
    driver = entry.profile.archetype
    laps = setup.track.total_laps
    pit = setup.pit.lane_loss_s + setup.pit.standstill_median_s
    params = setup.lap_model
    fresh = {
        c: fresh_stint_costs(
            params.compounds[c].offset_s, wear_table(setup, driver, c, PaceMode.BALANCED, laps)
        )
        for c in DRY_COMPOUNDS
    }
    start = entry.start_compound
    if start not in DRY_COMPOUNDS:
        return []
    out = []
    for stops in range(max_stops + 1):
        for later in product(DRY_COMPOUNDS, repeat=stops):
            sequence = (start, *later)
            if len(set(sequence)) < dry_compounds_required(setup, sequence):
                continue
            best = _best_split(sequence, fresh, laps, pit)
            if best is not None:
                out.append(best)
    out.sort(key=lambda x: (x[0], len(x[1])))
    return out


def _best_split(
    sequence: tuple[Compound, ...], fresh: dict[Compound, np.ndarray], laps: int, pit: float
) -> tuple[float, tuple[PlannedStop, ...]] | None:
    """Cheapest stop laps for ``sequence`` (dynamic programming over stint lengths)."""
    k = len(sequence)
    if k * MIN_STINT > laps:
        return None
    # cost[i][l] = cheapest way to cover the first l laps with the first i stints
    inf = float("inf")
    cost = [[inf] * (laps + 1) for _ in range(k + 1)]
    back = [[-1] * (laps + 1) for _ in range(k + 1)]
    cost[0][0] = 0.0
    for i, c in enumerate(sequence, start=1):
        f = fresh[c]
        for end in range(i * MIN_STINT, laps + 1):
            for begin in range((i - 1) * MIN_STINT, end - MIN_STINT + 1):
                prev = cost[i - 1][begin]
                if prev == inf:
                    continue
                v = prev + float(f[end - begin]) + (pit if i > 1 else 0.0)
                if v < cost[i][end]:
                    cost[i][end], back[i][end] = v, begin
    if cost[k][laps] == inf:
        return None
    ends, end = [], laps
    for i in range(k, 0, -1):
        end = back[i][end]
        ends.append(end)
    stops = sorted(ends)[1:]  # stint boundaries after the first stint start (lap 0)
    plan = tuple(
        PlannedStop(lap=lap, compound=c) for lap, c in zip(stops, sequence[1:], strict=True)
    )
    return cost[k][laps], plan


class Enumerator:
    """B2: enumerate plans, re-rank the cheapest ``shortlist`` by simulated races."""

    name = "enumerator"

    def __init__(self, *, shortlist: int = 6, futures: int = 3) -> None:
        self.shortlist = shortlist
        self.futures = futures
        self.plan: tuple[PlannedStop, ...] = ()
        self.scores: list[tuple[float, tuple[PlannedStop, ...]]] = []

    def start(self, session: RaceSession, driver_id: str) -> None:
        candidates = candidate_plans(session.setup, driver_id)[: self.shortlist]
        if not candidates:
            self.plan = ()
            return
        real_seed = session.state.rng.seed
        scored = []
        for _, plan in candidates:
            values = [
                _simulate_plan(session, driver_id, plan, imagined_seed(real_seed, driver_id, 0, k))
                for k in range(self.futures)
            ]
            scored.append((statistics.fmean(values), plan))
        scored.sort(key=lambda x: (-x[0], len(x[1])))
        self.scores = scored
        self.plan = scored[0][1]

    def decide(self, session: RaceSession, decision: Decision) -> int:
        return _stop_or_stay(decision, self.plan)


def _simulate_plan(
    session: RaceSession, driver_id: str, plan: tuple[PlannedStop, ...], seed: int
) -> float:
    """The car's score following ``plan`` from the start, in imagined future ``seed``."""
    start = engine.create_world(session.setup, seed=session.state.rng.seed, world_id="imagine")
    imagined = imagined_world(start, seed)
    trial = RaceSession(
        imagined.setup,
        seed=seed,
        agents=[driver_id],
        scoring=session.scoring,
        context=session.context,
        regulations=session.regulations,
        policy_id="enumerator",
    )
    follower = FixedPlan(plan)
    while (d := trial.next_decision()) is not None:
        trial.act(d.driver_id, follower.decide(trial, d))
    return trial.rewards()[driver_id]


# --- B3 --------------------------------------------------------------------------------------


class MonteCarlo:
    """B3: one-step look-ahead over imagined futures, the rule planner as the base policy."""

    name = "mpc"

    def __init__(self, *, futures: int = 3, min_tyre_age: int = 6) -> None:
        self.futures = futures
        self.min_tyre_age = min_tyre_age
        self.deliberations = 0
        self.rollouts = 0

    def start(self, session: RaceSession, driver_id: str) -> None:
        pass

    def decide(self, session: RaceSession, decision: Decision) -> int:
        state = session.state
        car = state.car(decision.driver_id)
        dry = state.weather.wetness < DRY_TRACK and not state.weather.raining
        pits = [
            encode(c, PaceMode.BALANCED)
            for c in PIT_OPTIONS[1:]
            if decision.mask[encode(c, PaceMode.BALANCED)]
            and not (dry and c in (Compound.INTERMEDIATE, Compound.WET))
        ]
        if not pits:
            return STAY
        if decision.reason == "LAP_END" and car.tyre_age_laps < self.min_tyre_age:
            return STAY
        self.deliberations += 1
        seeds = [
            imagined_seed(state.rng.seed, decision.driver_id, decision.lap, k)
            for k in range(self.futures)
        ]
        best, best_value = STAY, -float("inf")
        for option in (STAY, *pits):  # stay out first: it wins ties
            value = statistics.fmean(
                rollout_reward(
                    state, decision.driver_id, option, s, session.scoring, session.actor_id
                )
                for s in seeds
            )
            self.rollouts += len(seeds)
            if value > best_value + 1e-12:
                best, best_value = option, value
        return best


class RandomStrategist:
    name = "random"

    def __init__(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    def start(self, session: RaceSession, driver_id: str) -> None:
        pass

    def decide(self, session: RaceSession, decision: Decision) -> int:
        return int(self.rng.choice(np.flatnonzero(decision.mask)))
