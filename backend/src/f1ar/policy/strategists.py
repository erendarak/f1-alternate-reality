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
  the best mean score wins. Pace follows the rules (``rule_pace``; B3 doesn't optimise pace).
  Pruned for speed: on a normal lap end it doesn't deliberate while the tyres are younger than
  ``min_tyre_age`` laps, and on a dry track it doesn't consider wet tyres.
- **``RandomStrategist``**: uniformly random legal actions (lower bound).

Look-ahead uses the engine but never the real future (``training.rollouts``).

**Pace (M5 slice 3b):** since W1 the rule planner manages its tyres lap by lap
(``baseline_pace_mode``: CONSERVE when the set wouldn't reach its planned stop, PUSH when it
would anyway). Every strategist here that follows a plan drives the same way (``rule_pace``),
so plans are compared as the rules would drive them, not at a constant BALANCED pace.
"""

import statistics
from collections.abc import Sequence
from itertools import product
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from f1ar.domain.race_state import (
    DRY_COMPOUNDS,
    CarState,
    Compound,
    PaceMode,
    PlannedStop,
    RacecraftIntent,
    RacePhase,
    RaceSetup,
)
from f1ar.policy.baseline_strategy import (
    baseline_pace_mode,
    dry_compounds_required,
    plan_for_car,
    stint_costs,
)
from f1ar.simulation import lightweight_engine as engine
from f1ar.simulation.lap_model import life_use_per_lap
from f1ar.training.actions import PER_PIT, PIT_OPTIONS, encode
from f1ar.training.features import schema_for_inputs
from f1ar.training.policy_registry import NumpyPolicy
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


def rule_pace(
    setup: RaceSetup, car: CarState, plan: Sequence[PlannedStop], stop: Compound | None
) -> PaceMode:
    """The pace the rule planner would drive on the lap after this decision (W1).

    A decision comes right after the car crossed the line; a pace chosen now applies from the
    next lap. So the car is projected to the start of that lap (one more lap on this set, or a
    new set if it stops at the end of this one) with the rest of ``plan`` ahead, and the
    baseline's rule is asked (``baseline_pace_mode``)."""
    if car.compound not in DRY_COMPOUNDS or (stop is not None and stop not in DRY_COMPOUNDS):
        return PaceMode.BALANCED
    lap = car.laps_completed + 1
    rest = tuple(p for p in plan if p.lap > lap)
    if stop is not None:
        nxt = car.model_copy(
            update={
                "compound": stop,
                "tyre_age_laps": 0,
                "tyre_life_used": 0.0,
                "tyre_wear_s": 0.0,
                "laps_completed": lap,
                "planned_stops": rest,
            }
        )
    else:
        driver = setup.driver(car.driver_id).profile.archetype
        used = car.tyre_life_used + life_use_per_lap(
            setup.lap_model, driver, compound=car.compound, pace_mode=car.lap_pace_mode
        )
        nxt = car.model_copy(
            update={
                "tyre_life_used": used,
                "tyre_age_laps": car.tyre_age_laps + 1,
                "laps_completed": lap,
                "planned_stops": rest,
            }
        )
    return baseline_pace_mode(setup, nxt)


def _with_pace(
    session: RaceSession, decision: Decision, plan: Sequence[PlannedStop], stop: Compound | None
) -> int:
    """The action (stop or stay) at the rule pace; BALANCED if that pace isn't legal."""
    car = session.state.car(decision.driver_id)
    action = encode(stop, rule_pace(session.state.setup, car, plan, stop))
    if decision.mask[action]:
        return action
    fallback = encode(stop, PaceMode.BALANCED)
    return fallback if decision.mask[fallback] else STAY


def _stop_or_stay(session: RaceSession, decision: Decision, plan: Sequence[PlannedStop]) -> int:
    for stop in plan:
        if stop.lap == decision.lap:
            return _with_pace(session, decision, plan, stop.compound)
    return _with_pace(session, decision, plan, None)


class FixedPlan:
    """Follows a stop plan: stop at the end of each planned lap for its compound."""

    def __init__(self, plan: Sequence[PlannedStop], name: str = "fixed_plan") -> None:
        self.plan = tuple(plan)
        self.name = name

    def start(self, session: RaceSession, driver_id: str) -> None:
        pass

    def decide(self, session: RaceSession, decision: Decision) -> int:
        return _stop_or_stay(session, decision, self.plan)


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
    # W1-aware stint costs (tyre life, management, cliff), as the rule planner sees them
    fresh = {
        c: stint_costs(setup, driver, c, laps, pace_mode=PaceMode.BALANCED) for c in DRY_COMPOUNDS
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
        return _stop_or_stay(session, decision, self.plan)


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
        plan = plan_for_car(state.setup, car)
        stay = _with_pace(session, decision, plan, None)
        pits = [
            _with_pace(session, decision, plan, c)
            for c in PIT_OPTIONS[1:]
            if decision.mask[encode(c, PaceMode.BALANCED)]
            and not (dry and c in (Compound.INTERMEDIATE, Compound.WET))
        ]
        pits = [a for a in pits if a != STAY]
        if not pits:
            return stay
        if decision.reason == "LAP_END" and car.tyre_age_laps < self.min_tyre_age:
            return stay
        self.deliberations += 1
        seeds = [
            imagined_seed(state.rng.seed, decision.driver_id, decision.lap, k)
            for k in range(self.futures)
        ]
        best, best_value = stay, -float("inf")
        for option in (stay, *pits):  # stay out first: it wins ties
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


class LearnedStrategist:
    """A trained policy (M5 slice 3, D-040): the most likely legal action of the pinned actor
    network (``training.policy_registry``), deterministic.

    ``sample_seed``: sample from the policy's distribution instead, with a generator seeded from
    ``sample_seed`` and the race seed (reproducible: same race, same draws).

    ``pace``: force one pace mode (an ablation that separates what the policy gains from pit
    timing and compounds from what it gains from the editorial PUSH/CONSERVE effects). The pit
    choice is then the one with the highest probability summed over pace modes."""

    def __init__(
        self,
        policy: NumpyPolicy,
        name: str,
        pace: PaceMode | None = None,
        sample_seed: int | None = None,
    ) -> None:
        self.policy = policy
        self.observation_schema = schema_for_inputs(policy.n_inputs)
        self.name = name if pace is None else f"{name}+{pace.value.lower()}"
        if sample_seed is not None:
            self.name += "~sampled"
        self.pace = pace
        self.sample_seed = sample_seed
        self.rng: np.random.Generator | None = None

    def start(self, session: RaceSession, driver_id: str) -> None:
        if self.sample_seed is not None:
            self.rng = np.random.default_rng([self.sample_seed, session.state.rng.seed])

    def decide(self, session: RaceSession, decision: Decision) -> int:
        if self.rng is not None:
            p = self.policy.probabilities(decision.observation, decision.mask)
            return int(self.rng.choice(len(p), p=p))
        if self.pace is None:
            return self.policy.act(decision.observation, decision.mask)
        p = self.policy.probabilities(decision.observation, decision.mask)
        options = [(encode(pit, self.pace), pit) for pit in PIT_OPTIONS]
        legal = [(a, pit) for a, pit in options if decision.mask[a]]
        if not legal:
            return self.policy.act(decision.observation, decision.mask)
        by_pit = p.reshape(len(PIT_OPTIONS), PER_PIT).sum(axis=1)
        return max(legal, key=lambda o: by_pit[PIT_OPTIONS.index(o[1])])[0]


class RulesTeacher:
    """B1's rule planner expressed as the AI's actions (a teacher for imitation, D-040).

    At each decision it replans like the engine's baseline (``plan_for_car``) and asks for a stop
    when the plan stops at the end of this lap, or, under SC/VSC, when the next planned stop is
    within the neutralised pit window (the baseline's cheap stop). Pace follows the baseline's
    management rule (``rule_pace``, W1). Dry races only: the baseline's weather rule isn't
    reproduced. Not identical to B1 (the baseline replans at its own moments); how close it gets is
    measured (EXP-009 notes, tests)."""

    name = "rules_teacher"

    def start(self, session: RaceSession, driver_id: str) -> None:
        pass

    def decide(self, session: RaceSession, decision: Decision) -> int:
        state = session.state
        car = state.car(decision.driver_id)
        plan = plan_for_car(state.setup, car)
        if plan:
            nxt = plan[0]
            window = state.setup.race_control.neutralised_pit_window_laps
            neutralised = state.race_control.phase is not RacePhase.GREEN
            if nxt.lap == car.laps_completed + 1 or (
                neutralised and nxt.lap - car.laps_completed <= window
            ):
                action = _with_pace(session, decision, plan, nxt.compound)
                if action != STAY:
                    return action
        return _with_pace(session, decision, plan, None)


class RacecraftRule(RulesTeacher):
    """The rules teacher plus a hand-written racecraft rule (slice 4, D-048), a baseline for the
    new ATTACK / DEFEND lever: attack when the car ahead is within ``window_s`` and not faster
    (car pace), defend when a faster car is within ``window_s`` behind; else NORMAL."""

    name = "racecraft_rule"

    def __init__(self, window_s: float = 1.0) -> None:
        self.window_s = window_s

    def decide(self, session: RaceSession, decision: Decision) -> int:
        from f1ar.training.actions import decode
        from f1ar.training.features import _gaps, _neighbours

        pit, pace, _ = decode(super().decide(session, decision))
        state = session.state
        me = state.setup.driver(decision.driver_id)
        ahead_id, behind_id = _neighbours(state, decision.driver_id)
        ahead_s, behind_s = _gaps(state, decision.driver_id)
        intent = RacecraftIntent.NORMAL
        if (
            ahead_id is not None
            and ahead_s is not None
            and ahead_s < self.window_s
            and state.setup.driver(ahead_id).car_pace_offset_s >= me.car_pace_offset_s
        ):
            intent = RacecraftIntent.ATTACK
        elif (
            behind_id is not None
            and behind_s is not None
            and behind_s < self.window_s
            and state.setup.driver(behind_id).car_pace_offset_s < me.car_pace_offset_s
        ):
            intent = RacecraftIntent.DEFEND
        action = encode(pit, pace, intent)
        return action if decision.mask[action] else encode(pit, pace)
