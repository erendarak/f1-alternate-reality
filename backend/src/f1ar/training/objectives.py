"""What an AI car is trying to achieve, and how a race result is scored (M5, D-037).

**Objectives** (``docs/M5_DESIGN.md`` §3): what "winning" means depends on the championship and is
the **team principal's** decision. In M5 the team principal is a small rule set (editorial,
labelled; a learned principal comes later):

- ``CHAMPIONSHIP``: a title contender in a competitive car; scores points *and* points relative to
  its title rival in this race.
- ``POINTS``: everyone else; scores championship points, with a small tie-breaker for finishing
  position so that a car outside the points still has something to gain.
- ``RESULT`` (finishing position) and ``TIME`` (race time) are for curricula and tests.

**Scoring:** each objective maps the race result to a number ``x`` roughly in [-1, 1]. The
driver's **risk utility** ``u(x)`` then encodes character (editorial, from the persona's
``risk_appetite``): risk-averse drivers (appetite < 0.5) have a concave ``u`` and avoid bad tails,
risk-seeking drivers a convex one and chase the upside. RL maximises ``E[u(x)]``, so the same
car can be driven differently by different characters. The *preference* is editorial; the
*decisions* that follow from it are learned.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np

from f1ar.domain.race_state import CarStatus, RaceSetup, WorldState
from f1ar.policy.baseline_strategy import dry_compounds_required
from f1ar.simulation.lightweight_engine import standings

POINTS_2024 = (25, 18, 15, 12, 10, 8, 6, 4, 2, 1)
"""Points for P1-P10 (2024; the fastest-lap point and sprints aren't modelled)."""
SEASON_RACES_2024 = 24
POSITION_TIEBREAK = 0.02
"""Weight of finishing position next to points (editorial: a fraction of one point)."""
RISK_CURVATURE = 4.0
"""``a = RISK_CURVATURE x (0.5 - risk_appetite)`` in the risk utility (editorial)."""
TIME_SCALE_S = 60.0
"""TIME objective: one minute behind the winner = -1."""


class Objective(StrEnum):
    TIME = "TIME"
    RESULT = "RESULT"
    POINTS = "POINTS"
    CHAMPIONSHIP = "CHAMPIONSHIP"


@dataclass(frozen=True)
class ChampionshipContext:
    """The season around this race: points so far and races left (synthetic during training,
    the world's real standings once worlds run seasons)."""

    points: Mapping[str, float]
    races_left: int
    """Including this race."""
    season_races: int = SEASON_RACES_2024
    source: str = "editorial_assumption: synthetic standings (training sampler)"

    def table(self) -> list[str]:
        """Driver ids by points, leader first (ties: id order, deterministic)."""
        return sorted(self.points, key=lambda d: (-self.points[d], d))

    def position(self, driver_id: str) -> int:
        return self.table().index(driver_id) + 1

    def gap_to_leader(self, driver_id: str) -> float:
        return max(self.points.values()) - self.points[driver_id]

    def max_points_left(self) -> float:
        return float(POINTS_2024[0] * self.races_left)


@dataclass(frozen=True)
class TeamOrder:
    """The team principal's objective for one car in one race."""

    objective: Objective
    rival: str | None = None
    """CHAMPIONSHIP: the title rival whose result counts against this car's."""
    rival_weight: float = 0.0
    reason: str = ""


def car_ranks(setup: RaceSetup) -> dict[str, int]:
    """Team rank by car pace (0 = fastest car), per driver."""
    offsets = sorted({d.car_pace_offset_s for d in setup.drivers})
    return {d.driver_id: offsets.index(d.car_pace_offset_s) for d in setup.drivers}


def team_principal(
    setup: RaceSetup, driver_id: str, context: ChampionshipContext | None
) -> TeamOrder:
    """Editorial rules (M5): a driver in the title fight in a top-3 car plays for the
    championship, everyone else for points."""
    if context is None:
        return TeamOrder(Objective.POINTS, reason="no championship context: points")
    if max(context.points.values(), default=0.0) <= 0.0:
        return TeamOrder(Objective.POINTS, reason="season start, no title fight yet: points")
    rank = car_ranks(setup)[driver_id]
    position = context.position(driver_id)
    gap = context.gap_to_leader(driver_id)
    in_fight = position <= 3 and gap <= 0.6 * context.max_points_left()
    if in_fight and rank <= 2 and context.races_left > 0:
        table = context.table()
        rival = table[1] if table[0] == driver_id else table[0]
        return TeamOrder(
            Objective.CHAMPIONSHIP,
            rival=rival,
            rival_weight=1.0,
            reason=f"P{position} in the championship, {gap:.0f} pts behind, top-3 car",
        )
    return TeamOrder(
        Objective.POINTS,
        reason=f"P{position} in the championship, car rank {rank + 1}: points when possible",
    )


@dataclass(frozen=True)
class CarResult:
    position: int
    """Classified order: legal finishers (by laps, then time), retirements, disqualifications."""
    finished: bool
    points: int
    gap_s: float
    """Behind the winner, with laps down at the winner's average lap; NaN unless classified."""
    disqualified: bool = False
    """Finished without the compounds the rules require (see ``race_results``)."""


def race_results(state: WorldState) -> dict[str, CarResult]:
    """Results as the AI is scored. The engine only *reports* a broken two-dry-compounds rule
    (``TYRE_RULE_BREACHED``); here such a car is scored as disqualified (no points, classified
    last), an editorial stand-in for the regulation penalty. Without it, an AI would learn to
    skip the mandatory stop: an engine-bug exploit (D-037)."""
    setup = state.setup
    legal, retired, dsq = [], [], []
    for driver_id in standings(state):
        car = state.car(driver_id)
        if car.status is not CarStatus.FINISHED:
            retired.append(driver_id)
        elif len(set(car.compounds_used)) < dry_compounds_required(setup, car.compounds_used):
            dsq.append(driver_id)
        else:
            legal.append(driver_id)
    order = legal + retired + dsq
    winner = state.car(legal[0]) if legal else None
    winner_time = winner.finish_time_s or 0.0 if winner else 0.0
    winner_laps = winner.laps_completed if winner else 0
    avg_lap = winner_time / max(1, winner_laps)
    out = {}
    for i, driver_id in enumerate(order):
        car = state.car(driver_id)
        position = i + 1
        classified = driver_id in legal
        gap = math.nan
        if classified and car.finish_time_s is not None:
            gap = car.finish_time_s - winner_time + avg_lap * (winner_laps - car.laps_completed)
        points = POINTS_2024[position - 1] if classified and position <= len(POINTS_2024) else 0
        out[driver_id] = CarResult(
            position,
            car.status is CarStatus.FINISHED,
            points,
            gap,
            disqualified=driver_id in dsq,
        )
    return out


def objective_value(
    order: TeamOrder, driver_id: str, results: Mapping[str, CarResult], field_size: int
) -> float:
    """The race result as one number for ``order.objective`` (roughly -1 .. 1)."""
    me = results[driver_id]
    place = (field_size - me.position) / max(1, field_size - 1)  # 1 = win, 0 = last
    match order.objective:
        case Objective.TIME:
            if not me.finished or me.disqualified:
                return -1.0
            return max(-1.0, 1.0 - me.gap_s / TIME_SCALE_S)
        case Objective.RESULT:
            return place
        case Objective.POINTS:
            return me.points / POINTS_2024[0] + POSITION_TIEBREAK * place
        case Objective.CHAMPIONSHIP:
            rival = (
                results[order.rival].points
                if order.rival is not None and order.rival in results
                else 0
            )
            w = order.rival_weight
            value = (me.points + w * (me.points - rival)) / (POINTS_2024[0] * (1 + w))
            return value + POSITION_TIEBREAK * place


def risk_utility(x: float, risk_appetite: float) -> float:
    """``(1 - exp(-a x)) / a`` with ``a = RISK_CURVATURE (0.5 - risk_appetite)``: concave for
    careful drivers, linear at 0.5, convex for risk takers (editorial preference)."""
    a = RISK_CURVATURE * (0.5 - risk_appetite)
    if abs(a) < 1e-9:
        return x
    return float((1.0 - np.exp(-a * x)) / a)


@dataclass(frozen=True)
class Scoring:
    """How one car's race is scored: its team order and its driver's risk appetite."""

    order: TeamOrder
    risk_appetite: float
    notes: tuple[str, ...] = field(default=())

    def reward(self, driver_id: str, results: Mapping[str, CarResult], field_size: int) -> float:
        return risk_utility(
            objective_value(self.order, driver_id, results, field_size), self.risk_appetite
        )


def scoring_for(
    setup: RaceSetup,
    drivers: Sequence[str],
    context: ChampionshipContext | None,
    objective: Objective | None = None,
) -> dict[str, Scoring]:
    """Team-principal orders (or a forced ``objective``, for curricula) and risk appetites."""
    out = {}
    for d in drivers:
        order = (
            TeamOrder(objective, reason=f"forced: {objective.value}")
            if objective is not None
            else team_principal(setup, d, context)
        )
        out[d] = Scoring(order, setup.driver(d).profile.archetype.risk_appetite)
    return out
