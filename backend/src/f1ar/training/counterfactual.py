"""Decision-level learning signal from counterfactual races on identical luck (M5 slice 4b, D-049).

EXP-018 found that a terminal race reward can't credit one racecraft decision: one ATTACK lap
changes the crash chance by ~0.05 %, the race result has an sd of ~0.15. The engine's randomness
is **keyed** (seed, stream, driver, lap: ``rng_streams``), so a race can be forked at a decision
and every branch replays the *same* luck: same lap noise, incidents, pass draws, weather. The
difference between two branches' results is then the exact effect of that one decision in that
race (common random numbers), not drowned by unrelated luck.

``decision_targets`` forks a session at an AI car's decision, plays each racecraft intent for
the next lap (the teacher's pit/pace choice, then the rules teacher with NORMAL to the flag) and
scores each branch with the car's own objective and risk utility. A model trained on many such
targets learns E[advantage | what the car sees]; it never sees the future at decision time
(the branches use the real seed only to produce training targets, like hindsight analysis).
This is one step of policy improvement over the rules (racecraft only; pit and pace stay the
rules'), so with exact targets it can't be worse than them on average.
"""

import copy
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from f1ar.domain.race_state import RacecraftIntent
from f1ar.policy.strategists import RulesTeacher
from f1ar.training.actions import INTENT_OPTIONS, decode, encode
from f1ar.training.session import Decision, RaceSession

BRANCH_INTENTS = INTENT_OPTIONS
"""DEFEND, NORMAL, ATTACK (the order of the targets)."""


def fork(session: RaceSession) -> RaceSession:
    """An independent copy of ``session`` at this point (the world state is immutable; the
    session's own queues and counters are copied)."""
    other = copy.copy(session)
    other._pending = deque(session._pending)
    other.stop_compounds = {k: list(v) for k, v in session.stop_compounds.items()}
    other.scoring = dict(session.scoring)
    return other


def finish(session: RaceSession, choose: Callable[[RaceSession, Decision], int]) -> None:
    """Play the race to the end with ``choose`` deciding for every AI car."""
    while (d := session.next_decision()) is not None:
        session.act(d.driver_id, choose(session, d))


def _teacher_normal(teacher: RulesTeacher) -> Callable[[RaceSession, Decision], int]:
    def choose(session: RaceSession, d: Decision) -> int:
        pit, pace, _ = decode(teacher.decide(session, d))
        action = encode(pit, pace, RacecraftIntent.NORMAL)
        return action if d.mask[action] else encode(pit, pace)

    return choose


@dataclass(frozen=True)
class Targets:
    rewards: tuple[float, float, float]
    """The car's score with DEFEND / NORMAL / ATTACK for the next lap (same luck)."""

    @property
    def advantages(self) -> np.ndarray:
        """DEFEND - NORMAL, ATTACK - NORMAL."""
        d, n, a = self.rewards
        return np.array([d - n, a - n])


def _teacher_with(
    teacher: RulesTeacher, intent: RacecraftIntent, driver_id: str, until_lap: int
) -> Callable[[RaceSession, Decision], int]:
    """The rules teacher's pit/pace; ``intent`` for ``driver_id`` until ``until_lap``, then
    NORMAL."""

    def choose(session: RaceSession, d: Decision) -> int:
        pit, pace, _ = decode(teacher.decide(session, d))
        mine = intent if d.driver_id == driver_id and d.lap < until_lap else RacecraftIntent.NORMAL
        action = encode(pit, pace, mine)
        return action if d.mask[action] else encode(pit, pace)

    return choose


def decision_targets(session: RaceSession, decision: Decision, *, laps: int = 5) -> Targets | None:
    """Fork at ``decision`` and score each racecraft intent held for the next ``laps`` laps (a
    commitment, like a driver deciding to attack a car), then NORMAL to the flag. ``None`` when
    an intent isn't legal now (e.g. the final lap)."""
    teacher = RulesTeacher()
    out = []
    for intent in BRANCH_INTENTS:
        pit, pace, _ = decode(teacher.decide(session, decision))
        if not decision.mask[encode(pit, pace, intent)]:
            return None
        branch = fork(session)
        finish(branch, _teacher_with(teacher, intent, decision.driver_id, decision.lap + laps))
        out.append(branch.rewards()[decision.driver_id])
    return Targets((out[0], out[1], out[2]))


@dataclass(frozen=True)
class PitTargets:
    rules: float
    """The car's score following the rules teacher from here."""
    alternative: float
    """... deviating once: boxing now when the rules stay out (for the rules' next compound), or
    staying out this lap when the rules box (then the rules again)."""
    rules_boxes: bool

    @property
    def advantage(self) -> float:
        return self.alternative - self.rules


def pit_targets(session: RaceSession, decision: Decision) -> PitTargets | None:
    """Fork at ``decision``: the rules' pit choice vs the one-lap alternative, on the same luck
    (slice 4b: pit timing, where a decision moves the result by seconds). ``None`` if the
    alternative isn't legal or the rules have no further stop."""
    from f1ar.policy.baseline_strategy import plan_for_car

    teacher = RulesTeacher()
    rest = _teacher_normal(teacher)
    pit, pace, _ = decode(teacher.decide(session, decision))
    car = session.state.car(decision.driver_id)
    if pit is None:
        plan = plan_for_car(session.state.setup, car)
        if not plan:
            return None
        alt = encode(plan[0].compound, pace)
    else:
        alt = encode(None, pace)
    if not decision.mask[alt]:
        return None
    base = fork(session)
    finish(base, rest)
    other = fork(session)
    other.act(decision.driver_id, alt)
    first = True

    def then_rules(s: RaceSession, d: Decision) -> int:
        nonlocal first
        if first and d.driver_id == decision.driver_id and d.lap == decision.lap:
            first = False
            return alt  # the same decision point can come again (e.g. an SC event): keep it
        return rest(s, d)

    finish(other, then_rules)
    me = decision.driver_id
    return PitTargets(base.rewards()[me], other.rewards()[me], pit is not None)
