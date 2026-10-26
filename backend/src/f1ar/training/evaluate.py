"""Paired evaluation of strategists (M5, plan §9.10): one car, one race, one seed.

Every method plays the *same* race (same setup, seed, other cars, scoring), so differences
between methods on a case are paired: common random numbers make them far less noisy than
independent races. ``strategist=None`` is B1: the car stays with the rule planner.
"""

import time
from dataclasses import dataclass

from f1ar.domain.race_state import RaceSetup
from f1ar.policy.strategists import Strategist
from f1ar.training.features import RL_OBSERVATION_SCHEMA
from f1ar.training.objectives import ChampionshipContext, Scoring, race_results
from f1ar.training.regulations import REGULATIONS_2024, RegulationSet
from f1ar.training.session import RaceSession


@dataclass(frozen=True)
class CaseResult:
    method: str
    event_id: str
    driver_id: str
    seed: int
    reward: float
    position: int
    points: int
    finished: bool
    disqualified: bool
    stops: int
    sequence: str
    decisions: int
    seconds: float


def evaluate(
    setup: RaceSetup,
    driver_id: str,
    seed: int,
    scoring: Scoring,
    strategist: Strategist | None,
    *,
    context: ChampionshipContext | None = None,
    regulations: RegulationSet = REGULATIONS_2024,
    racecraft: bool = True,
    regulation_guard: bool = False,
) -> CaseResult:
    t0 = time.perf_counter()
    agents = [driver_id] if strategist is not None else []
    session = RaceSession(
        setup,
        seed=seed,
        agents=agents,
        scoring={driver_id: scoring},
        context=context,
        regulations=regulations,
        policy_id=strategist.name if strategist is not None else "none",
        observation_schema=getattr(strategist, "observation_schema", RL_OBSERVATION_SCHEMA),
        racecraft=racecraft,
        regulation_guard=regulation_guard,
    )
    if strategist is not None:
        strategist.start(session, driver_id)
    while (d := session.next_decision()) is not None:
        assert strategist is not None
        session.act(d.driver_id, strategist.decide(session, d))
    results = race_results(session.state)
    me = results[driver_id]
    car = session.state.car(driver_id)
    stints = [setup.driver(driver_id).start_compound, *session.stop_compounds.get(driver_id, [])]
    sequence = "-".join(c.value[0] for c in stints)
    return CaseResult(
        method=strategist.name if strategist is not None else "rules",
        event_id=setup.event_id,
        driver_id=driver_id,
        seed=seed,
        reward=scoring.reward(driver_id, results, len(setup.drivers)),
        position=me.position,
        points=me.points,
        finished=me.finished,
        disqualified=me.disqualified,
        stops=car.pit_stops,
        sequence=sequence,
        decisions=session.decisions,
        seconds=time.perf_counter() - t0,
    )
