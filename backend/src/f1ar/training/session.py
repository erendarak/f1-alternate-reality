"""A race in which some (or all) cars are run by an AI policy (M5, ``docs/M5_DESIGN.md`` §7).

The session drives the pure engine and stops whenever an AI car has to decide:

- at the end of each of its laps (the car is still running),
- for every AI car still running: a safety car or VSC is deployed, rain starts or stops.

A decision is one of 18 actions: pit choice ``[stay out, S, M, H, I, W]`` x pace
``[conserve, balanced, push]``, masked by the engine's legal actions and the regulation set.
Cars not run by the AI keep their controller (normally the rule-based baseline).

AI cars get ``StrategyController.AGENT``: the baseline planner leaves them alone, and every
command they send carries ``actor_id = "AGENT:<policy id>"`` (logged like any command). Same
setup + seed + actions = same race (the engine is deterministic and keyed).
"""

from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from f1ar.domain.race_state import (
    CarStatus,
    Compound,
    RaceEvent,
    RaceSetup,
    SetStrategyController,
    StrategyController,
    WorldState,
    WorldStatus,
)
from f1ar.simulation import lightweight_engine as engine
from f1ar.training.actions import (
    INTENT_OPTIONS,
    PACE_OPTIONS,
    PIT_OPTIONS,
    agent_commands,
    decode,
)
from f1ar.training.features import RL_OBSERVATION_SCHEMA, rl_observation
from f1ar.training.objectives import (
    CarResult,
    ChampionshipContext,
    Scoring,
    race_results,
)
from f1ar.training.regulations import REGULATIONS_2024, RegulationSet

DECISION_EVENTS = ("SAFETY_CAR_DEPLOYED", "VSC_DEPLOYED", "RAIN_STARTED", "RAIN_STOPPED")
"""Race events that make every running AI car decide (besides its own lap end)."""
MAX_TICKS = 200_000
"""Safety net: a race never needs this many ticks (Monaco ~ 30k)."""


@dataclass(frozen=True)
class Decision:
    driver_id: str
    observation: np.ndarray
    mask: np.ndarray
    """Bool, ``N_ACTIONS`` long: which actions are legal now."""
    reason: str
    """``LAP_END`` or the race event that triggered it."""
    lap: int


class RaceSession:
    def __init__(
        self,
        setup: RaceSetup,
        *,
        seed: int,
        agents: Sequence[str],
        scoring: Mapping[str, Scoring],
        context: ChampionshipContext | None = None,
        regulations: RegulationSet = REGULATIONS_2024,
        policy_id: str = "untrained",
        world_id: str = "training",
        observation_schema: str = RL_OBSERVATION_SCHEMA,
    ) -> None:
        missing = [d for d in agents if d not in scoring]
        if missing:
            raise ValueError(f"no scoring for agent(s) {missing}")
        self.setup = setup
        self.agents = tuple(agents)
        self.scoring = dict(scoring)
        self.context = context
        self.regulations = regulations
        self.observation_schema = observation_schema
        self.actor_id = f"AGENT:{policy_id}"
        state = engine.create_world(setup, seed=seed, world_id=world_id)
        for d in self.agents:
            action = SetStrategyController(
                driver_id=d, controller=StrategyController.AGENT, actor_id=self.actor_id
            )
            state = engine.apply_action(state, action).state
        self.state: WorldState = engine.start(state)
        self._pending: deque[tuple[str, str]] = deque()
        self.decisions = 0
        self.stop_compounds: dict[str, list[Compound]] = {}
        """Every car's new compound at each of its pit stops, in order (from PIT_STOP events)."""

    # --- the loop ---------------------------------------------------------------------------

    @property
    def done(self) -> bool:
        return self.state.world_status is not WorldStatus.RUNNING

    def next_decision(self) -> Decision | None:
        """Advance the race to the next AI decision; None when the race is over."""
        ticks = 0
        while True:
            while self._pending:
                driver_id, reason = self._pending.popleft()
                if self.state.car(driver_id).status is CarStatus.RUNNING:
                    return self._decision(driver_id, reason)
            if self.done:
                return None
            result = engine.step(self.state)
            self.state = result.state
            ticks += 1
            if ticks > MAX_TICKS:
                raise RuntimeError("race did not finish (tick safety net)")
            self._enqueue(result.events)

    def apply_external(self, result: engine.StepResult) -> None:
        """A change made outside the AI loop (a race-director action, a user command): take
        its state and let the AI cars react to its events."""
        self.state = result.state
        self._enqueue(result.events)

    def _enqueue(self, events: Sequence[RaceEvent]) -> None:
        queued = {d for d, _ in self._pending}
        for e in events:
            if e.type == "PIT_STOP" and e.driver_id is not None:
                compound = Compound(str(e.payload["to_compound"]))
                self.stop_compounds.setdefault(e.driver_id, []).append(compound)
            if e.type == "LAP_COMPLETED" and e.driver_id in self.agents:
                if e.driver_id not in queued:
                    self._pending.append((e.driver_id, "LAP_END"))
                    queued.add(e.driver_id)
            elif e.type in DECISION_EVENTS:
                for d in self.agents:
                    if d not in queued:
                        self._pending.append((d, e.type))
                        queued.add(d)

    def _decision(self, driver_id: str, reason: str) -> Decision:
        self.decisions += 1
        car = self.state.car(driver_id)
        return Decision(
            driver_id=driver_id,
            observation=rl_observation(
                self.state,
                driver_id,
                self.scoring[driver_id].order,
                self.context,
                self.observation_schema,
            ),
            mask=self.action_mask(driver_id),
            reason=reason,
            lap=car.laps_completed + 1,
        )

    def action_mask(self, driver_id: str) -> np.ndarray:
        valid = engine.valid_actions(self.state, driver_id)
        pit_ok = [
            c is None or (c in valid.pit_compounds and self.regulations.allows_pit(c))
            for c in PIT_OPTIONS
        ]
        pace_ok = [m in valid.pace_modes and self.regulations.allows_pace(m) for m in PACE_OPTIONS]
        intent_ok = [i in valid.intents for i in INTENT_OPTIONS]
        return np.array(
            [p and q and r for p in pit_ok for q in pace_ok for r in intent_ok], dtype=bool
        )

    def act(self, driver_id: str, action: int) -> None:
        """Apply one decision for ``driver_id`` (must be legal under its mask)."""
        if driver_id not in self.agents:
            raise ValueError(f"{driver_id} is not run by the AI")
        if not self.action_mask(driver_id)[action]:
            raise ValueError(f"action {action} {decode(action)} is not legal for {driver_id}")
        for c in agent_commands(self.state.car(driver_id), action, self.actor_id):
            self.state = engine.apply_action(self.state, c).state

    # --- the result ---------------------------------------------------------------------------

    def results(self) -> dict[str, CarResult]:
        if not self.done:
            raise RuntimeError("the race is still running")
        return race_results(self.state)

    def rewards(self) -> dict[str, float]:
        """Each AI car's score for the finished race (its objective under its risk utility)."""
        results = self.results()
        n = len(self.setup.drivers)
        return {d: self.scoring[d].reward(d, results, n) for d in self.agents}
