"""Gymnasium environments over ``RaceSession`` (M5, ``docs/M5_DESIGN.md`` §7).

- ``RaceStrategyEnv``: one AI car per episode (the scenario picks it); the other cars follow
  their controller (the rule baseline) or, if given, an ``opponent_policy`` that runs them as
  AI too (a frozen earlier policy, slice 5). One step = one decision of the learning car.
  The reward is 0 until the race ends, then the car's score (objective under its risk utility).
  ``action_masks()`` follows the sb3-contrib MaskablePPO convention.
- ``run_race``: a whole race where one callable decides for every AI car (evaluation,
  benchmarks, self-play data); returns every AI car's reward.

Episodes are reproducible: ``reset(seed=s)`` draws the scenario from ``s``.
"""

from collections.abc import Callable
from typing import Any

import gymnasium as gym
import numpy as np

from f1ar.domain.race_state import RaceSetup
from f1ar.training.features import RL_FEATURE_NAMES, RL_OBSERVATION_SCHEMA
from f1ar.training.regulations import REGULATIONS_2024, RegulationSet
from f1ar.training.scenarios import Scenario, ScenarioSampler
from f1ar.training.session import N_ACTIONS, Decision, RaceSession

Policy = Callable[[Decision], int]
"""Picks a legal action index for a decision."""


def random_policy(rng: np.random.Generator) -> Policy:
    def act(d: Decision) -> int:
        return int(rng.choice(np.flatnonzero(d.mask)))

    return act


class RaceStrategyEnv(gym.Env[np.ndarray, np.int64]):
    metadata = {"render_modes": []}  # noqa: RUF012 (gymnasium's convention)

    def __init__(
        self,
        sampler: ScenarioSampler,
        *,
        regulations: RegulationSet = REGULATIONS_2024,
        opponent_policy: Policy | None = None,
        opponents: int | str = 0,
        policy_id: str = "learning",
    ) -> None:
        """``opponents``: how many other cars ``opponent_policy`` runs (an int or ``"all"``)."""
        self.sampler = sampler
        self.regulations = regulations
        self.opponent_policy = opponent_policy
        self.opponents = opponents
        self.policy_id = policy_id
        self.observation_space = gym.spaces.Box(
            low=-10.0, high=10.0, shape=(len(RL_FEATURE_NAMES),), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(N_ACTIONS)
        self.session: RaceSession | None = None
        self.scenario: Scenario | None = None
        self.me = ""
        self._decision: Decision | None = None

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        while True:
            scenario, setup = self.sampler.sample(self.np_random)
            self.me = scenario.agents[0]
            agents = [self.me, *self._opponent_cars(scenario, setup)]
            scoring = {
                **scenario.scoring,
                **self.sampler.scoring(setup, agents, scenario.context),
            }
            self.scenario = scenario
            self.session = RaceSession(
                setup,
                seed=scenario.seed,
                agents=agents,
                scoring=scoring,
                context=scenario.context,
                regulations=self.regulations,
                policy_id=self.policy_id,
            )
            self._decision = self._advance()
            if self._decision is not None:  # the car retired before deciding: draw again
                return self._decision.observation, self._info()

    def _opponent_cars(self, scenario: Scenario, setup: RaceSetup) -> list[str]:
        if self.opponent_policy is None or self.opponents == 0:
            return []
        others = [d.driver_id for d in setup.drivers if d.driver_id != self.me]
        n = len(others) if self.opponents == "all" else int(self.opponents)
        rng = np.random.default_rng(scenario.seed)
        return sorted(str(d) for d in rng.choice(others, size=min(n, len(others)), replace=False))

    def _advance(self) -> Decision | None:
        """Next decision of the learning car; opponents' decisions are answered on the way."""
        assert self.session is not None
        while True:
            d = self.session.next_decision()
            if d is None or d.driver_id == self.me:
                return d
            assert self.opponent_policy is not None
            self.session.act(d.driver_id, self.opponent_policy(d))

    def step(self, action: np.int64 | int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        assert self.session is not None and self._decision is not None, "call reset() first"
        self.session.act(self.me, int(action))
        last = self._decision
        self._decision = self._advance()
        if self._decision is None:
            reward = self.session.rewards()[self.me]
            return last.observation, reward, True, False, self._info(final=True)
        return self._decision.observation, 0.0, False, False, self._info()

    def action_masks(self) -> np.ndarray:
        if self._decision is None:
            return np.ones(N_ACTIONS, dtype=bool)
        return self._decision.mask

    def _info(self, *, final: bool = False) -> dict[str, Any]:
        assert self.session is not None and self.scenario is not None
        info: dict[str, Any] = {
            "event_id": self.scenario.event_id,
            "driver_id": self.me,
            "objective": self.session.scoring[self.me].order.objective.value,
            "observation_schema": RL_OBSERVATION_SCHEMA,
            "regulations": self.regulations.version,
        }
        if self._decision is not None:
            info["reason"] = self._decision.reason
            info["lap"] = self._decision.lap
        if final:
            result = self.session.results()[self.me]
            info["position"] = result.position
            info["points"] = result.points
            info["finished"] = result.finished
        return info


def run_race(session: RaceSession, policy: Policy) -> dict[str, float]:
    """Race to the end with ``policy`` deciding for every AI car; their rewards."""
    while (d := session.next_decision()) is not None:
        session.act(d.driver_id, policy(d))
    return session.rewards()
