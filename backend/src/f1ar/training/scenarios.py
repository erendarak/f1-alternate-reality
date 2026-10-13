"""Training scenarios: which race, which seed, which weather, which cars are AI, and the season
around it (M5, ``docs/M5_DESIGN.md`` §3, §7).

Training races are **ephemeral** (plan §9.11): they never touch a user's saved worlds.

**Championship context** during training is synthetic (labelled ``editorial_assumption``):
``races_done`` past races are drawn with each car's strength (the race's car pace offsets) plus
the driver's ``race_pace_skill`` and noise, scored with 2024 points. It gives the policy title
fights and points chances in plausible proportions; it is not a model of the 2024 season. Once
worlds run seasons (M9), the world's real standings are used instead.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np

from f1ar.domain.race_state import RaceSetup
from f1ar.training.objectives import (
    POINTS_2024,
    SEASON_RACES_2024,
    ChampionshipContext,
    Objective,
    Scoring,
    scoring_for,
)

SetupFactory = Callable[[str, bool], RaceSetup]
"""``(event_id, random_rain) -> RaceSetup``."""
STANDINGS_NOISE = 1.0
"""Per-race noise of the synthetic standings, in units of the car/driver strength score."""


def repo_setup_factory(root: Path) -> SetupFactory:
    """Data-backed setups from the repo's fetched races, built once per (event, weather)."""
    from f1ar.simulation.scenarios import event_data_dir, load_event, setup_from_codes

    @cache
    def factory(event_id: str, random_rain: bool) -> RaceSetup:
        data_dir = event_data_dir(root, event_id)
        if data_dir is None:
            raise ValueError(f"{event_id} isn't fetched: scripts/fetch_season.py")
        roster, profiles = load_event(root / "configs", event_id)
        return setup_from_codes(
            roster,
            profiles,
            None,
            data_dir=data_dir,
            configs_dir=root / "configs",
            random_rain=random_rain,
        )

    return factory


def synthetic_standings(
    setup: RaceSetup, rng: np.random.Generator, races_done: int
) -> ChampionshipContext:
    drivers = [d.driver_id for d in setup.drivers]
    strength = np.array(
        [
            -d.car_pace_offset_s / 0.3 + 2.0 * d.profile.archetype.race_pace_skill
            for d in setup.drivers
        ]
    )
    points = np.zeros(len(drivers))
    for _ in range(races_done):
        order = np.argsort(-(strength + rng.normal(0.0, STANDINGS_NOISE, len(drivers))))
        for pos, i in enumerate(order[: len(POINTS_2024)]):
            points[i] += POINTS_2024[pos]
    return ChampionshipContext(
        points={d: float(p) for d, p in zip(drivers, points, strict=True)},
        races_left=SEASON_RACES_2024 - races_done,
    )


@dataclass(frozen=True)
class Scenario:
    event_id: str
    seed: int
    random_rain: bool
    agents: tuple[str, ...]
    context: ChampionshipContext | None
    scoring: dict[str, Scoring]


class ScenarioSampler:
    def __init__(
        self,
        events: Sequence[str],
        factory: SetupFactory,
        *,
        agents: int | str = 1,
        rain_probability: float = 0.0,
        championship_probability: float = 0.8,
        objective: Objective | None = None,
    ) -> None:
        """``agents``: how many cars the AI runs (an int, or ``"all"``); ``objective`` forces one
        objective for every AI car (curricula), otherwise the team principal decides."""
        if not events:
            raise ValueError("no events to sample from")
        self.events = tuple(events)
        self.factory = factory
        self.agents = agents
        self.rain_probability = rain_probability
        self.championship_probability = championship_probability
        self.objective = objective

    def scoring(
        self, setup: RaceSetup, agents: Sequence[str], context: ChampionshipContext | None
    ) -> dict[str, Scoring]:
        """Scoring for (more) AI cars under this sampler's rules."""
        return scoring_for(setup, agents, context, self.objective)

    def sample(self, rng: np.random.Generator) -> tuple[Scenario, RaceSetup]:
        event_id = self.events[int(rng.integers(len(self.events)))]
        random_rain = bool(rng.random() < self.rain_probability)
        setup = self.factory(event_id, random_rain)
        drivers = [d.driver_id for d in setup.drivers]
        n = len(drivers) if self.agents == "all" else int(self.agents)
        picked = rng.choice(drivers, size=min(n, len(drivers)), replace=False)
        agents = tuple(sorted(str(d) for d in picked))
        context = None
        if rng.random() < self.championship_probability:
            races_done = int(rng.integers(1, SEASON_RACES_2024))
            context = synthetic_standings(setup, rng, races_done)
        scenario = Scenario(
            event_id=event_id,
            seed=int(rng.integers(1, 2**31 - 1)),
            random_rain=random_rain,
            agents=agents,
            context=context,
            scoring=self.scoring(setup, agents, context),
        )
        return scenario, setup
