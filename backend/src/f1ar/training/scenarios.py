"""Training scenarios: which race, which seed, which weather, which cars are AI, and the season
around it (M5, ``docs/M5_DESIGN.md`` §3, §7).

Training races are **ephemeral** (plan §9.11): they never touch a user's saved worlds.

**Championship context** during training is synthetic (labelled ``editorial_assumption``):
``races_done`` past races are drawn with each car's strength (the race's car pace offsets) plus
the driver's ``race_pace_skill`` and noise, scored with 2024 points. It gives the policy title
fights and points chances in plausible proportions; it is not a model of the 2024 season. Once
worlds run seasons (M9), the world's real standings are used instead.

**Tyre randomisation** (M5 slice 3b, D-047; ``tyre_randomisation`` > 0, training only): each
training race gets perturbed tyres (life, wear, pace offset per dry compound, the cliff) and
pit loss, so a policy has to read them from its observation (``rl_obs_v2``) instead of
memorising the 16 training tracks. The rule planner (the opponents and the paired baseline)
sees the same perturbed setup. Labelled ``editorial_assumption`` in the setup's inputs: a
training device, never a race the user plays.

**Character randomisation** (slice 4, D-048; ``character_randomisation`` > 0, training only): the
AI car's ``risk_appetite`` is redrawn per race (uniform over the persona scale), so one network
learns how the whole range of characters should race and each real driver's persona picks its
point; the scoring (risk utility) and the observation both use the drawn value. Labelled
``editorial_assumption`` in the setup's inputs.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np

from f1ar.domain.race_state import DRY_COMPOUNDS, Compound, RaceSetup
from f1ar.training.objectives import (
    SEASON_RACES_2024,
    ChampionshipContext,
    Objective,
    Scoring,
    scoring_for,
)

SetupFactory = Callable[[str, bool], RaceSetup]
"""``(event_id, random_rain) -> RaceSetup``."""
WORLDS = ("plain", "a1b")
"""``plain``: the setup as built (teams know the truth, M5); ``a1b``: hidden drawn truth,
per-team practice + private data, in-race learning, hidden trend, rival rules (A2, D-064)."""
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


def randomise_tyres(setup: RaceSetup, rng: np.random.Generator, scale: float) -> RaceSetup:
    """A copy of ``setup`` with perturbed tyres and pit loss (log-normal factors with sd
    ``scale``; soft/hard pace offsets vs the medium shifted by ``scale`` s)."""
    if scale <= 0:
        return setup
    params = setup.lap_model

    def factor() -> float:
        return float(np.exp(rng.normal(0.0, scale)))

    compounds = dict(params.compounds)
    for c in DRY_COMPOUNDS:
        cp = compounds[c]
        wear = factor()
        update: dict[str, float | None] = {
            "deg_s_per_lap": cp.deg_s_per_lap * wear,
            "deg_quad_s_per_lap2": cp.deg_quad_s_per_lap2 * wear,
            "life_laps": cp.life_laps * factor() if cp.life_laps is not None else None,
        }
        if c is not Compound.MEDIUM:
            update["offset_s"] = cp.offset_s + float(rng.normal(0.0, scale))
        compounds[c] = cp.model_copy(update=update)
    lap_model = params.model_copy(
        update={"compounds": compounds, "cliff_s_per_lap": params.cliff_s_per_lap * factor()}
    )
    pit = setup.pit.model_copy(update={"lane_loss_s": setup.pit.lane_loss_s * factor()})
    note = f"editorial_assumption: training perturbation of tyres and pit loss (sd {scale})"
    return setup.model_copy(
        update={
            "lap_model": lap_model,
            "pit": pit,
            "inputs": {**setup.inputs, "tyre_randomisation": note},
        }
    )


def randomise_character(
    setup: RaceSetup, drivers: Sequence[str], rng: np.random.Generator
) -> RaceSetup:
    """A copy of ``setup`` whose ``drivers`` get a risk appetite drawn uniformly in [0.05, 0.95]."""
    new = []
    for d in setup.drivers:
        if d.driver_id in drivers:
            arche = d.profile.archetype.model_copy(
                update={"risk_appetite": round(float(rng.uniform(0.05, 0.95)), 3)}
            )
            d = d.model_copy(update={"profile": d.profile.model_copy(update={"archetype": arche})})
        new.append(d)
    note = "editorial_assumption: AI car's risk appetite redrawn for training (slice 4)"
    return setup.model_copy(
        update={"drivers": tuple(new), "inputs": {**setup.inputs, "character_randomisation": note}}
    )


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
        for pos, i in enumerate(order[: len(setup.rules.points)]):
            points[i] += setup.rules.points[pos]
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
        tyre_randomisation: float = 0.0,
        character_randomisation: bool = False,
        world: str = "plain",
        configs: Path | None = None,
    ) -> None:
        """``agents``: how many cars the AI runs (an int, or ``"all"``); ``objective`` forces one
        objective for every AI car (curricula), otherwise the team principal decides;
        ``tyre_randomisation``: sd of the training perturbation (``randomise_tyres``)."""
        if not events:
            raise ValueError("no events to sample from")
        self.events = tuple(events)
        self.factory = factory
        self.agents = agents
        self.rain_probability = rain_probability
        self.championship_probability = championship_probability
        self.objective = objective
        self.tyre_randomisation = tyre_randomisation
        self.character_randomisation = character_randomisation
        if world not in WORLDS:
            raise ValueError(f"world must be one of {WORLDS}")
        if world != "plain" and configs is None:
            raise ValueError("a hidden-truth world needs the configs directory")
        self.world = world
        self.configs = configs

    def scoring(
        self, setup: RaceSetup, agents: Sequence[str], context: ChampionshipContext | None
    ) -> dict[str, Scoring]:
        """Scoring for (more) AI cars under this sampler's rules."""
        return scoring_for(setup, agents, context, self.objective)

    def sample(self, rng: np.random.Generator) -> tuple[Scenario, RaceSetup]:
        event_id = self.events[int(rng.integers(len(self.events)))]
        random_rain = bool(rng.random() < self.rain_probability)
        setup = randomise_tyres(self.factory(event_id, random_rain), rng, self.tyre_randomisation)
        if self.world == "a1b":  # A2 (D-064): a new hidden truth every episode, realistic teams
            from f1ar.simulation.race_day import TruthMode, with_hidden_truth, with_rival_rules

            assert self.configs is not None
            setup = with_hidden_truth(
                setup,
                self.configs,
                mode=TruthMode.DRAWN,
                truth_seed=int(rng.integers(1, 2**31 - 1)),
                learning=True,
                track_trend=True,
                per_team=True,
            )
            setup = with_rival_rules(setup, self.configs)
        drivers = [d.driver_id for d in setup.drivers]
        n = len(drivers) if self.agents == "all" else int(self.agents)
        picked = rng.choice(drivers, size=min(n, len(drivers)), replace=False)
        agents = tuple(sorted(str(d) for d in picked))
        if self.character_randomisation:
            setup = randomise_character(setup, agents, rng)
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
