"""A2 (D-064): rl_obs_v4 sees no hidden truth; the A1b training world; 18 effective actions."""

import numpy as np

from f1ar.domain.race_state import RaceSetup
from f1ar.simulation import lightweight_engine as engine
from f1ar.simulation.race_day import scaled
from f1ar.training.actions import INTENT_OPTIONS, decode
from f1ar.training.features import RL_FEATURE_SETS, rl_observation
from f1ar.training.objectives import Objective, TeamOrder
from f1ar.training.scenarios import ScenarioSampler, repo_setup_factory
from f1ar.training.session import RaceSession
from tests.conftest import CONFIGS, REPO_ROOT, VER
from tests.unit.test_race_day import _hidden
from tests.unit.test_tyre_life import _with_life

ORDER = TeamOrder(Objective.POINTS)


def _obs(setup: RaceSetup, schema: str) -> np.ndarray:
    state = engine.create_world(setup, seed=1, world_id="w")
    return rl_observation(state, VER, ORDER, None, schema)


def test_v4_does_not_see_the_truth(two_car_setup: RaceSetup) -> None:
    hidden = _hidden(_with_life(two_car_setup), life=1.0, wear=1.0)
    other_truth = hidden.model_copy(
        update={
            "lap_model": scaled(
                hidden.lap_model, dict.fromkeys(("SOFT", "MEDIUM", "HARD"), 0.5), 2.0
            )
        }
    )
    assert np.array_equal(_obs(hidden, "rl_obs_v4"), _obs(other_truth, "rl_obs_v4"))
    assert not np.array_equal(_obs(hidden, "rl_obs_v3"), _obs(other_truth, "rl_obs_v3"))
    assert len(_obs(hidden, "rl_obs_v4")) == len(RL_FEATURE_SETS["rl_obs_v4"])
    assert not any(
        n.startswith(("plan.", "track.soft", "track.medium.", "track.hard"))
        for n in RL_FEATURE_SETS["rl_obs_v4"]
    )


def test_a1b_world_and_eighteen_actions() -> None:
    sampler = ScenarioSampler(
        ["2024_italian_gp"], repo_setup_factory(REPO_ROOT), world="a1b", configs=CONFIGS
    )
    a, setup_a = sampler.sample(np.random.default_rng(1))
    _, setup_b = sampler.sample(np.random.default_rng(2))
    k = setup_a.knowledge
    assert k is not None and k.per_team and k.learning and setup_a.rival_rules is not None
    # a new hidden truth per episode
    assert setup_a.lap_model.compounds != setup_b.lap_model.compounds
    session = RaceSession(
        setup_a,
        seed=a.seed,
        agents=list(a.agents),
        scoring=a.scoring,
        observation_schema="rl_obs_v4",
        racecraft=False,
    )
    d = session.next_decision()
    assert d is not None and d.observation.shape == (len(RL_FEATURE_SETS["rl_obs_v4"]),)
    legal = [decode(int(i)) for i in np.flatnonzero(d.mask)]
    assert legal and all(intent is INTENT_OPTIONS[1] for _, _, intent in legal)


def test_regulation_guard_forces_an_unused_dry_compound(two_car_setup: RaceSetup) -> None:
    from f1ar.domain.race_state import Compound
    from f1ar.training.objectives import scoring_for

    scoring = scoring_for(two_car_setup, [VER], None)
    guarded = RaceSession(
        two_car_setup, seed=1, agents=[VER], scoring=scoring, regulation_guard=True
    )
    total = two_car_setup.track.total_laps
    car = guarded.state.car(VER).model_copy(
        update={"laps_completed": total - 2, "compounds_used": (Compound.MEDIUM,)}
    )
    guarded.state = guarded.state.model_copy(
        update={"cars": tuple(car if c.driver_id == VER else c for c in guarded.state.cars)}
    )
    pits = {decode(int(i))[0] for i in np.flatnonzero(guarded.action_mask(VER))}
    assert pits and pits <= {Compound.SOFT, Compound.HARD}
    guarded.regulation_guard = False
    assert None in {decode(int(i))[0] for i in np.flatnonzero(guarded.action_mask(VER))}
