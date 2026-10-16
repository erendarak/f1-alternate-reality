"""M5 slice 3b (D-047): rl_obs_v2, the rules teacher at the rule pace, tyre randomisation."""

import numpy as np
import pytest

from f1ar.domain.race_state import DRY_COMPOUNDS, Compound, RaceSetup, WorldState
from f1ar.policy.baseline_strategy import plan_for_car
from f1ar.policy.strategists import RulesTeacher
from f1ar.training.evaluate import evaluate
from f1ar.training.features import (
    RL_FEATURE_SETS,
    RL_V1_NAMES,
    RL_V2_NAMES,
    rl_observation,
    schema_for_inputs,
)
from f1ar.training.objectives import Objective, TeamOrder, scoring_for
from f1ar.training.scenarios import randomise_tyres
from tests.conftest import VER, run_ticks


def test_v2_extends_v1_and_schemas_are_recognised_by_size() -> None:
    assert RL_V2_NAMES[: len(RL_V1_NAMES)] == RL_V1_NAMES
    assert len(set(RL_V2_NAMES)) == len(RL_V2_NAMES)
    assert {schema_for_inputs(len(n)) for n in RL_FEATURE_SETS.values()} == set(RL_FEATURE_SETS)
    with pytest.raises(ValueError):
        schema_for_inputs(3)


def test_v2_reads_the_tyres_and_the_rule_plan(two_car_world: WorldState) -> None:
    state = run_ticks(two_car_world, 600)
    order = TeamOrder(Objective.POINTS)
    v1 = rl_observation(state, VER, order, None, "rl_obs_v1")
    v2 = rl_observation(state, VER, order, None, "rl_obs_v2")
    assert v1.shape == (len(RL_V1_NAMES),) and (v2[: len(v1)] == v1).all()
    assert np.isfinite(v2).all() and (np.abs(v2) <= 10).all()
    f = dict(zip(RL_V2_NAMES, v2, strict=True))
    car = state.car(VER)
    plan = plan_for_car(state.setup, car)
    assert f["plan.has_stop"] == float(bool(plan))
    if plan:
        assert f["plan.laps_to_stop"] == pytest.approx((plan[0].lap - car.laps_completed) / 40)
        assert f[f"plan.compound.{plan[0].compound.value.lower()}"] == 1.0
    assert sum(f[f"plan.pace.{m}"] for m in ("conserve", "balanced", "push")) == 1.0
    # the track features follow the setup (what randomisation changes)
    soft = state.setup.lap_model.compounds[Compound.SOFT]
    assert f["track.soft.wear_s_per_lap"] == pytest.approx(soft.deg_s_per_lap * 10, rel=1e-6)


def test_randomise_tyres(isolated_setup: RaceSetup) -> None:
    assert randomise_tyres(isolated_setup, np.random.default_rng(0), 0.0) is isolated_setup
    a = randomise_tyres(isolated_setup, np.random.default_rng(3), 0.3)
    b = randomise_tyres(isolated_setup, np.random.default_rng(3), 0.3)
    assert a == b  # reproducible from the sampler's generator
    before, after = isolated_setup.lap_model.compounds, a.lap_model.compounds
    assert all(after[c].deg_s_per_lap != before[c].deg_s_per_lap for c in DRY_COMPOUNDS)
    assert after[Compound.MEDIUM].offset_s == before[Compound.MEDIUM].offset_s
    assert a.pit.lane_loss_s != isolated_setup.pit.lane_loss_s
    assert a.inputs["tyre_randomisation"].startswith("editorial_assumption")


def _with_tyre_life(setup: RaceSetup) -> RaceSetup:
    lives = {Compound.SOFT: 14.0, Compound.MEDIUM: 22.0, Compound.HARD: 32.0}
    compounds = {
        c: (cp.model_copy(update={"life_laps": lives[c]}) if c in lives else cp)
        for c, cp in setup.lap_model.compounds.items()
    }
    lap_model = setup.lap_model.model_copy(update={"compounds": compounds, "cliff_s_per_lap": 0.5})
    return setup.model_copy(update={"lap_model": lap_model})


def test_rules_teacher_plays_like_the_rule_planner(isolated_setup: RaceSetup) -> None:
    """With nothing else on track and tyre life on (so the rule pace manages and pushes), the
    teacher (stops + rule pace) reproduces B1's race."""
    setup = _with_tyre_life(isolated_setup)
    scoring = scoring_for(setup, [VER], None)[VER]
    for seed in (1, 2):
        rules = evaluate(setup, VER, seed, scoring, None)
        teacher = evaluate(setup, VER, seed, scoring, RulesTeacher())
        assert (teacher.sequence, teacher.stops) == (rules.sequence, rules.stops)
        assert teacher.position == rules.position
