"""M5 slice 4b (D-049): counterfactual races on identical luck, and the learned racecraft."""

import numpy as np

from f1ar.domain.race_state import RacecraftIntent, RaceSetup
from f1ar.policy.strategists import RacecraftLearned, RulesTeacher
from f1ar.training.actions import intent_of
from f1ar.training.counterfactual import _teacher_normal, decision_targets, finish, fork
from f1ar.training.features import RL_V3_NAMES
from f1ar.training.objectives import scoring_for
from f1ar.training.policy_registry import NumpyPolicy
from f1ar.training.session import RaceSession
from tests.conftest import VER


def _session(setup: RaceSetup, seed: int = 3) -> RaceSession:
    return RaceSession(setup, seed=seed, agents=[VER], scoring=scoring_for(setup, [VER], None))


def test_a_fork_replays_the_same_luck(two_car_setup: RaceSetup) -> None:
    choose = _teacher_normal(RulesTeacher())
    main = _session(two_car_setup)
    for _ in range(10):
        d = main.next_decision()
        assert d is not None
        main.act(VER, choose(main, d))
    a, b = fork(main), fork(main)
    finish(a, choose)
    finish(b, choose)
    finish(main, choose)
    assert a.state.canonical_hash() == b.state.canonical_hash() == main.state.canonical_hash()
    assert a.rewards() == main.rewards()


def test_targets_are_exact_differences(two_car_setup: RaceSetup) -> None:
    choose = _teacher_normal(RulesTeacher())
    s = _session(two_car_setup)
    d = s.next_decision()
    assert d is not None
    t = decision_targets(s, d, laps=3)
    assert t is not None
    rest = fork(s)
    finish(rest, choose)
    assert t.rewards[1] == rest.rewards()[VER]  # NORMAL branch = the race as it would be
    assert np.allclose(t.advantages, [t.rewards[0] - t.rewards[1], t.rewards[2] - t.rewards[1]])


def _model(attack: float, defend: float) -> NumpyPolicy:
    n = len(RL_V3_NAMES)
    return NumpyPolicy(
        weights=(np.zeros((2, n)),), biases=(np.array([defend, attack]),), activation="tanh"
    )


def test_the_learned_racecraft_follows_its_advantages(two_car_setup: RaceSetup) -> None:
    s = _session(two_car_setup)
    d = s.next_decision()
    assert d is not None
    assert intent_of(RacecraftLearned(_model(0.05, -0.01)).decide(s, d)) is RacecraftIntent.ATTACK
    assert intent_of(RacecraftLearned(_model(-0.01, 0.05)).decide(s, d)) is RacecraftIntent.DEFEND
    # below the margin: NORMAL
    assert intent_of(RacecraftLearned(_model(0.001, 0.0)).decide(s, d)) is RacecraftIntent.NORMAL


def test_pit_targets_compare_the_rules_with_the_one_lap_alternative(
    two_car_setup: RaceSetup,
) -> None:
    from f1ar.training.counterfactual import pit_targets

    choose = _teacher_normal(RulesTeacher())
    s = _session(two_car_setup)
    t = None
    while (d := s.next_decision()) is not None and t is None:
        t = pit_targets(s, d)
        s.act(VER, choose(s, d))
    assert t is not None
    rest = _session(two_car_setup)
    finish(rest, choose)
    assert t.rules == rest.rewards()[VER]  # the rules branch is the race as it would be
    assert t.advantage == t.alternative - t.rules


def test_the_learned_pit_call_deviates_only_when_predicted_better(
    two_car_setup: RaceSetup,
) -> None:
    from f1ar.policy.strategists import PitLearned
    from f1ar.training.actions import pit_of

    def model(value: float) -> NumpyPolicy:
        n = len(RL_V3_NAMES)
        return NumpyPolicy(weights=(np.zeros((1, n)),), biases=(np.array([value]),))

    teacher = RulesTeacher()
    s = _session(two_car_setup)
    d = s.next_decision()
    assert d is not None
    # far from the planned stop: no deviation whatever the model says
    yes = PitLearned(model(1.0), margin=0.0, window=0)
    assert pit_of(yes.decide(s, d)) is pit_of(teacher.decide(s, d))
    always = PitLearned(model(1.0), margin=0.0, window=99)
    never = PitLearned(model(-1.0), margin=0.0, window=99)
    assert pit_of(never.decide(s, d)) is pit_of(teacher.decide(s, d))
    assert pit_of(always.decide(s, d)) is not pit_of(teacher.decide(s, d))
    assert always.deviations == 1 and never.deviations == 0
