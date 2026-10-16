"""M5 slice 3 (D-040): the training stack, the torch-free policy runtime and the policy registry
(pinning: hash, observation schema, features, actions, regulations, engine version)."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from stable_baselines3.common.vec_env import DummyVecEnv

from f1ar.domain.race_state import PaceMode, RaceSetup
from f1ar.policy.strategists import LearnedStrategist
from f1ar.simulation.lightweight_engine import ENGINE_VERSION
from f1ar.training.actions import N_ACTIONS, PER_PIT, PIT_OPTIONS, decode, encode
from f1ar.training.evaluate import evaluate
from f1ar.training.features import RL_FEATURE_NAMES, RL_OBSERVATION_SCHEMA
from f1ar.training.objectives import scoring_for
from f1ar.training.policy_registry import (
    NumpyPolicy,
    PolicyLoadError,
    PolicyManifest,
    feature_names_hash,
    list_policies,
    load_policy,
    manifest_path,
    register_policy,
    update_evaluation,
)
from f1ar.training.ppo import (
    TrainConfig,
    TrainingEnv,
    build_model,
    export_actor,
    rules_reward,
    torch_logits,
)
from f1ar.training.regulations import REGULATIONS_2024
from f1ar.training.scenarios import ScenarioSampler
from f1ar.training.session import Decision
from tests.conftest import VER

N_FEATURES = len(RL_FEATURE_NAMES)


def _random_policy(seed: int = 0) -> NumpyPolicy:
    rng = np.random.default_rng(seed)
    shapes = [(16, N_FEATURES), (16, 16), (N_ACTIONS, 16)]
    return NumpyPolicy(
        weights=tuple(rng.normal(0, 0.5, s) for s in shapes),
        biases=tuple(rng.normal(0, 0.5, s[0]) for s in shapes),
    )


def _fields(policy_id: str = "test_policy_v1", **over: Any) -> dict[str, Any]:
    return {
        "policy_id": policy_id,
        "algorithm": "test",
        "architecture": "MLP",
        "observation_schema": RL_OBSERVATION_SCHEMA,
        "feature_names_sha256": feature_names_hash(),
        "regulations": REGULATIONS_2024.version,
        "engine_version": ENGINE_VERSION,
        "training": {"seed": 0},
        "lineage": {"parent_policy": None, "season": 2024},
        "evaluation": {},
    } | over


# --- the numpy runtime ----------------------------------------------------------------------


def test_numpy_policy_respects_the_mask_and_round_trips(tmp_path: Path) -> None:
    policy = _random_policy()
    obs = np.random.default_rng(1).normal(size=N_FEATURES)
    mask = np.zeros(N_ACTIONS, dtype=bool)
    mask[[2, 7, 11]] = True
    assert policy.act(obs, mask) in (2, 7, 11)
    p = policy.probabilities(obs, mask)
    assert p[~mask].sum() == 0 and p.sum() == pytest.approx(1.0)
    assert policy.act(obs, mask) == int(np.argmax(p))
    policy.save(tmp_path / "p.npz")
    again = NumpyPolicy.load(tmp_path / "p.npz")
    assert np.array_equal(again.logits(obs), policy.logits(obs))
    with pytest.raises(ValueError, match="no legal action"):
        policy.act(obs, np.zeros(N_ACTIONS, dtype=bool))


# --- the registry: pinning and refusal ------------------------------------------------------


def test_registry_round_trip_and_evaluation_update(tmp_path: Path) -> None:
    m = register_policy(tmp_path, _random_policy(), _fields())
    assert m.n_features == N_FEATURES and m.n_actions == N_ACTIONS
    assert m.source_type == "learned_parameter"
    loaded = load_policy(
        tmp_path, "test_policy_v1", regulations="fia_2024_v1", engine_version=ENGINE_VERSION
    )
    assert loaded.manifest == m
    update_evaluation(tmp_path, "test_policy_v1", {"exp009": {"vs_rules": 0.01}})
    again = load_policy(tmp_path, "test_policy_v1")
    assert again.manifest.evaluation == {"exp009": {"vs_rules": 0.01}}
    assert again.manifest.artifact_sha256 == m.artifact_sha256  # weights unchanged
    assert [p.policy_id for p in list_policies(tmp_path)] == ["test_policy_v1"]


def test_registry_refuses_a_changed_artifact(tmp_path: Path) -> None:
    register_policy(tmp_path, _random_policy(), _fields())
    _random_policy(seed=99).save(tmp_path / "models" / "policies" / "test_policy_v1.npz")
    with pytest.raises(PolicyLoadError, match="hash"):
        load_policy(tmp_path, "test_policy_v1")


@pytest.mark.parametrize(
    ("over", "kwargs", "message"),
    [
        ({"observation_schema": "rl_obs_v0"}, {}, "observation"),
        ({"feature_names_sha256": "sha256:other"}, {}, "feature list"),
        ({}, {"regulations": "fia_2025_v1"}, "fia_2024_v1"),
        ({}, {"engine_version": "lightweight_engine_v9"}, "trained on"),
    ],
)
def test_registry_refuses_a_mismatched_pin(
    tmp_path: Path, over: dict[str, Any], kwargs: dict[str, str], message: str
) -> None:
    register_policy(tmp_path, _random_policy(), _fields(**over))
    with pytest.raises(PolicyLoadError, match=message):
        load_policy(tmp_path, "test_policy_v1", **kwargs)


def test_registry_refuses_missing_policies(tmp_path: Path) -> None:
    with pytest.raises(PolicyLoadError, match="no manifest"):
        load_policy(tmp_path, "nope_v1")


def test_registered_policies_in_the_repo_load() -> None:
    """Every tracked manifest matches its artifact and this code (hash, schema, features)."""
    root = Path(__file__).resolve().parents[2]
    for m in list_policies(root):
        assert manifest_path(root, m.policy_id).exists()
        loaded = load_policy(root, m.policy_id, regulations=m.regulations)
        assert isinstance(loaded.manifest, PolicyManifest)


# --- the strategist -------------------------------------------------------------------------


def test_learned_strategist_is_deterministic_and_legal(isolated_setup: RaceSetup) -> None:
    scoring = scoring_for(isolated_setup, [VER], None)[VER]
    s = LearnedStrategist(_random_policy(), "random_net")
    a = evaluate(isolated_setup, VER, 5, scoring, s)
    b = evaluate(isolated_setup, VER, 5, scoring, LearnedStrategist(_random_policy(), "x"))
    key = ("reward", "position", "stops", "sequence")
    assert [getattr(a, k) for k in key] == [getattr(b, k) for k in key]
    assert a.decisions > 0


def test_forced_pace_only_changes_the_pace(isolated_setup: RaceSetup) -> None:
    policy = _random_policy()
    forced = LearnedStrategist(policy, "net", pace=PaceMode.BALANCED)
    assert forced.name == "net+balanced"
    obs = np.random.default_rng(3).normal(size=N_FEATURES)
    mask = np.ones(N_ACTIONS, dtype=bool)
    d = Decision(VER, obs, mask, "LAP_END", 3)
    action = forced.decide(None, d)  # type: ignore[arg-type]
    pit, pace, _ = decode(action)
    assert pace is PaceMode.BALANCED
    by_pit = policy.probabilities(obs, mask).reshape(6, 9).sum(axis=1)
    assert action == encode(pit, PaceMode.BALANCED)
    assert pit is PIT_OPTIONS[int(by_pit.argmax())]


# --- training: env, export ------------------------------------------------------------------


def _cfg(**over: Any) -> TrainConfig:
    base: dict[str, Any] = {
        "events": ("x",),
        "total_steps": 64,
        "n_envs": 1,
        "n_steps": 32,
        "batch_size": 32,
        "n_epochs": 1,
        "device": "cpu",
        "net_arch": (16, 16),
    }
    return TrainConfig(**(base | over))


def test_paired_reward_subtracts_the_rule_planner(isolated_setup: RaceSetup) -> None:
    sampler = ScenarioSampler(["x"], lambda _e, _r: isolated_setup, championship_probability=0)
    env = TrainingEnv(sampler, reward_baseline="rules")
    env.reset(seed=3)
    stay = encode(None, PaceMode.BALANCED)
    while True:
        _, reward, done, _, info = env.step(stay)
        if done:
            break
    assert env.session is not None
    assert info["rules_score"] == pytest.approx(rules_reward(env.session, env.me))
    assert reward == pytest.approx(info["score"] - info["rules_score"])
    assert info["stops"] == 0 and info["push_share"] == 0.0
    # the rule planner's score is what evaluate() gives B1 on the same race
    scoring = env.session.scoring[env.me]
    b1 = evaluate(isolated_setup, env.me, env.session.state.rng.seed, scoring, None)
    assert info["rules_score"] == pytest.approx(b1.reward)


def test_exported_actor_matches_torch(isolated_setup: RaceSetup) -> None:
    sampler = ScenarioSampler(["x"], lambda _e, _r: isolated_setup, championship_probability=0)
    env = DummyVecEnv([lambda: TrainingEnv(sampler)])
    model = build_model(_cfg(), env)
    with torch.no_grad():  # non-zero biases so the test covers them
        for p in model.policy.parameters():
            p.add_(torch.randn_like(p) * 0.1)
    obs = np.random.default_rng(0).normal(size=(5, N_FEATURES)).astype(np.float32)
    policy = export_actor(model)
    assert policy.n_inputs == N_FEATURES and policy.n_actions == N_ACTIONS
    assert np.allclose(policy.logits(obs.astype(np.float64)), torch_logits(model, obs), atol=1e-5)
    env.close()


def test_a_tiny_training_run_learns_without_errors(isolated_setup: RaceSetup) -> None:
    sampler = ScenarioSampler(["x"], lambda _e, _r: isolated_setup, championship_probability=0)
    env = DummyVecEnv([lambda: TrainingEnv(sampler)])
    model = build_model(_cfg(), env)
    model.learn(total_timesteps=64)
    assert model.num_timesteps >= 64
    env.close()


def _isolated_env() -> TrainingEnv:
    """Module-level factory (picklable for spawned workers): VER + HAM, no traffic. Its sampler
    holds a local function, like the real one (``repo_setup_factory``): it must never be sent
    back from a worker."""
    from f1ar.domain.driver_profiles import load_profiles
    from f1ar.domain.identities import load_roster
    from f1ar.simulation.scenarios import (
        build_setup,
        default_race_control_params,
        default_traffic_params,
    )
    from tests.conftest import CONFIGS, HAM, MONZA_2024

    roster = load_roster(CONFIGS / "rosters" / f"{MONZA_2024}.json", expected_event_id=MONZA_2024)
    profiles = load_profiles(
        CONFIGS / "drivers" / "seed_2024_monza_drivers.yaml",
        CONFIGS / "drivers" / "archetypes.yaml",
        roster,
    )
    setup = build_setup(roster, profiles, [VER, HAM]).model_copy(
        update={
            "traffic": default_traffic_params(enabled=False),
            "race_control": default_race_control_params(enabled=False),
        }
    )
    sampler = ScenarioSampler(["x"], lambda _e, _r: setup, championship_probability=0)
    return TrainingEnv(sampler)


def test_grouped_vec_env_matches_dummy_vec_env() -> None:
    from f1ar.training.vec_env import GroupedSubprocVecEnv

    grouped = GroupedSubprocVecEnv([_isolated_env] * 4, workers=2)
    dummy = DummyVecEnv([_isolated_env] * 4)
    try:
        grouped.seed(11)
        dummy.seed(11)
        assert np.array_equal(np.asarray(grouped.reset()), np.asarray(dummy.reset()))
        rng = np.random.default_rng(0)
        for _ in range(60):
            masks = np.array(dummy.env_method("action_masks"))
            assert np.array_equal(np.array(grouped.env_method("action_masks")), masks)
            actions = np.array([rng.choice(np.flatnonzero(m)) for m in masks])
            go, gr, gd, gi = grouped.step(actions)
            do, dr, dd, di = dummy.step(actions)
            assert np.array_equal(np.asarray(go), np.asarray(do))
            assert np.array_equal(gr, dr) and np.array_equal(gd, dd)
            assert [i.get("score") for i in gi] == [i.get("score") for i in di]
        assert grouped.get_attr("me") == dummy.get_attr("me")
        assert grouped.has_attr("action_masks") and not grouped.has_attr("no_such_attr")
    finally:
        grouped.close()
        dummy.close()


def test_stay_prior_sets_the_untrained_stop_rate(isolated_setup: RaceSetup) -> None:
    sampler = ScenarioSampler(["x"], lambda _e, _r: isolated_setup, championship_probability=0)
    env = DummyVecEnv([lambda: TrainingEnv(sampler)])
    model = build_model(_cfg(stay_prior=0.97), env)
    policy = export_actor(model)
    obs = np.random.default_rng(0).normal(0, 0.5, size=(50, N_FEATURES))
    p = policy.probabilities(obs, np.ones((50, N_ACTIONS), dtype=bool))
    stay = p[:, :PER_PIT].sum(axis=1)  # the stay-out block (any pace, any intent)
    assert np.allclose(stay, 0.97, atol=0.01)
    assert np.allclose(p[:, :PER_PIT], stay[:, None] / PER_PIT, atol=0.01)  # uniform inside
    with pytest.raises(ValueError):
        build_model(_cfg(stay_prior=1.0), env)
    env.close()


def test_sampled_play_is_reproducible_per_race(isolated_setup: RaceSetup) -> None:
    scoring = scoring_for(isolated_setup, [VER], None)[VER]

    def run(seed: int) -> list[Any]:
        s = LearnedStrategist(_random_policy(), "net", sample_seed=0)
        r = evaluate(isolated_setup, VER, seed, scoring, s)
        return [r.reward, r.stops, r.sequence, r.method]

    assert run(5) == run(5)
    assert run(5)[3] == "net~sampled"


def test_behaviour_cloning_learns_a_rare_stop_rule(
    isolated_setup: RaceSetup, tmp_path: Path
) -> None:
    """Stops are ~2 % of the teacher's decisions; weighted cloning must still learn them."""
    from f1ar.training.warm_start import Demonstrations, behaviour_clone

    rng = np.random.default_rng(0)
    n = 8000  # rl_obs_v2 is 86 wide: with 4,000 noisy rows the rare rule is seed-dependent
    obs = rng.uniform(0, 1, size=(n, N_FEATURES)).astype(np.float32)
    stop = obs[:, 0] > 0.97  # a sharp, rare rule
    stop_action = encode(PIT_OPTIONS[3], PaceMode.BALANCED)
    actions = np.where(stop, stop_action, encode(None, PaceMode.BALANCED))
    demos = Demonstrations(
        obs=obs,
        masks=np.ones((n, N_ACTIONS), dtype=bool),
        actions=actions,
        returns=obs[:, 1].astype(np.float64),
        episode=np.arange(n) // 40,
    )
    demos.save(tmp_path / "d.npz")
    again = Demonstrations.load(tmp_path / "d.npz")
    assert np.array_equal(again.actions, demos.actions) and again.episodes == 200
    sampler = ScenarioSampler(["x"], lambda _e, _r: isolated_setup, championship_probability=0)
    env = DummyVecEnv([lambda: TrainingEnv(sampler)])
    model = build_model(_cfg(net_arch=(64, 64)), env)
    stats = behaviour_clone(model, demos, epochs=150, batch_size=512, stop_weight=5.0)
    assert stats["train"]["stop_recall"] > 0.8
    # with ~3 % stops "never stop" alone scores 0.97: recall is the real test, accuracy a bound
    assert stats["train"]["pit_pace_accuracy"] > 0.95
    # slice 4: the intent isn't imitated (the rules can't attack): it stays about uniform
    shares = stats["train"]["intent_mean_probability"]
    assert min(shares.values()) > 0.2
    assert set(stats) >= {"train", "validation", "stop_share"}
    env.close()


def test_soften_keeps_the_choices_and_raises_entropy(isolated_setup: RaceSetup) -> None:
    from f1ar.training.warm_start import Demonstrations, soften

    sampler = ScenarioSampler(["x"], lambda _e, _r: isolated_setup, championship_probability=0)
    env = DummyVecEnv([lambda: TrainingEnv(sampler)])
    model = build_model(_cfg(net_arch=(16, 16)), env)
    head = model.policy.action_net
    assert isinstance(head, torch.nn.Linear)
    with torch.no_grad():  # a confident policy, like a cloned one
        head.weight.mul_(50000.0)  # 54 actions: needs a sharper head than 18 did
    rng = np.random.default_rng(1)
    obs = rng.normal(0, 1, size=(500, N_FEATURES)).astype(np.float32)
    masks = np.ones((500, N_ACTIONS), dtype=bool)
    demos = Demonstrations(obs, masks, np.zeros(500, np.int64), np.zeros(500), np.arange(500))
    before = export_actor(model)
    t = soften(model, demos, 0.25)
    after = export_actor(model)
    assert t > 1.0
    o = obs.astype(np.float64)
    assert np.array_equal(before.logits(o).argmax(axis=1), after.logits(o).argmax(axis=1))
    p = after.probabilities(o, masks).reshape(len(o), -1, 3).sum(axis=2)  # (pit, pace) marginal
    entropy = float(-(p * np.log(np.clip(p, 1e-12, None))).sum(axis=1).mean())
    assert entropy == pytest.approx(0.25, abs=0.01)
    env.close()
