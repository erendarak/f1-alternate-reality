"""Training the strategy policy with MaskablePPO (M5 slice 3, D-040; needs the ``train`` group).

- **Workers:** ``n_envs`` CPU processes (``GroupedSubprocVecEnv``), each stepping
  ``envs_per_worker`` envs (``TrainingEnv``): one learning car vs the rule field on the training
  events (the scenario sampler picks event, car and seed). The network update runs on
  ``device`` (CUDA on the owner's RTX 4070).
- **Reward:** the env's terminal score (the car's objective under its risk utility, D-038).
  ``reward_baseline="rules"`` subtracts the score the *rule planner* gets for the same car in the
  same race (same setup and seed: common random numbers), a control variate: it doesn't depend
  on the policy's actions, so the policy gradient keeps its expectation, but much of the noise
  from incidents and safety cars cancels. Costs one extra rule-only race per episode.
- **Progress:** one line per rollout (steps/s, ETA, episode stats) and, every ``eval_every``
  steps, a paired check of the current policy vs the rule planner on fixed *training* cases;
  each check also writes a checkpoint. The held-out evaluation is a separate experiment (EXP-009).
- **Export:** the actor network becomes a ``NumpyPolicy`` (``policy_registry``), the form a world
  pins; ``torch_logits`` lets a test check that both agree.
"""

import csv
import json
import statistics
import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecNormalize

from f1ar.domain.race_state import PaceMode, RacecraftIntent
from f1ar.policy.strategists import LearnedStrategist
from f1ar.simulation import lightweight_engine as engine
from f1ar.training.actions import PER_PIT, PIT_OPTIONS, intent_of, pace_of
from f1ar.training.evaluate import evaluate
from f1ar.training.gym_env import RaceStrategyEnv
from f1ar.training.objectives import race_results, scoring_for
from f1ar.training.policy_registry import NumpyPolicy
from f1ar.training.scenarios import ScenarioSampler, repo_setup_factory
from f1ar.training.session import RaceSession
from f1ar.training.vec_env import GroupedSubprocVecEnv

REWARD_BASELINES = ("none", "rules")


@dataclass(frozen=True)
class TrainConfig:
    events: tuple[str, ...]
    total_steps: int
    seed: int = 0
    n_envs: int = 14
    """Worker processes."""
    envs_per_worker: int = 4
    """Envs stepped in a row by each worker (``GroupedSubprocVecEnv``: less lockstep waiting)."""
    n_steps: int = 128
    """Decisions per env per rollout (one update per ``n_envs x envs_per_worker x n_steps``)."""
    batch_size: int = 1792
    n_epochs: int = 5
    learning_rate: float = 3e-4
    gamma: float = 1.0
    """Terminal reward only: no discounting within a race."""
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    net_arch: tuple[int, ...] = (128, 128)
    reward_baseline: str = "none"
    stay_prior: float = 0.0
    """> 0: the untrained policy stays out with this probability (``set_stay_prior``)."""
    warm_start_episodes: int = 0
    """Races played by the rules teacher for behaviour cloning before PPO (0 = from scratch)."""
    bc_epochs: int = 150
    bc_stop_weight: float = 5.0
    bc_target_entropy: float = 0.5
    """After cloning, soften the policy to this mean entropy (nats) so PPO explores (0 = off)."""
    normalize_reward: bool = False
    """Scale rewards by a running std of returns (``VecNormalize``, observations untouched).
    Pilot A (D-040): near-random play scores only the 0.02 position tie-break, a reward scale the
    value net can't resolve; scaling doesn't change which policy is best."""
    championship_probability: float = 0.0
    """Slice 3 trains for points (no championship context); objectives come in slice 4."""
    rain_probability: float = 0.0
    tyre_randomisation: float = 0.0
    character_randomisation: bool = False
    """Redraw the learning car's risk appetite per race (``randomise_character``, slice 4)."""
    """Sd of the per-race tyre/pit-loss perturbation (``training.scenarios.randomise_tyres``)."""
    device: str = "cuda"
    eval_every: int = 100_000
    eval_events: tuple[str, ...] = ()
    eval_slots: tuple[int, ...] = (1, 8, 15)
    eval_seeds: tuple[int, ...] = (1001, 1002)

    def __post_init__(self) -> None:
        if self.reward_baseline not in REWARD_BASELINES:
            raise ValueError(f"reward_baseline must be one of {REWARD_BASELINES}")


def rules_session(session: RaceSession) -> RaceSession:
    """The same race (setup, seed) with no AI car: every car follows the rule planner."""
    return RaceSession(
        session.setup,
        seed=session.state.rng.seed,
        agents=[],
        scoring={},
        context=session.context,
        regulations=session.regulations,
    )


def advance(rules: RaceSession, until_s: float | None = None) -> None:
    """Run a rules-only race to sim time ``until_s`` (None: to the end)."""
    while not rules.done and (until_s is None or rules.state.sim_clock_s < until_s):
        rules.state = engine.step(rules.state).state


def rules_reward(session: RaceSession, driver_id: str, rules: RaceSession | None = None) -> float:
    """The score ``driver_id`` gets in the same race (setup, seed, scoring) under the rule
    planner: the car isn't AI-run at all."""
    rules = rules if rules is not None else rules_session(session)
    advance(rules)
    results = race_results(rules.state)
    return session.scoring[driver_id].reward(driver_id, results, len(session.setup.drivers))


class TrainingEnv(RaceStrategyEnv):
    """``RaceStrategyEnv`` with the optional paired reward and per-episode statistics.

    The paired reward's rules-only race runs *alongside* the AI race: after every decision it is
    advanced to the same sim time. Run in one go at the episode end it stalled every worker in
    lockstep (main run attempt 1: 53 instead of ~100 decisions/s)."""

    def __init__(self, sampler: ScenarioSampler, *, reward_baseline: str = "none") -> None:
        super().__init__(sampler, policy_id="training")
        self.reward_baseline = reward_baseline
        self._pace: list[PaceMode] = []
        self._intent: list[RacecraftIntent] = []
        self._rules: RaceSession | None = None

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        self._pace = []
        self._intent = []
        obs, info = super().reset(seed=seed, options=options)
        if self.reward_baseline == "rules":
            assert self.session is not None
            self._rules = rules_session(self.session)
            advance(self._rules, self.session.state.sim_clock_s)
        return obs, info

    def step(self, action: np.int64 | int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self._pace.append(pace_of(int(action)))
        self._intent.append(intent_of(int(action)))
        obs, reward, done, truncated, info = super().step(action)
        assert self.session is not None
        if self._rules is not None and not done:
            advance(self._rules, self.session.state.sim_clock_s)
        if done:
            info["score"] = reward
            info["stops"] = self.session.state.car(self.me).pit_stops
            info["decisions"] = len(self._pace)
            info["push_share"] = self._pace.count(PaceMode.PUSH) / len(self._pace)
            info["conserve_share"] = self._pace.count(PaceMode.CONSERVE) / len(self._pace)
            info["attack_share"] = self._intent.count(RacecraftIntent.ATTACK) / len(self._intent)
            info["defend_share"] = self._intent.count(RacecraftIntent.DEFEND) / len(self._intent)
            if self.reward_baseline == "rules":
                base = rules_reward(self.session, self.me, self._rules)
                info["rules_score"] = base
                reward = reward - base
        return obs, reward, done, truncated, info


def make_env(root: Path, cfg: TrainConfig) -> TrainingEnv:
    sampler = ScenarioSampler(
        cfg.events,
        repo_setup_factory(root),
        agents=1,
        rain_probability=cfg.rain_probability,
        championship_probability=cfg.championship_probability,
        tyre_randomisation=cfg.tyre_randomisation,
        character_randomisation=cfg.character_randomisation,
    )
    return TrainingEnv(sampler, reward_baseline=cfg.reward_baseline)


def make_vec_env(root: Path, cfg: TrainConfig, *, subprocess: bool = True) -> VecEnv:
    n = cfg.n_envs * cfg.envs_per_worker
    fns: list[Callable[[], Any]] = [partial(make_env, root, cfg) for _ in range(n)]
    vec: VecEnv = GroupedSubprocVecEnv(fns, cfg.n_envs) if subprocess else DummyVecEnv(fns)
    vec.seed(cfg.seed)
    if cfg.normalize_reward:
        vec = VecNormalize(vec, norm_obs=False, norm_reward=True, gamma=cfg.gamma)
    return vec


def build_model(cfg: TrainConfig, env: VecEnv, run_dir: Path | None = None) -> MaskablePPO:
    model = MaskablePPO(
        "MlpPolicy",
        env,
        learning_rate=cfg.learning_rate,
        n_steps=cfg.n_steps,
        batch_size=cfg.batch_size,
        n_epochs=cfg.n_epochs,
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
        clip_range=cfg.clip_range,
        ent_coef=cfg.ent_coef,
        vf_coef=cfg.vf_coef,
        max_grad_norm=cfg.max_grad_norm,
        policy_kwargs={
            "net_arch": {"pi": list(cfg.net_arch), "vf": list(cfg.net_arch)},
            "activation_fn": torch.nn.Tanh,
        },
        seed=cfg.seed,
        device=cfg.device,
        verbose=0,
    )
    if cfg.stay_prior > 0.0:
        set_stay_prior(model, cfg.stay_prior)
    if run_dir is not None:
        model.set_logger(configure(str(run_dir), ["csv"]))
    return model


def set_stay_prior(model: MaskablePPO, p_stay: float) -> None:
    """Set the action head's bias so the *untrained* policy stays out with probability
    ``p_stay`` (pace modes uniform; the stop share split evenly over the 5 compounds). The
    head's weights start near zero (orthogonal init, gain 0.01), so the bias sets the initial
    behaviour. Nothing about *when* or *for what* to stop is given: only how often stops are
    tried while exploring (pilots A/B, D-040: near-random play stops on most laps, finishes last
    whatever it does, and gets no gradient)."""
    if not 0.0 < p_stay < 1.0:
        raise ValueError("p_stay must be in (0, 1)")
    n_stay = PER_PIT
    n_stop = (len(PIT_OPTIONS) - 1) * PER_PIT
    stop_logit = float(np.log(((1.0 - p_stay) / n_stop) / (p_stay / n_stay)))
    head = model.policy.action_net
    assert isinstance(head, torch.nn.Linear)
    with torch.no_grad():
        head.bias.copy_(torch.tensor([0.0] * n_stay + [stop_logit] * n_stop, dtype=head.bias.dtype))


def export_actor(model: MaskablePPO) -> NumpyPolicy:
    """The actor MLP (``mlp_extractor.policy_net`` + ``action_net``) as numpy weights."""
    policy = model.policy
    layers = [m for m in policy.mlp_extractor.policy_net if isinstance(m, torch.nn.Linear)]
    head = policy.action_net
    assert isinstance(head, torch.nn.Linear)
    layers.append(head)
    acts = {type(m) for m in policy.mlp_extractor.policy_net if not isinstance(m, torch.nn.Linear)}
    if acts != {torch.nn.Tanh}:
        raise ValueError(f"export supports tanh MLPs only, got {acts}")
    return NumpyPolicy(
        weights=tuple(layer.weight.detach().cpu().double().numpy() for layer in layers),
        biases=tuple(layer.bias.detach().cpu().double().numpy() for layer in layers),
        activation="tanh",
    )


def torch_logits(model: MaskablePPO, obs: np.ndarray) -> np.ndarray:
    policy = model.policy
    with torch.no_grad():
        x = torch.as_tensor(obs, dtype=torch.float32, device=policy.device)
        features = policy.extract_features(x, policy.pi_features_extractor)
        assert isinstance(features, torch.Tensor)
        latent = policy.mlp_extractor.forward_actor(features)
        out: np.ndarray = policy.action_net(latent).cpu().double().numpy()
    return out


# --- progress and periodic checks ----------------------------------------------------------


@dataclass
class EvalCase:
    event_id: str
    driver_id: str
    seed: int
    rules_reward: float


def eval_cases(root: Path, cfg: TrainConfig) -> list[EvalCase]:
    """Fixed training-track cases with the rule planner's score (computed once)."""
    factory = repo_setup_factory(root)
    out = []
    for event_id in cfg.eval_events:
        setup = factory(event_id, False)
        by_grid = {d.grid_position: d.driver_id for d in setup.drivers}
        for slot in cfg.eval_slots:
            driver = by_grid[slot]
            scoring = scoring_for(setup, [driver], None)[driver]
            for seed in cfg.eval_seeds:
                r = evaluate(setup, driver, seed, scoring, None)
                out.append(EvalCase(event_id, driver, seed, r.reward))
    return out


def paired_check(
    root: Path, policy: NumpyPolicy, cases: Sequence[EvalCase], *, sampled: bool = False
) -> dict[str, Any]:
    """The policy vs the rule planner on fixed cases: its most likely action, or ``sampled``
    from its distribution (seeded per race)."""
    factory = repo_setup_factory(root)
    diffs, stops, points = [], [], []
    for c in cases:
        setup = factory(c.event_id, False)
        scoring = scoring_for(setup, [c.driver_id], None)[c.driver_id]
        s = LearnedStrategist(policy, "check", sample_seed=0 if sampled else None)
        r = evaluate(setup, c.driver_id, c.seed, scoring, s)
        diffs.append(r.reward - c.rules_reward)
        stops.append(r.stops)
        points.append(r.points)
    return {
        "vs_rules": round(statistics.fmean(diffs), 4),
        "better": sum(d > 1e-9 for d in diffs),
        "worse": sum(d < -1e-9 for d in diffs),
        "stops": round(statistics.fmean(stops), 2),
        "points": round(statistics.fmean(points), 2),
    }


@dataclass
class ProgressLog:
    rows: list[dict[str, Any]] = field(default_factory=list)


class ProgressCallback(BaseCallback):
    def __init__(
        self,
        root: Path,
        cfg: TrainConfig,
        run_dir: Path,
        cases: Sequence[EvalCase],
        log: ProgressLog,
    ) -> None:
        super().__init__()
        self.root = root
        self.cfg = cfg
        self.run_dir = run_dir
        self.cases = cases
        self.log = log
        self.episodes: deque[dict[str, Any]] = deque(maxlen=500)
        self.total_episodes = 0
        self.t0 = time.perf_counter()
        self.next_eval = cfg.eval_every
        self.best: float | None = None

    def _on_step(self) -> bool:
        for done, info in zip(self.locals["dones"], self.locals["infos"], strict=True):
            if done and "score" in info:
                self.episodes.append(info)
                self.total_episodes += 1
        return True

    def _on_rollout_end(self) -> None:
        steps = self.num_timesteps
        elapsed = time.perf_counter() - self.t0
        rate = steps / max(elapsed, 1e-9)
        eta = (self.cfg.total_steps - steps) / max(rate, 1e-9)
        recent = list(self.episodes)[-200:]
        row: dict[str, Any] = {
            "steps": steps,
            "minutes": round(elapsed / 60, 2),
            "steps_per_s": round(rate, 1),
            "episodes": self.total_episodes,
        }
        if recent:

            def mean(key: str) -> float:
                return round(statistics.fmean(float(e[key]) for e in recent), 4)

            row |= {
                "score": mean("score"),
                "points": mean("points"),
                "position": mean("position"),
                "stops": mean("stops"),
                "push": mean("push_share"),
                "conserve": mean("conserve_share"),
                "attack": mean("attack_share"),
                "defend": mean("defend_share"),
                "dsq": mean("disqualified"),
            }
            if "rules_score" in recent[0]:
                row["vs_rules_train"] = round(
                    statistics.fmean(e["score"] - e["rules_score"] for e in recent), 4
                )
        logged = self.model.logger.name_to_value
        for key in ("train/entropy_loss", "train/approx_kl", "train/explained_variance"):
            if key in logged:
                row[key.split("/")[1]] = round(float(logged[key]), 4)
        if steps >= self.next_eval and self.cases:
            self.next_eval += self.cfg.eval_every
            row |= self._check(steps)
        self.log.rows.append(row)
        print(
            f"[{row['minutes']:6.1f} min] {steps:>9,} steps {rate:6.0f}/s "
            f"ETA {eta / 60:5.1f} min | "
            + " ".join(f"{k}={v}" for k, v in row.items() if k not in ("steps", "minutes")),
            flush=True,
        )
        with (self.run_dir / "progress.jsonl").open("a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(row) + "\n")

    def _check(self, steps: int) -> dict[str, Any]:
        t = time.perf_counter()
        policy = export_actor(self.model)  # type: ignore[arg-type]
        result = paired_check(self.root, policy, self.cases)
        sampled = paired_check(self.root, policy, self.cases, sampled=True)
        ckpt = self.run_dir / "checkpoints" / f"step_{steps}"
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(ckpt))
        policy.save(ckpt.with_suffix(".npz"))
        if self.best is None or result["vs_rules"] > self.best:
            self.best = result["vs_rules"]
            self.model.save(str(self.run_dir / "best"))
            policy.save(self.run_dir / "best.npz")
            (self.run_dir / "best.json").write_text(
                json.dumps({"steps": steps, **result}) + "\n", encoding="utf-8", newline="\n"
            )
        return (
            {f"check_{k}": v for k, v in result.items()}
            | {f"sampled_{k}": v for k, v in sampled.items()}
            | {"check_s": round(time.perf_counter() - t, 1)}
        )


def train(root: Path, cfg: TrainConfig, run_dir: Path) -> tuple[MaskablePPO, ProgressLog]:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(
        json.dumps(asdict(cfg), indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"computing the rule planner on {len(cfg.eval_events)} check events ...", flush=True)
    t0 = time.perf_counter()
    cases = eval_cases(root, cfg)
    print(f"  {len(cases)} cases in {time.perf_counter() - t0:.0f} s", flush=True)
    demos = None
    if cfg.warm_start_episodes > 0:
        from f1ar.training.warm_start import collect

        t0 = time.perf_counter()
        n = cfg.warm_start_episodes
        print(f"warm start: the rules teacher plays {n} races ...", flush=True)
        demos = collect(root, cfg, n, cfg.n_envs)
        elapsed = time.perf_counter() - t0
        print(f"  {len(demos.actions):,} decisions in {elapsed:.0f} s", flush=True)
    env = make_vec_env(root, cfg)
    try:
        model = build_model(cfg, env, run_dir)
        if demos is not None:
            from f1ar.training.warm_start import behaviour_clone

            stats: dict[str, Any] = behaviour_clone(
                model, demos, epochs=cfg.bc_epochs, stop_weight=cfg.bc_stop_weight, seed=cfg.seed
            )
            if cfg.bc_target_entropy > 0:
                from f1ar.training.warm_start import soften

                stats["temperature"] = round(soften(model, demos, cfg.bc_target_entropy), 3)
            export_actor(model).save(run_dir / "clone.npz")  # the softened clone, before PPO
            if cases:
                stats["check_before_ppo"] = paired_check(root, export_actor(model), cases)
                stats["sampled_before_ppo"] = paired_check(
                    root, export_actor(model), cases, sampled=True
                )
            print(f"warm start: {stats}", flush=True)
            (run_dir / "warm_start.json").write_text(
                json.dumps(stats, indent=1) + "\n", encoding="utf-8", newline="\n"
            )
        log = ProgressLog()
        callback = ProgressCallback(root, cfg, run_dir, cases, log)
        model.learn(total_timesteps=cfg.total_steps, callback=callback)
        model.save(str(run_dir / "final"))
        export_actor(model).save(run_dir / "final.npz")
    finally:
        env.close()
    with (run_dir / "progress.csv").open("w", encoding="utf-8", newline="") as f:
        keys = sorted({k for r in log.rows for k in r})
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(log.rows)
    return model, log
