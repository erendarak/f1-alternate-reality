"""Warm start from the rule planner: imitation before PPO (M5 slice 3, D-040; optional).

``RulesTeacher`` (B1's planner expressed as actions) plays the learning car on training
scenarios; its decisions become a dataset (observation, mask, action) and each decision's
return (the race's reward). ``behaviour_clone`` then trains the actor to imitate the teacher
(masked cross-entropy) and the value net to predict the return, so PPO starts from about B1
with a value function that already fits it, instead of from random play, where only the
position tie-break pays and pit stops happen on most laps (pilot A).

The weights stay ``learned_parameter``; the manifest records that the policy started from the
rules (lineage ``trained_from``), so a learned behaviour that only copies the teacher can't be
passed off as discovered.
"""

import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sb3_contrib import MaskablePPO

from f1ar.policy.strategists import RulesTeacher
from f1ar.training.actions import N_INTENT, PER_PIT
from f1ar.training.ppo import TrainConfig, TrainingEnv, make_env


@dataclass(frozen=True)
class Demonstrations:
    obs: np.ndarray
    masks: np.ndarray
    actions: np.ndarray
    returns: np.ndarray
    episode: np.ndarray
    """Episode index of each decision (validation splits by whole races)."""

    @property
    def episodes(self) -> int:
        return int(self.episode.max()) + 1 if len(self.episode) else 0

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            np.savez_compressed(
                f,
                obs=self.obs,
                masks=self.masks,
                actions=self.actions,
                returns=self.returns,
                episode=self.episode,
            )

    @classmethod
    def load(cls, path: Path) -> "Demonstrations":
        with np.load(path) as z:
            return cls(*(np.array(z[k]) for k in ("obs", "masks", "actions", "returns", "episode")))

    def subset(self, keep: np.ndarray) -> "Demonstrations":
        return Demonstrations(
            self.obs[keep],
            self.masks[keep],
            self.actions[keep],
            self.returns[keep],
            self.episode[keep],
        )


@cache
def _env(root: Path, cfg: TrainConfig) -> TrainingEnv:
    return make_env(root, cfg)


def teacher_episode(job: tuple[Path, TrainConfig, int]) -> dict[str, Any]:
    root, cfg, seed = job
    env = _env(root, cfg)
    teacher = RulesTeacher()
    obs, _ = env.reset(seed=seed)
    rows_obs, rows_mask, rows_act = [], [], []
    while True:
        decision = env._decision
        assert decision is not None and env.session is not None
        action = teacher.decide(env.session, decision)
        rows_obs.append(obs)
        rows_mask.append(env.action_masks())
        rows_act.append(action)
        obs, reward, done, _, _ = env.step(action)
        if done:
            return {
                "obs": np.array(rows_obs, dtype=np.float32),
                "masks": np.array(rows_mask, dtype=bool),
                "actions": np.array(rows_act, dtype=np.int64),
                "reward": float(reward),
            }


TEACHER_VERSION = "rules_teacher_v2"
"""v2 (M5 slice 3b): the rule pace (``rule_pace``) instead of BALANCED."""


def demos_key(cfg: TrainConfig, episodes: int) -> str:
    """Everything the demonstrations depend on: a change in any of them means new demos (slice 3
    keyed the cache by size and seed only, which would have reused rl_obs_v1 demos)."""
    from f1ar.simulation.lightweight_engine import ENGINE_VERSION
    from f1ar.training.features import RL_FEATURE_NAMES, RL_OBSERVATION_SCHEMA

    material = json.dumps(
        {
            "episodes": episodes,
            "seed": cfg.seed,
            "events": cfg.events,
            "schema": RL_OBSERVATION_SCHEMA,
            "features": RL_FEATURE_NAMES,
            "engine": ENGINE_VERSION,
            "teacher": TEACHER_VERSION,
            "tyre_randomisation": cfg.tyre_randomisation,
            "rain_probability": cfg.rain_probability,
            "championship_probability": cfg.championship_probability,
            "character_randomisation": cfg.character_randomisation,
        },
        sort_keys=True,
    )
    return hashlib.sha256(material.encode()).hexdigest()[:12]


def collect(root: Path, cfg: TrainConfig, episodes: int, workers: int) -> Demonstrations:
    """The teacher's decisions on ``episodes`` training scenarios; cached on disk under a key of
    everything they depend on (``demos_key``; the scenarios and the teacher are deterministic)."""
    cache_path = (
        root
        / "artifacts"
        / "training"
        / "demos"
        / f"teacher_{episodes}_seed{cfg.seed}_{demos_key(cfg, episodes)}.npz"
    )
    if cache_path.exists():
        return Demonstrations.load(cache_path)
    # seeds far from the PPO workers' (cfg.seed + env index) so no race repeats
    base = 10_000_000 + cfg.seed * 100_000
    jobs = [(root, replace(cfg, eval_events=()), base + i) for i in range(episodes)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        eps = list(pool.map(teacher_episode, jobs, chunksize=4))
    demos = Demonstrations(
        obs=np.concatenate([e["obs"] for e in eps]),
        masks=np.concatenate([e["masks"] for e in eps]),
        actions=np.concatenate([e["actions"] for e in eps]),
        returns=np.concatenate([np.full(len(e["actions"]), e["reward"]) for e in eps]),
        episode=np.concatenate([np.full(len(e["actions"]), i) for i, e in enumerate(eps)]),
    )
    demos.save(cache_path)
    return demos


def block_log_probs(log_probs: torch.Tensor) -> torch.Tensor:
    """Log-probabilities of each (pit, pace) choice, summed over the racecraft intents."""
    return torch.logsumexp(log_probs.reshape(len(log_probs), -1, N_INTENT), dim=2)


def _metrics(model: MaskablePPO, d: Demonstrations) -> dict[str, float]:
    policy = model.policy
    device = policy.device
    obs = torch.as_tensor(d.obs, device=device)
    actions = torch.as_tensor(d.actions, device=device)
    returns = torch.as_tensor(d.returns, dtype=torch.float32, device=device)
    with torch.no_grad():
        dist = policy.get_distribution(obs, action_masks=d.masks)
        pred = dist.distribution.probs.argmax(dim=1)  # type: ignore[union-attr]
        value = policy.predict_values(obs).flatten()
    pit_pred = pred >= PER_PIT  # the stay-out block comes first (actions.encode)
    pit_true = actions >= PER_PIT
    with torch.no_grad():
        log_p = dist.distribution.logits  # type: ignore[union-attr]
        block = block_log_probs(log_p).argmax(dim=1)
        intent_p = log_p.exp().reshape(len(log_p), -1, N_INTENT).sum(dim=1).mean(dim=0)
    hits = int((pit_pred & pit_true).sum())
    same_stop = int((pit_pred & pit_true & (pred // PER_PIT == actions // PER_PIT)).sum())
    return {
        "accuracy": round(float((pred == actions).float().mean()), 4),
        "pit_pace_accuracy": round(float((block == actions // N_INTENT).float().mean()), 4),
        "intent_mean_probability": {  # type: ignore[dict-item]
            name: round(float(intent_p[i]), 4)
            for i, name in enumerate(("defend", "normal", "attack"))
        },
        "stop_recall": round(hits / max(1, int(pit_true.sum())), 4),
        "stop_precision": round(hits / max(1, int(pit_pred.sum())), 4),
        "stop_compound_right": round(same_stop / max(1, hits), 4),
        "value_r2": round(
            1.0
            - float(torch.nn.functional.mse_loss(value, returns)) / max(1e-9, float(returns.var())),
            4,
        ),
        "nll": round(float(-dist.log_prob(actions).mean()), 4),
    }


def behaviour_clone(
    model: MaskablePPO,
    demos: Demonstrations,
    *,
    epochs: int = 60,
    batch_size: int = 2048,
    lr: float = 1e-3,
    stop_weight: float = 5.0,
    validation: float = 0.1,
    seed: int = 0,
) -> dict[str, Any]:
    """Fit the actor to the teacher's actions (stop decisions weighted ``stop_weight``: they are
    ~2 % of decisions, and plain cross-entropy learns "never stop", pilot D) and the value net to
    the returns. Reports metrics on ``validation`` of the races, held out.

    Slice 4 (D-048): only the teacher's **pit and pace** choice is imitated (the likelihood of its
    (pit, pace) block, summed over the racecraft intents). The rules can't attack or defend, so
    their NORMAL isn't a preference to copy: cloning it made attempt 1 so sure of NORMAL that PPO
    never tried anything else (attack share 0.05 %). The intent stays as initialised (about
    uniform, held there by a cross-entropy to the uniform split) and is learned by PPO from
    race results."""
    rng = np.random.default_rng(seed)
    held = rng.random(demos.episodes) < validation
    train_d = demos.subset(~held[demos.episode])
    val_d = demos.subset(held[demos.episode])
    policy = model.policy
    device = policy.device
    obs = torch.as_tensor(train_d.obs, device=device)
    masks = train_d.masks
    actions = torch.as_tensor(train_d.actions, device=device)
    returns = torch.as_tensor(train_d.returns, dtype=torch.float32, device=device)
    weight = torch.where(actions >= PER_PIT, stop_weight, 1.0)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    n = len(actions)
    policy.set_training_mode(True)
    for _ in range(epochs):
        order = torch.randperm(n, generator=gen)
        for start in range(0, n, batch_size):
            idx_cpu = order[start : start + batch_size]
            idx = idx_cpu.to(device)
            dist = policy.get_distribution(obs[idx], action_masks=masks[idx_cpu.numpy()])
            w = weight[idx]
            log_p = dist.distribution.logits  # type: ignore[union-attr]
            target = actions[idx] // N_INTENT
            chosen = block_log_probs(log_p).gather(1, target.unsqueeze(1)).squeeze(1)
            nll = -(chosen * w).sum() / w.sum()
            # no prior on the intent: keep it uniform inside the teacher's (pit, pace) choice
            # (the summed likelihood alone lets one intent take over: rich get richer)
            inside = log_p.reshape(len(log_p), -1, N_INTENT)[torch.arange(len(target)), target]
            nll = nll - torch.log_softmax(inside, dim=1).mean()
            value = policy.predict_values(obs[idx]).flatten()
            loss = nll + 0.5 * torch.nn.functional.mse_loss(value, returns[idx])
            opt.zero_grad()
            loss.backward()  # type: ignore[no-untyped-call]
            opt.step()
    policy.set_training_mode(False)
    return {
        "races": demos.episodes,
        "decisions": len(demos.actions),
        "stop_share": round(float((demos.actions >= PER_PIT).mean()), 4),
        "epochs": epochs,
        "stop_weight": stop_weight,
        "train": _metrics(model, train_d),
        "validation": _metrics(model, val_d),
    }


def soften(model: MaskablePPO, demos: Demonstrations, target_entropy: float) -> float:
    """Divide the action head by a temperature ``T`` so the mean policy entropy on the
    demonstrations is ``target_entropy`` (nats). The ranking of the actions, and so the argmax
    play (= the teacher's), doesn't change; only how much PPO explores around it. Without this
    the cloned policy is so sure of itself (entropy ~0.02) that PPO never tries anything new
    (pilots D2/D3, D-040). Returns ``T``."""
    policy = model.policy
    head = policy.action_net
    assert isinstance(head, torch.nn.Linear)
    obs = torch.as_tensor(demos.obs, device=policy.device)
    masks = torch.as_tensor(demos.masks, device=policy.device)
    with torch.no_grad():
        latent = policy.mlp_extractor.forward_actor(policy.extract_features(obs))  # type: ignore[arg-type]
        logits = head(latent)

        def entropy(t: float) -> float:
            """Of the (pit, pace) choice (slice 4: the intents are near uniform anyway)."""
            z = torch.where(masks, logits / t, torch.tensor(-1e9, device=logits.device))
            p = block_log_probs(torch.log_softmax(z, dim=1)).exp()
            return float(-(p * torch.log(p.clamp_min(1e-12))).sum(dim=1).mean())

        lo, hi = 1.0, 1.0
        while entropy(hi) < target_entropy and hi < 1e4:
            hi *= 2.0
        for _ in range(40):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if entropy(mid) < target_entropy else (lo, mid)
        t = (lo + hi) / 2
        head.weight.div_(t)
        head.bias.div_(t)
    return t
