"""Train the strategy policy with MaskablePPO (M5 slice 3, D-040).

One learning car vs the rule field on the 2024 dry_train races (data/manifests split), CPU
workers for the simulation, the network update on CUDA. Prints one progress line per rollout and
a paired check vs the rule planner on fixed training cases every ``--eval-every`` steps.
``--register`` writes the chosen network to the policy registry (configs/policies/).

Usage:
    uv run python scripts/train_policy.py --name pilot_a --steps 600000
    uv run python scripts/train_policy.py --name main --steps 3000000 --reward-baseline rules \
        --register strategist_m5s3_v1
"""

import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import sb3_contrib
import torch
import yaml

from f1ar.simulation.lightweight_engine import ENGINE_VERSION
from f1ar.training.features import RL_OBSERVATION_SCHEMA
from f1ar.training.policy_registry import (
    NumpyPolicy,
    feature_names_hash,
    file_sha256,
    register_policy,
)
from f1ar.training.ppo import TrainConfig, train
from f1ar.training.regulations import REGULATIONS_2024

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "data" / "manifests" / "dataset_laps_2024_v1.yaml"
CHECK_EVENTS = (
    "2024_bahrain_gp",
    "2024_spanish_gp",
    "2024_hungarian_gp",
    "2024_singapore_gp",
)


def dry_train_events() -> tuple[str, ...]:
    splits = yaml.safe_load(SPLITS.read_text(encoding="utf-8"))["splits"]
    return tuple(splits["dry_train"])


def git_commit() -> str:
    out = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(ROOT), "status", "--porcelain"], capture_output=True, text=True
    ).stdout.strip()
    return out + ("+dirty" if dirty else "")


def profiles_revision() -> str:
    """Hash of the driver profile configs the training races used (lineage)."""
    files = sorted((ROOT / "configs" / "drivers").glob("*.yaml"))
    import hashlib

    h = hashlib.sha256()
    for f in files:
        h.update(f.name.encode() + b"\0" + f.read_bytes())
    return "sha256:" + h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", required=True, help="run name (artifacts/training/<name>)")
    parser.add_argument("--steps", type=int, default=600_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=14)
    parser.add_argument("--envs-per-worker", type=int, default=4)
    parser.add_argument("--reward-baseline", choices=("none", "rules"), default="none")
    parser.add_argument("--normalize-reward", action="store_true")
    parser.add_argument("--bc-entropy", type=float, default=0.5, metavar="NATS")
    parser.add_argument("--stay-prior", type=float, default=0.0, metavar="P")
    parser.add_argument(
        "--warm-start", type=int, default=0, metavar="RACES", help="imitate the rules first"
    )
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--eval-every", type=int, default=100_000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--register", metavar="POLICY_ID", help="register best.npz under this id")
    parser.add_argument(
        "--register-only", action="store_true", help="don't train; register an existing run"
    )
    args = parser.parse_args()
    run_dir = ROOT / "artifacts" / "training" / args.name
    cfg = TrainConfig(
        events=dry_train_events(),
        total_steps=args.steps,
        seed=args.seed,
        n_envs=args.workers,
        envs_per_worker=args.envs_per_worker,
        reward_baseline=args.reward_baseline,
        normalize_reward=args.normalize_reward,
        warm_start_episodes=args.warm_start,
        stay_prior=args.stay_prior,
        bc_target_entropy=args.bc_entropy,
        ent_coef=args.ent_coef,
        learning_rate=args.lr,
        eval_every=args.eval_every,
        eval_events=CHECK_EVENTS,
        device=args.device,
    )
    if not args.register_only:
        print(
            f"training {args.name}: {cfg.total_steps:,} decisions, {cfg.n_envs} workers, "
            f"{len(cfg.events)} events, device {cfg.device} "
            f"({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no CUDA'})",
            flush=True,
        )
        started = datetime.now(UTC)
        train(ROOT, cfg, run_dir)
        meta = {"started": started.isoformat(timespec="seconds"), "git_commit": git_commit()}
        meta["finished"] = datetime.now(UTC).isoformat(timespec="seconds")
        (run_dir / "run.json").write_text(json.dumps(meta) + "\n", encoding="utf-8", newline="\n")
    if args.register:
        meta = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        best = json.loads((run_dir / "best.json").read_text(encoding="utf-8"))
        saved = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        policy = NumpyPolicy.load(run_dir / "best.npz")
        hidden = [int(w.shape[0]) for w in policy.weights[:-1]]
        manifest = register_policy(
            ROOT,
            policy,
            {
                "policy_id": args.register,
                "algorithm": f"MaskablePPO (sb3-contrib {sb3_contrib.__version__}, "
                f"torch {torch.__version__})",
                "architecture": f"MLP {policy.n_inputs}-{'-'.join(map(str, hidden))}-"
                f"{policy.n_actions}, tanh; actor only (value net not exported)",
                "observation_schema": RL_OBSERVATION_SCHEMA,
                "feature_names_sha256": feature_names_hash(),
                "regulations": REGULATIONS_2024.version,
                "engine_version": ENGINE_VERSION,
                "training": {
                    "run": args.name,
                    "config": saved,
                    "selected": f"best paired check vs rules on training cases at step "
                    f"{best['steps']} (check: {best})",
                    "git_commit": meta["git_commit"],
                    "started": meta["started"],
                    "finished": meta["finished"],
                    "note": meta.get("note", ""),
                    "torch_checkpoint": f"artifacts/training/{args.name}/best.zip (not tracked)",
                    "torch_checkpoint_sha256": file_sha256(run_dir / "best.zip"),
                    "setup": "one learning car vs the rule field, dry, points objective",
                },
                "lineage": {
                    "parent_policy": None,
                    "season": 2024,
                    "profiles_revision": profiles_revision(),
                    "trained_from": (
                        f"behaviour cloning of the rules teacher ({saved['warm_start_episodes']} "
                        "races), then PPO"
                        if saved.get("warm_start_episodes")
                        else f"scratch (random init, stay-out prior {saved.get('stay_prior')})"
                    ),
                },
                "evaluation": {},
                "notes": (
                    "Weights are learned_parameter; personas, risk utilities, the team "
                    "principal's rules and the PUSH/CONSERVE effects they were trained against "
                    "are editorial_assumption.",
                ),
            },
        )
        print(f"registered {manifest.policy_id}: {manifest.artifact} {manifest.artifact_sha256}")


if __name__ == "__main__":
    main()
