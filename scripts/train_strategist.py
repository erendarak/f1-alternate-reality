"""Train the strategist from scratch in the realistic world (A2, D-064).

No rule planner, no cloning of the rules, no planner advice: a random MaskablePPO network that
sees ``rl_obs_v4`` (what the pit wall sees) and chooses pit x pace (racecraft NORMAL) for one car
among the expert rule teams of the A1b world (hidden truth drawn per episode, per-team practice
+ private data, in-race learning, hidden trend, rival rules), on the 2018-2025 dry_train races.
The reward is the race's POINTS score minus the expert's score for the same car in the same race
(identical luck). An editorial "stay out" prior sets how often the untrained network tries a
stop (exploration only, nothing about when or for what), and a regulation guard forces a
compliant stop on the last chance when the two-dry-compound rule isn't met yet (the rulebook).

Checkpoints at fixed budgets (``--budgets``) are the points of the A3 training-budget curve.

Usage:
    uv run python scripts/train_strategist.py --name a2_pilot --steps 100000     # pilot
    uv run python scripts/train_strategist.py --name a2_main --steps 1600000 \\
        --budgets 25000 50000 100000 200000 400000 800000 1600000
"""

import argparse
import json
from pathlib import Path

import torch

from f1ar.training.ppo import TrainConfig, train

ROOT = Path(__file__).resolve().parents[1]


def training_events(first: int, last: int) -> tuple[str, ...]:
    """dry_train races with a race-day prior and data (dry_test races are A3's held-out set)."""
    from f1ar.simulation.race_day import load_prior

    prior = load_prior(ROOT / "configs")
    assert prior is not None
    return tuple(
        sorted(
            e
            for e, v in prior["events"].items()
            if v["split"] == "dry_train"
            and first <= v["season"] <= last
            and (ROOT / "data" / "processed" / e).exists()
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", required=True, help="run name (artifacts/training/<name>)")
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=14)
    parser.add_argument("--envs-per-worker", type=int, default=4)
    parser.add_argument("--stay-prior", type=float, default=0.95, metavar="P")
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--from-season", type=int, default=2018)
    parser.add_argument("--to-season", type=int, default=2025)
    parser.add_argument("--budgets", type=int, nargs="*", default=[])
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    events = training_events(args.from_season, args.to_season)
    cfg = TrainConfig(
        events=events,
        total_steps=args.steps,
        seed=args.seed,
        n_envs=args.workers,
        envs_per_worker=args.envs_per_worker,
        reward_baseline="rules",
        stay_prior=args.stay_prior,
        ent_coef=args.ent_coef,
        learning_rate=args.lr,
        eval_events=(),  # the periodic checks run in the old world: the A3 evaluation replaces them
        world="a1b",
        observation_schema="rl_obs_v4",
        racecraft=False,
        regulation_guard=True,
        checkpoint_steps=tuple(args.budgets),
        device=args.device,
    )
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no CUDA"
    print(
        f"A2 {args.name}: from scratch, {cfg.total_steps:,} decisions, {len(events)} training "
        f"races {args.from_season}-{args.to_season}, {cfg.n_envs} workers, {gpu}",
        flush=True,
    )
    run_dir = ROOT / "artifacts" / "training" / args.name
    _, log = train(ROOT, cfg, run_dir)
    last = log.rows[-1] if log.rows else {}
    print(json.dumps(last), flush=True)
    print(f"A2 {args.name} done", flush=True)


if __name__ == "__main__":
    main()
