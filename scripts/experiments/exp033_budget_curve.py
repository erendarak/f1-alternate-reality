"""EXP-033: the training-budget curve (A2/A3, D-064).

Every checkpoint of a from-scratch run (``artifacts/training/<run>/checkpoints/budget_*.npz``,
budget 0 = the untrained network) drives one car on **held-out** races (2018-2025 dry_test, never
trained on) in the A1b world, with a hidden truth drawn per race and seed; the same car under the
expert rule team in the same race (identical luck) is the reference. Several cars per race (grid
slots ``--slots``); the AI samples from its policy as in training (seeded per race), pit x pace,
racecraft NORMAL, with the regulation guard of training.

Per budget: score vs the expert (M5 POINTS objective; CI by bootstrap over races), points per
car-race and per 22-race season, finishing position, stops, disqualifications.

Usage:
    uv run python scripts/experiments/exp033_budget_curve.py --run a2_pilot --seeds 2 \\
        --json docs/experiments/exp033_pilot.json
"""

import argparse
import functools
import hashlib
import json
import random
import statistics
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

print = functools.partial(print, flush=True)  # noqa: A001 (progress through pipes)
ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "configs"
SEASON_RACES = 22


def event_seed(event: str, k: int) -> int:
    return int(hashlib.sha256(event.encode()).hexdigest()[:6], 16) * 100 + k


@functools.cache
def _base(event: str) -> Any:
    from f1ar.simulation.scenarios import event_data_dir, load_event, setup_from_codes

    roster, profiles = load_event(CONFIGS, event)
    return setup_from_codes(
        roster, profiles, None, data_dir=event_data_dir(ROOT, event), configs_dir=CONFIGS
    )


def _world(event: str, seed: int) -> Any:
    from f1ar.simulation.race_day import TruthMode, with_hidden_truth, with_rival_rules

    setup = with_hidden_truth(
        _base(event),
        CONFIGS,
        mode=TruthMode.DRAWN,
        truth_seed=seed,
        learning=True,
        track_trend=True,
        per_team=True,
    )
    return with_rival_rules(setup, CONFIGS)


def job(args: tuple[str, int, tuple[int, ...], tuple[str, ...]]) -> list[dict[str, Any]]:
    from f1ar.policy.strategists import LearnedStrategist
    from f1ar.training.evaluate import evaluate
    from f1ar.training.objectives import scoring_for
    from f1ar.training.policy_registry import NumpyPolicy

    event, seed, slots, checkpoints = args
    setup = _world(event, seed)
    by_grid = {d.grid_position: d.driver_id for d in setup.drivers}
    rows = []
    for slot in slots:
        driver = by_grid.get(slot)
        if driver is None:
            continue
        scoring = scoring_for(setup, [driver], None)[driver]
        expert = evaluate(setup, driver, seed, scoring, None, racecraft=False)
        # the policy samples its actions as in training (seeded: same race, same draws)
        for ckpt in checkpoints:
            policy = NumpyPolicy.load(Path(ckpt))
            strategist = LearnedStrategist(policy, Path(ckpt).stem, sample_seed=seed)
            ai = evaluate(
                setup, driver, seed, scoring, strategist, racecraft=False, regulation_guard=True
            )
            rows.append(
                {
                    "event": event,
                    "seed": seed,
                    "driver_id": driver,
                    "checkpoint": Path(ckpt).stem,
                    "score_vs_expert": ai.reward - expert.reward,
                    "points": ai.points,
                    "expert_points": expert.points,
                    "position": ai.position,
                    "expert_position": expert.position,
                    "stops": ai.stops,
                    "expert_stops": expert.stops,
                    "dsq": ai.disqualified,
                }
            )
    return rows


def ci(rows: list[dict[str, Any]], key: str) -> tuple[float, float, float]:
    per: dict[str, list[float]] = {}
    for r in rows:
        per.setdefault(r["event"], []).append(float(r[key]))
    means = {e: statistics.fmean(v) for e, v in per.items()}
    keys = sorted(means)
    rng = random.Random(33)
    boot = sorted(statistics.fmean(means[rng.choice(keys)] for _ in keys) for _ in range(2000))
    return statistics.fmean(means.values()), boot[50], boot[1949]


def main() -> None:
    from f1ar.simulation.race_day import load_prior

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--slots", type=int, nargs="+", default=[1, 4, 8, 12, 16, 20])
    parser.add_argument("--workers", type=int, default=14)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    ckpt_dir = ROOT / "artifacts" / "training" / args.run / "checkpoints"
    ckpts = sorted(ckpt_dir.glob("budget_*.npz"), key=lambda p: int(p.stem.split("_")[1]))
    if not ckpts:
        raise SystemExit(f"no budget checkpoints in {ckpt_dir}")
    prior = load_prior(CONFIGS)
    assert prior is not None
    events = sorted(
        e
        for e, v in prior["events"].items()
        if v["split"] == "dry_test"
        and v["season"] >= 2018
        and (ROOT / "data" / "processed" / e).exists()
    )
    names = tuple(str(p) for p in ckpts)
    jobs = [
        (e, event_seed(e, k), tuple(args.slots), names)
        for e in events
        for k in range(1, args.seeds + 1)
    ]
    print(f"EXP-033 {args.run}: {len(ckpts)} checkpoints x {len(events)} held-out races")
    print(f"  x {args.seeds} seeds x {len(args.slots)} cars = {len(jobs)} jobs")
    t0 = time.perf_counter()
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(job, j) for j in jobs]
        for i, f in enumerate(as_completed(futures), 1):
            rows.extend(f.result())
            if i % 10 == 0 or i == len(jobs):
                el = time.perf_counter() - t0
                eta = el / i * (len(jobs) - i) / 60
                print(f"  {i}/{len(jobs)} jobs, {el / 60:.1f} min, ETA {eta:.1f} min")
    out: dict[str, Any] = {
        "run": args.run,
        "seeds": args.seeds,
        "slots": args.slots,
        "events": events,
        "budgets": {},
    }
    for p in ckpts:
        mine = [r for r in rows if r["checkpoint"] == p.stem]
        meta = json.loads(p.with_suffix(".json").read_text(encoding="utf-8"))
        mean, lo, hi = ci(mine, "score_vs_expert")
        pts = statistics.fmean(r["points"] for r in mine)
        exp_pts = statistics.fmean(r["expert_points"] for r in mine)
        out["budgets"][str(meta["budget"])] = {
            "steps": meta["steps"],
            "cases": len(mine),
            "score_vs_expert": round(mean, 4),
            "ci95": [round(lo, 4), round(hi, 4)],
            "points_per_race": round(pts, 3),
            "expert_points_per_race": round(exp_pts, 3),
            "points_per_season_vs_expert": round((pts - exp_pts) * SEASON_RACES, 1),
            "position": round(statistics.fmean(r["position"] for r in mine), 2),
            "expert_position": round(statistics.fmean(r["expert_position"] for r in mine), 2),
            "stops": round(statistics.fmean(r["stops"] for r in mine), 2),
            "expert_stops": round(statistics.fmean(r["expert_stops"] for r in mine), 2),
            "dsq_share": round(statistics.fmean(r["dsq"] for r in mine), 3),
            "better_than_expert": sum(r["score_vs_expert"] > 1e-9 for r in mine),
            "worse_than_expert": sum(r["score_vs_expert"] < -1e-9 for r in mine),
        }
        print(meta["budget"], json.dumps(out["budgets"][str(meta["budget"])]))
    print(f"EXP-033 done in {(time.perf_counter() - t0) / 60:.1f} min")
    if args.json:
        args.json.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
