"""EXP-032: A2 pre-gate: does what a car sees predict the best first-stop timing? (D-063)

1. Collect (``training/pit_timing``): in the A1b world (hidden historical truth, per-team practice
   + private data, in-race learning, hidden trend, rival rules), every rule car forked 3 laps
   before its planned first stop; the stop forced at -2..+3 laps on identical luck; the features
   its team could see at the fork. 2018-2025 dry_train races (training) and dry_test (held out).
2. Fit, on training forks only, models of each offset's gain over the plan, and pick per fork the
   offset with the best predicted gain (or the plan): a fixed offset, ridge regression, gradient
   boosted trees.
3. Held out: the realised gain of each model's picks, with a bootstrap CI over races; next to
   hindsight (the best offset known afterwards: the ceiling).

Gate: a learnable gain above ~0.005 score per car-race with a CI above 0 -> train A2 on it.

Usage:
    uv run python scripts/experiments/exp032_timing_pregate.py --seeds 3 \\
        --json docs/experiments/exp032.json        # collect + fit (~1 h)
    uv run python scripts/experiments/exp032_timing_pregate.py --from-rows \\
        --json docs/experiments/exp032.json        # refit from artifacts/exp032_rows.json
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

import numpy as np

print = functools.partial(print, flush=True)  # noqa: A001 (progress through pipes)
ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "configs"
ROWS = ROOT / "artifacts" / "exp032_rows.json"
OFFSETS = (-2, -1, 0, 1, 2, 3)
MOVES = tuple(d for d in OFFSETS if d != 0)


def event_seed(event: str, k: int) -> int:
    return int(hashlib.sha256(event.encode()).hexdigest()[:6], 16) * 100 + k


def job(args: tuple[str, int, str]) -> list[dict[str, Any]]:
    from f1ar.simulation import lightweight_engine as engine
    from f1ar.simulation.race_day import with_hidden_truth, with_rival_rules
    from f1ar.simulation.scenarios import event_data_dir, load_event, setup_from_codes
    from f1ar.training.pit_timing import timing_rows

    event, seed, split = args
    roster, profiles = load_event(CONFIGS, event)
    base = setup_from_codes(
        roster, profiles, None, data_dir=event_data_dir(ROOT, event), configs_dir=CONFIGS
    )
    setup = with_hidden_truth(
        base, CONFIGS, truth_seed=seed, learning=True, track_trend=True, per_team=True
    )
    setup = with_rival_rules(setup, CONFIGS)
    world = engine.create_world(setup, seed=seed, world_id="exp032")
    return [{**r, "event": event, "seed": seed, "split": split} for r in timing_rows(world)]


def collect(seeds: int, workers: int) -> list[dict[str, Any]]:
    from f1ar.simulation.race_day import load_prior

    prior = load_prior(CONFIGS)
    assert prior is not None
    events = {
        e: v["split"]
        for e, v in prior["events"].items()
        if v["season"] >= 2018 and (ROOT / "data" / "processed" / e).exists()
    }
    jobs = [
        (e, event_seed(e, k), s) for e, s in sorted(events.items()) for k in range(1, seeds + 1)
    ]
    print(f"collecting: {len(events)} races x {seeds} seeds = {len(jobs)} jobs")
    t0 = time.perf_counter()
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(job, j) for j in jobs]
        for i, f in enumerate(as_completed(futures), 1):
            rows.extend(f.result())
            if i % 10 == 0 or i == len(jobs):
                el = time.perf_counter() - t0
                eta = el / i * (len(jobs) - i) / 60
                done = f"{i}/{len(jobs)} jobs, {len(rows)} forks"
                print(f"  {done}, {el / 60:.1f} min, ETA {eta:.1f} min")
    ROWS.write_text(json.dumps(rows) + "\n", encoding="utf-8", newline="\n")
    return rows


def matrices(rows: list[dict[str, Any]], names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    x = np.array([[r["features"][n] for n in names] for r in rows], dtype=float)
    y = np.array([[r["scores"][str(d)] - r["scores"]["0"] for d in MOVES] for r in rows])
    return x, y


def realised(rows: list[dict[str, Any]], picks: list[int]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(
            r["scores"][str(d)] - r["scores"]["0"] for r, d in zip(rows, picks, strict=True)
        ),
    }


def by_race_ci(rows: list[dict[str, Any]], gains: list[float]) -> tuple[float, float, float]:
    per: dict[str, list[float]] = {}
    for r, g in zip(rows, gains, strict=True):
        per.setdefault(r["event"], []).append(g)
    means = {e: statistics.fmean(v) for e, v in per.items()}
    keys = sorted(means)
    rng = random.Random(32)
    boot = sorted(statistics.fmean(means[rng.choice(keys)] for _ in keys) for _ in range(2000))
    return statistics.fmean(means.values()), boot[50], boot[1949]


def evaluate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    for r in rows:  # json keys
        r["scores"] = {str(k): v for k, v in r["scores"].items()}
    train = [r for r in rows if r["split"] == "dry_train"]
    test = [r for r in rows if r["split"] == "dry_test"]
    names = sorted(train[0]["features"])
    xt, yt = matrices(train, names)
    xh, yh = matrices(test, names)

    def picks_from(pred: np.ndarray) -> list[int]:
        best = pred.argmax(axis=1)
        return [MOVES[b] if pred[i, b] > 0 else 0 for i, b in enumerate(best)]

    out: dict[str, Any] = {
        "train_forks": len(train),
        "test_forks": len(test),
        "train_races": len({r["event"] for r in train}),
        "test_races": len({r["event"] for r in test}),
        "features": names,
    }
    hind = [max(r["scores"].values()) - r["scores"]["0"] for r in test]
    out["hindsight"] = dict(zip(("mean", "lo", "hi"), by_race_ci(test, hind), strict=True))
    fixed = MOVES[int(np.argmax(yt.mean(axis=0)))]
    fixed_gain = float(yt.mean(axis=0).max())
    fixed_pick = fixed if fixed_gain > 0 else 0
    models: dict[str, list[int]] = {
        "plan (d=0)": [0] * len(test),
        "fixed offset": [fixed_pick] * len(test),
    }
    ridge = make_pipeline(StandardScaler(), Ridge(alpha=10.0)).fit(xt, yt)
    models["ridge"] = picks_from(ridge.predict(xh))
    preds = np.column_stack(
        [
            HistGradientBoostingRegressor(
                max_depth=3, learning_rate=0.05, max_iter=300, min_samples_leaf=50, random_state=0
            )
            .fit(xt, yt[:, j])
            .predict(xh)
            for j in range(len(MOVES))
        ]
    )
    models["boosted trees"] = picks_from(preds)
    out["fixed_offset"] = fixed_pick
    out["models"] = {}
    for name, picks in models.items():
        gains = [r["scores"][str(d)] - r["scores"]["0"] for r, d in zip(test, picks, strict=True)]
        mean, lo, hi = by_race_ci(test, gains)
        out["models"][name] = {
            "gain": round(mean, 4),
            "ci95": [round(lo, 4), round(hi, 4)],
            "moved_share": round(sum(p != 0 for p in picks) / len(picks), 3),
            "share_of_hindsight": round(mean / out["hindsight"]["mean"], 3)
            if out["hindsight"]["mean"]
            else 0.0,
        }
    # in-sample fit of the trees (how much is memorised vs generalises)
    tree_train = np.column_stack(
        [
            HistGradientBoostingRegressor(
                max_depth=3, learning_rate=0.05, max_iter=300, min_samples_leaf=50, random_state=0
            )
            .fit(xt, yt[:, j])
            .predict(xt)
            for j in range(len(MOVES))
        ]
    )
    tp = picks_from(tree_train)
    out["boosted_trees_in_sample_gain"] = round(
        statistics.fmean(
            r["scores"][str(d)] - r["scores"]["0"] for r, d in zip(train, tp, strict=True)
        ),
        4,
    )
    out["hindsight"] = {k: round(v, 4) for k, v in out["hindsight"].items()}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--workers", type=int, default=14)
    parser.add_argument("--from-rows", action="store_true")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    t0 = time.perf_counter()
    rows = (
        json.loads(ROWS.read_text(encoding="utf-8"))
        if args.from_rows
        else collect(args.seeds, args.workers)
    )
    out = {"seeds": args.seeds, **evaluate(rows)}
    print(json.dumps({k: v for k, v in out.items() if k != "features"}, indent=1))
    print(f"EXP-032 done in {(time.perf_counter() - t0) / 60:.1f} min")
    if args.json:
        args.json.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
