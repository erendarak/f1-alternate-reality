"""Fit the rule teams' rival reactions to real undercut behaviour (A1b, D-062):
configs/calibration/rival_rules_2024_v1.json and its registry manifest.

The undercut detector of EXP-020 (``calibration/undercut``) counts attempts per race in the real
2024 dry races and in simulated ones of the same tracks. The simulated field is the A1b world
(hidden historical truth, per-team practice + private data, in-race learning, hidden trend).
The two windows (``attack_window_laps``, ``cover_window_laps``) are chosen on the dry_train races
so that attempts per race match the real ones; the gaps and the stuck-lap count stay editorial.
The dry_test races check it.

Usage:
    uv run python scripts/fit_rival_rules.py --seeds 3
"""

import argparse
import functools
import hashlib
import itertools
import json
import statistics
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

print = functools.partial(print, flush=True)  # noqa: A001 (progress through pipes)
ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
OUT = Path("calibration") / "rival_rules_2024_v1.json"
SPLITS = ROOT / "data" / "manifests" / "dataset_laps_2024_v1.yaml"
DATASET = ROOT / "data" / "processed" / "datasets" / "laps_2024_v1.parquet"
WINDOWS = (0, 1, 2, 3, 4, 6)


def event_seed(event: str, k: int) -> int:
    return int(hashlib.sha256(event.encode()).hexdigest()[:6], 16) * 100 + k


@functools.cache
def _world(event: str, seed: int) -> Any:
    from f1ar.simulation.race_day import with_hidden_truth
    from f1ar.simulation.scenarios import event_data_dir, load_event, setup_from_codes

    roster, profiles = load_event(CONFIGS, event)
    base = setup_from_codes(
        roster, profiles, None, data_dir=event_data_dir(ROOT, event), configs_dir=CONFIGS
    )
    return with_hidden_truth(
        base, CONFIGS, truth_seed=seed, learning=True, track_trend=True, per_team=True
    )


def job(args: tuple[str, int, int | None, int | None]) -> dict[str, Any]:
    from f1ar.calibration.undercut import simulated_laps, undercut_attempts
    from f1ar.domain.provenance import SourceType
    from f1ar.domain.race_state import RivalRules

    event, seed, attack, cover = args
    setup = _world(event, seed)
    if attack is not None and cover is not None:
        rules = RivalRules(
            source_type=SourceType.EDITORIAL_ASSUMPTION,
            note="fit",
            attack_window_laps=attack,
            cover_window_laps=cover,
        )
        setup = setup.model_copy(update={"rival_rules": rules})
    attempts = undercut_attempts(simulated_laps(setup, seed))
    return {
        "event": event,
        "attack": attack,
        "cover": cover,
        "attempts": len(attempts),
        "gains": [a.gain_s for a in attempts],
        "passed": [a.passed for a in attempts],
    }


def stats(rows: list[dict[str, Any]]) -> dict[str, float]:
    gains = [g for r in rows for g in r["gains"]]
    passed = [p for r in rows for p in r["passed"]]
    return {
        "attempts_per_race": round(statistics.fmean(r["attempts"] for r in rows), 2),
        "gain_median_s": round(statistics.median(gains), 2) if gains else float("nan"),
        "passed_share": round(statistics.fmean(passed), 3) if passed else float("nan"),
    }


def main() -> None:
    from f1ar.calibration.undercut import real_laps, summary, undercut_attempts
    from f1ar.simulation.race_day import load_prior

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--workers", type=int, default=14)
    args = parser.parse_args()
    splits = yaml.safe_load(SPLITS.read_text(encoding="utf-8"))["splits"]
    prior = load_prior(CONFIGS)
    assert prior is not None
    train = [e for e in splits["dry_train"] if e in prior["events"]]
    test = [e for e in splits["dry_test"] if e in prior["events"]]
    laps = pd.read_parquet(DATASET)
    real = {e: undercut_attempts(real_laps(laps[laps["event_id"] == e])) for e in train + test}
    real_train = statistics.fmean(len(real[e]) for e in train)
    real_test = statistics.fmean(len(real[e]) for e in test)
    print(f"real attempts per race: train {real_train:.2f}, test {real_test:.2f}")
    grid: list[tuple[int | None, int | None]] = [(None, None)]
    grid += list(itertools.product(WINDOWS, WINDOWS))
    jobs = [
        (e, event_seed(e, k), a, c)
        for a, c in grid
        for e in train
        for k in range(1, args.seeds + 1)
    ]
    print(f"{len(jobs)} simulated races ({len(grid)} settings x {len(train)} races)")
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(job, jobs, chunksize=4))
    print(f"  done in {(time.perf_counter() - t0) / 60:.1f} min")
    table: list[dict[str, Any]] = []
    for a, c in grid:
        s = stats([r for r in rows if r["attack"] == a and r["cover"] == c])
        table.append({"attack_window_laps": a, "cover_window_laps": c, **s})
    fitted = [t for t in table if t["attack_window_laps"] is not None]
    best = min(fitted, key=lambda t: abs(t["attempts_per_race"] - real_train))
    print("no rules:", table[0])
    print("chosen:", best)
    check = [
        (e, event_seed(e, k), w, v)
        for e in test
        for k in range(1, args.seeds + 1)
        for w, v in ((None, None), (best["attack_window_laps"], best["cover_window_laps"]))
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        held = list(pool.map(job, check, chunksize=2))
    test_rules = stats([r for r in held if r["attack"] is not None])
    test_none = stats([r for r in held if r["attack"] is None])
    print(f"test: real {real_test:.2f}, rules {test_rules}, no rules {test_none}")
    out = {
        "label": "rule teams' rival reactions: undercut and cover (A1b, D-062)",
        "source_type": "estimated_from_data",
        "fitted": {
            "attack_window_laps": best["attack_window_laps"],
            "cover_window_laps": best["cover_window_laps"],
        },
        "editorial": {"attack_gap_s": 1.0, "attack_laps": 2, "cover_gap_s": 3.0},
        "method": (
            "windows chosen on 2024 dry_train races so that simulated undercut attempts per race "
            "(EXP-020 detector) match the real ones; A1b world (historical truth, per-team "
            "practice + private data, in-race learning, hidden trend), seeds per race"
        ),
        "real": {
            "train_attempts_per_race": round(real_train, 2),
            "test_attempts_per_race": round(real_test, 2),
            "all": summary([a for v in real.values() for a in v]),
        },
        "train": {"grid": table, "chosen": best},
        "test": {"rules": test_rules, "no_rules": test_none},
        "fitted_on": train,
        "seeds": args.seeds,
    }
    text = json.dumps(out, indent=1) + "\n"
    (CONFIGS / OUT).write_text(text, encoding="utf-8", newline="\n")
    manifest = {
        "model_name": "rival_rules",
        "model_version": "v1",
        "kind": "undercut / cover windows of the rule teams, fitted to real attempt rates",
        "source_type": "estimated_from_data",
        "artifact": (Path("configs") / OUT).as_posix(),
        "artifact_sha256": "sha256:" + hashlib.sha256(text.encode()).hexdigest(),
        "feature_schema_version": 1,
        "training_dataset_manifest": "data/manifests/dataset_laps_2024_v1.yaml",
        "training_dataset_content_sha256": yaml.safe_load(SPLITS.read_text(encoding="utf-8"))[
            "file"
        ]["content_sha256"],
        "training_races": train,
        "metrics": {
            "real_attempts_per_race": round(real_train, 2),
            "sim_attempts_per_race": best["attempts_per_race"],
            "held_out_check": "EXP-031",
        },
        "status": "fitted on 2024 (A1b, D-062); applied to every season (labelled)",
    }
    (CONFIGS / "models" / "rival_rules_2024_v1.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8", newline="\n"
    )
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
