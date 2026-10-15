"""Fit the W2 driver race pace (vs teammate) and lap noise (engine v0.12.0, D-043).

Writes configs/calibration/driver_pace_2024_v1.json and its registry manifest. Fitted on the
dry_train races only: EXP-012 checks teammate gaps and lap noise on the dry_test races too.

Then the overtaking threshold shift: realistic lap noise gives more passing chances, so one
global shift of every track's pass threshold is fitted by bisection so the simulated rule field
(``--seeds`` per-race seeds on the dry_train races) passes as often as the real one.

Usage:
    uv run python scripts/fit_driver_pace.py
"""

import argparse
import hashlib
import json
import statistics
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
import yaml

from f1ar.calibration.driver_pace_fit import PACE_FILE, fit
from f1ar.calibration.reference_metrics import real_metrics

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "processed" / "datasets" / "laps_2024_v1.parquet"
SPLITS = ROOT / "data" / "manifests" / "dataset_laps_2024_v1.yaml"


def event_seed(event: str, k: int) -> int:
    return int(hashlib.sha256(event.encode()).hexdigest()[:6], 16) * 100 + k


def passes(job: tuple[str, int]) -> int:
    from f1ar.calibration.simulated_metrics import sim_metrics
    from f1ar.simulation.scenarios import event_data_dir, load_event, setup_from_codes

    event, seed = job
    roster, profiles = load_event(ROOT / "configs", event)
    setup = setup_from_codes(
        roster, profiles, None, data_dir=event_data_dir(ROOT, event), configs_dir=ROOT / "configs"
    )
    return sim_metrics(setup, seed).overtakes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--workers", type=int, default=14)
    parser.add_argument("--iterations", type=int, default=7)
    args = parser.parse_args()
    manifest = yaml.safe_load(SPLITS.read_text(encoding="utf-8"))
    train = list(manifest["splits"]["dry_train"])
    cfg = fit(pd.read_parquet(DATASET), train)
    out = ROOT / "configs" / PACE_FILE

    def write() -> None:
        out.write_text(json.dumps(cfg, indent=1) + "\n", encoding="utf-8", newline="\n")

    real = statistics.fmean(real_metrics(ROOT / "data" / "processed" / e).overtakes for e in train)
    jobs = [(e, event_seed(e, k)) for e in train for k in range(1, args.seeds + 1)]
    lo, hi, history = 0.0, 2.0, []
    for _ in range(args.iterations):
        mid = (lo + hi) / 2
        cfg["overtake_threshold_shift_s"] = round(mid, 4)
        write()
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            sim = statistics.fmean(pool.map(passes, jobs, chunksize=2))
        history.append({"shift_s": round(mid, 4), "sim_passes": round(sim, 2)})
        print(f"shift {mid:.3f} s: {sim:.1f} passes per race (real {real:.1f})", flush=True)
        lo, hi = (mid, hi) if sim > real else (lo, mid)
    cfg["overtake_threshold_shift_s"] = round((lo + hi) / 2, 4)
    cfg["overtake_shift_fit"] = {
        "real_passes_per_race": round(real, 2),
        "seeds_per_race": args.seeds,
        "history": history,
    }
    write()
    body = {
        "model_name": "driver_pace",
        "model_version": "v1",
        "kind": "same-lap regression per race; driver vs teammate and residual SD, shrunk",
        "source_type": "estimated_from_data",
        "artifact": out.relative_to(ROOT).as_posix(),
        "artifact_sha256": "sha256:" + hashlib.sha256(out.read_bytes()).hexdigest(),
        "feature_schema_version": 1,
        "training_dataset_manifest": SPLITS.relative_to(ROOT).as_posix(),
        "training_dataset_content_sha256": manifest["file"]["content_sha256"],
        "training_races": sorted(train),
        "metrics": {
            "field_noise_s": cfg["field_noise_s"],
            "overtake_threshold_shift_s": cfg["overtake_threshold_shift_s"],
            "held_out_check": "EXP-012",
        },
        "status": "validated_for_engine_use (W2, engine v0.12.0, D-043)",
    }
    (ROOT / "configs" / "models" / "driver_pace_2024_v1.yaml").write_text(
        yaml.safe_dump(body, sort_keys=False, width=100), encoding="utf-8", newline="\n"
    )
    for code, d in sorted(cfg["drivers"].items(), key=lambda kv: kv[1]["pace_vs_teammate_s"]):
        pace, noise = d["pace_vs_teammate_s"], d["lap_noise_s"]
        print(f"{code} {d['team_id']:<20} pace {pace:+.3f} s  noise {noise:.3f} s")
    print(f"wrote {out.relative_to(ROOT)}; field noise {cfg['field_noise_s']} s")


if __name__ == "__main__":
    main()
