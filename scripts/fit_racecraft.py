"""Fit the W3 racecraft (engine v0.13.0, D-044).

Writes configs/calibration/racecraft_2024_v1.json and its registry manifest:

1. Overtaking refit on the real lap-level encounters of the 21 dry races with the W2 lap noise,
   plus attack/defend terms per driver (``calibration/racecraft_fit``).
2. Launch skill per driver (lap-1 gains beyond grid slot and car pace) and the lap-1 risk profile
   by grid slot (editorial shape).
3. By simulation on the dry_train races (per-race seeds): the launch spread so lap-1 passes
   match the real ones, then one race-level shift of every pass threshold so all passes per race
   match (the encounter fit alone gave 12 % too many), then the spread again.

Usage:
    uv run python scripts/fit_racecraft.py --seeds 4     # ~5-10 min
"""

import argparse
import functools
import hashlib
import json
import statistics
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from f1ar.calibration.driver_pace_fit import load_pace
from f1ar.calibration.lap_model_fit import load_shape
from f1ar.calibration.racecraft_fit import (
    RACECRAFT_FILE,
    fit_overtaking_skills,
    lap1_grid_profile,
    launch_skills,
)
from f1ar.calibration.reference_metrics import real_metrics
from f1ar.calibration.traffic_fit import encounters
from f1ar.simulation.scenarios import default_traffic_params

print = functools.partial(print, flush=True)  # noqa: A001 (progress through pipes)
ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
DATASET = ROOT / "data" / "processed" / "datasets" / "laps_2024_v1.parquet"
SPLITS = ROOT / "data" / "manifests" / "dataset_laps_2024_v1.yaml"


def event_seed(event: str, k: int) -> int:
    return int(hashlib.sha256(event.encode()).hexdigest()[:6], 16) * 100 + k


def passes(job: tuple[str, int, float, float, bool]) -> float:
    from f1ar.calibration.simulated_metrics import sim_metrics
    from f1ar.simulation.scenarios import event_data_dir, load_event, setup_from_codes

    event, seed, spread, shift, lap1_only = job
    roster, profiles = load_event(CONFIGS, event)
    setup = setup_from_codes(
        roster,
        profiles,
        None,
        data_dir=event_data_dir(ROOT, event),
        configs_dir=CONFIGS,
        racecraft_overrides={"start_spread_s": spread, "race_level_shift_s": shift},
    )
    if lap1_only:
        return float(sim_metrics(setup, seed, until_lap=1).extra["pass_causes"]["lap1"])
    return float(sim_metrics(setup, seed).overtakes)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--workers", type=int, default=14)
    parser.add_argument("--iterations", type=int, default=7)
    args = parser.parse_args()
    t0 = time.perf_counter()
    manifest = yaml.safe_load(SPLITS.read_text(encoding="utf-8"))
    splits = manifest["splits"]
    train, test = list(splits["dry_train"]), list(splits["dry_test"])
    dry_events = train + test
    laps = pd.read_parquet(DATASET)
    dry = laps[laps["event_id"].isin(dry_events)]
    pace = load_pace(CONFIGS)
    assert pace is not None, "fit the W2 driver pace first (scripts/fit_driver_pace.py)"
    noise = {c: float(v["lap_noise_s"]) for c, v in pace["drivers"].items()}
    sd_of = {
        str(c): noise.get(str(c), float(pace["field_noise_s"])) for c in dry["driver_code"].unique()
    }
    tp = default_traffic_params()
    shape, _ = load_shape(CONFIGS)
    enc = encounters(
        dry,
        shape,
        sd_of,
        dirty_air_window_s=tp.dirty_air_window_s,
        dirty_air_loss_s=tp.dirty_air_loss_s,
    )
    print(f"{len(enc)} encounters; fitting overtaking with driver terms ...")
    overtaking = fit_overtaking_skills(enc)
    print(
        f"threshold {overtaking['threshold_s']} s, scale {overtaking['scale_s']} s, "
        f"min gap {overtaking['min_gap_s']} s ({(time.perf_counter() - t0) / 60:.1f} min)"
    )
    old = json.loads((CONFIGS / "calibration" / "traffic_2024_v1.json").read_text(encoding="utf-8"))
    cfg: dict[str, Any] = {
        "label": "2024 racecraft (W3, D-044)",
        "events": dry_events,
        "overtaking": overtaking,
        "launch": {
            **launch_skills(dry, dry_events),
            "start_spread_s": float(old["start_spread_s"]),
        },
        "lap1_grid_profile": list(lap1_grid_profile()),
        "lap1_grid_profile_note": "editorial shape: Dartmouth Sports Analytics 2022-23 lap-1 "
        "incident involvements by grid row (rows 4-7 interpolated), mean 1",
    }
    out = CONFIGS / RACECRAFT_FILE

    def write() -> None:
        out.write_text(json.dumps(cfg, indent=1) + "\n", encoding="utf-8", newline="\n")

    write()
    real = {e: real_metrics(ROOT / "data" / "processed" / e) for e in train}
    lap1_target = statistics.fmean(real[e].extra["pass_causes"]["lap1"] for e in train)
    total_target = statistics.fmean(real[e].overtakes for e in train)
    lap1_jobs = [(e, event_seed(e, k)) for e in train for k in range(1, 2 * args.seeds + 1)]
    race_jobs = [(e, event_seed(e, k)) for e in train for k in range(1, args.seeds + 1)]
    history: list[dict[str, Any]] = []

    def run(pool: ProcessPoolExecutor, jobs: list[tuple[str, int]], lap1: bool) -> float:
        sp = float(cfg["launch"]["start_spread_s"])
        sh = float(cfg["overtaking"].get("race_level_shift_s", 0.0))
        return statistics.fmean(
            pool.map(passes, [(e, s, sp, sh, lap1) for e, s in jobs], chunksize=2)
        )

    def bisect(pool: ProcessPoolExecutor, what: str, lo: float, hi: float) -> None:
        """Spread: lap-1 passes rise with it; shift: all passes fall with it."""
        for _ in range(args.iterations):
            mid = (lo + hi) / 2
            if what == "spread":
                cfg["launch"]["start_spread_s"] = round(mid, 4)
                sim = run(pool, lap1_jobs, True)
                target, too_low = lap1_target, sim < lap1_target
            else:
                cfg["overtaking"]["race_level_shift_s"] = round(mid, 4)
                sim = run(pool, race_jobs, False)
                target, too_low = total_target, sim > total_target
            history.append({"param": what, "value": round(mid, 4), "passes": round(sim, 2)})
            print(f"{what} {mid:+.3f} s: {sim:.2f} passes (real {target:.2f})")
            lo, hi = (mid, hi) if too_low else (lo, mid)
        if what == "spread":
            cfg["launch"]["start_spread_s"] = round((lo + hi) / 2, 4)
        else:
            cfg["overtaking"]["race_level_shift_s"] = round((lo + hi) / 2, 4)

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        bisect(pool, "spread", 0.5, 3.0)
        bisect(pool, "shift", -0.5, 0.5)
        bisect(pool, "spread", 0.5, 3.0)  # the shift moves lap-1 passes a little too
        write()
        total = run(pool, race_jobs, False)
    real_total = total_target
    cfg["fit_history"] = history
    cfg["check"] = {
        "dry_train_passes_sim": round(total, 2),
        "dry_train_passes_real": round(real_total, 2),
        "dry_train_lap1_passes_real": round(lap1_target, 2),
    }
    write()
    print(f"all passes per race: simulated {total:.1f}, real {real_total:.1f}")
    body = {
        "model_name": "racecraft",
        "model_version": "v1",
        "kind": "ML on lap-level encounters (W2 noise convolved) with ridge driver terms; "
        "lap-1 gains; launch spread by simulation",
        "source_type": "estimated_from_data (lap-1 grid profile editorial)",
        "artifact": out.relative_to(ROOT).as_posix(),
        "artifact_sha256": "sha256:" + hashlib.sha256(out.read_bytes()).hexdigest(),
        "feature_schema_version": 1,
        "training_dataset_manifest": SPLITS.relative_to(ROOT).as_posix(),
        "training_dataset_content_sha256": manifest["file"]["content_sha256"],
        "training_races": sorted(dry_events),
        "metrics": {"check": cfg["check"], "held_out_check": "EXP-013"},
        "status": "validated_for_engine_use (W3, engine v0.13.0, D-044)",
    }
    (CONFIGS / "models" / "racecraft_2024_v1.yaml").write_text(
        yaml.safe_dump(body, sort_keys=False, width=100), encoding="utf-8", newline="\n"
    )
    print(f"done in {(time.perf_counter() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
