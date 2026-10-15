"""Fit the W6 wet-race model (D-046): writes configs/calibration/wet_race_2024_v1.json and its
registry manifest.

1. Fresh wet tyres under SC/VSC: threshold model on the real neutralisations of the wet races
   (``calibration/wet_race_fit.refresh_observations`` / ``fit_refresh``).
2. The wet incident factor, with that rule on: simulates the three wet 2024 races (historical
   weather) for a grid of ``wet_incident_factor`` values with per-race seeds and compares
   neutralisations (SC + VSC + red flags) and retirements per race with the real ones (Poisson
   likelihood; ``calibration/wet_race_fit``).

Usage:
    uv run python scripts/fit_wet_race.py --seeds 20      # ~3 min
"""

import argparse
import functools
import hashlib
import json
import statistics
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
import yaml

from f1ar.calibration.reference_metrics import real_metrics, red_flag_periods
from f1ar.calibration.simulated_metrics import sim_metrics
from f1ar.calibration.wet_race_fit import (
    GRID,
    WET_EVENTS,
    WET_FILE,
    fit_refresh,
    profile_fit,
    refresh_observations,
)
from f1ar.simulation.scenarios import event_data_dir, load_event, setup_from_codes

print = functools.partial(print, flush=True)  # noqa: A001 (progress through pipes)
ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "processed" / "datasets" / "laps_2024_v1.parquet"
SPLITS = ROOT / "data" / "manifests" / "dataset_laps_2024_v1.yaml"


def event_seed(event: str, k: int) -> int:
    return int(hashlib.sha256(event.encode()).hexdigest()[:6], 16) * 100 + k


def real_counts(event: str) -> dict[str, float]:
    data = ROOT / "data" / "processed" / event
    m = real_metrics(data)
    red = len(red_flag_periods(pd.read_parquet(data / "track_status.parquet")))
    return {
        "neutralisations": float(m.safety_cars + m.vscs + red),
        "retirements": float(m.retirements),
        "safety_cars": float(m.safety_cars),
        "vscs": float(m.vscs),
        "red_flags": float(red),
    }


def run(job: tuple[str, float, int, dict[str, float]]) -> tuple[str, float, dict[str, float]]:
    event, factor, seed, refresh = job
    roster, profiles = load_event(ROOT / "configs", event)
    setup = setup_from_codes(
        roster,
        profiles,
        None,
        data_dir=event_data_dir(ROOT, event),
        wet_overrides={"wet_incident_factor": factor, **refresh},
    )
    m = sim_metrics(setup, seed)
    red = float(m.extra["event_counts"].get("RED_FLAG", 0))
    return (
        event,
        factor,
        {
            "neutralisations": m.safety_cars + m.vscs + red,
            "retirements": float(m.retirements),
            "safety_cars": float(m.safety_cars),
            "vscs": float(m.vscs),
            "red_flags": red,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--workers", type=int, default=14)
    args = parser.parse_args()
    laps = pd.read_parquet(DATASET)
    obs = []
    for e in WET_EVENTS:
        status = pd.read_parquet(ROOT / "data" / "processed" / e / "track_status.parquet")
        red = [start for start, _ in red_flag_periods(status)]
        obs += [{**o, "event": e} for o in refresh_observations(laps[laps["event_id"] == e], red)]
    refresh_fit = fit_refresh(obs)
    print("refresh", json.dumps(refresh_fit))
    refresh = {
        "wet_refresh_min_age_laps": float(refresh_fit["min_age_laps"]),
        "wet_refresh_share": float(refresh_fit["share"]),
    }
    real = {e: real_counts(e) for e in WET_EVENTS}
    print("real", json.dumps(real))
    jobs = [
        (e, f, event_seed(e, k), refresh)
        for f in GRID
        for e in WET_EVENTS
        for k in range(args.seeds)
    ]
    rows = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, row in enumerate(pool.map(run, jobs, chunksize=2), 1):
            rows.append(row)
            if i % 50 == 0 or i == len(jobs):
                left = (time.time() - t0) / i * (len(jobs) - i)
                print(f"{i}/{len(jobs)} races, ~{left / 60:.1f} min left")
    keys = list(rows[0][2])
    means = {
        f: {
            e: {k: statistics.fmean(r[2][k] for r in rows if r[0] == e and r[1] == f) for k in keys}
            for e in WET_EVENTS
        }
        for f in GRID
    }
    fit = profile_fit(means, real)
    print("fit", json.dumps(fit))
    near = min(GRID, key=lambda g: abs(g - fit["best"]))
    cfg = {
        "label": "2024 incidents in the wet (W6, D-046)",
        "fitted_on": list(WET_EVENTS),
        "seeds_per_race": args.seeds,
        "wet_refresh": {**refresh_fit, "observations": obs},
        "wet_incident_factor": fit,
        "previous_editorial_value": 2.0,
        "real_per_race": real,
        "sim_per_race": {
            str(f): {e: {k: round(v, 3) for k, v in means[f][e].items()} for e in WET_EVENTS}
            for f in (0.0, 2.0, near)
        },
        "not_modelled": {
            "track temperature -> tyre wear": "no effect in 2024 (within races: sign flips, median "
            "~0 per degree; across tracks: Spearman 0.00 with tyre life); EXP-015",
            "graining / blistering": "no published numbers",
            "visibility red flags, SC vs VSC choice in rain": "rain is on/off in the data (no "
            "intensity)",
        },
    }
    out = ROOT / "configs" / WET_FILE
    out.write_text(json.dumps(cfg, indent=1) + "\n", encoding="utf-8", newline="\n")
    manifest = yaml.safe_load(SPLITS.read_text(encoding="utf-8"))
    body = {
        "model_name": "wet_race",
        "model_version": "v1",
        "kind": "fresh wet tyres under SC/VSC (threshold ML on real neutralisations); wet incident "
        "factor by simulation (Poisson likelihood, profile 95% range)",
        "source_type": "estimated_from_data",
        "artifact": out.relative_to(ROOT).as_posix(),
        "artifact_sha256": "sha256:" + hashlib.sha256(out.read_bytes()).hexdigest(),
        "feature_schema_version": 1,
        "training_dataset_manifest": SPLITS.relative_to(ROOT).as_posix(),
        "training_dataset_content_sha256": manifest["file"]["content_sha256"],
        "training_races": list(WET_EVENTS),
        "metrics": {
            "refresh_min_age_laps": refresh_fit["min_age_laps"],
            "refresh_share": refresh_fit["share"],
            "best": fit["best"],
            "range_95": fit["range_95"],
            "held_out_check": "none (3 wet races, all used); EXP-015",
        },
        "status": "validated_for_engine_use (W6, D-046)",
    }
    (ROOT / "configs" / "models" / "wet_race_2024_v1.yaml").write_text(
        yaml.safe_dump(body, sort_keys=False, width=100), encoding="utf-8", newline="\n"
    )
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
