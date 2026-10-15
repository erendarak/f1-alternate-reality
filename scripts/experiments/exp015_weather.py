"""EXP-015: weather extensions (W6) - what 2024 measures, and the wet races with the fitted
wet-race model (fresh wet tyres under SC/VSC, wet incident factor).

1. **Track temperature vs tyre wear** (negative result check): within each dry race, a lap-time
   regression with lap terms, stint intercepts, an age slope per compound and an
   ``age x (track temp - race mean)`` term; and across tracks, the W1 medium tyre life vs the
   race's mean track temperature (Spearman).
2. **Wet races** (Canada, Britain, São Paulo; historical weather): W6 (the fitted file) vs
   the same engine without it (= v0.13 behaviour: no fresh wet sets under SC/VSC, editorial
   incident factor 2.0), seeds DIFFERENT from the fit's (offset 1000): neutralisations, retirements,
   finishing order, stops. Same races as the fit: an in-sample check, there are no other wet
   2024 races.
3. **Random-rain what-if** on 5 dry races: incidents with W6 vs v0.13.
4. **Dry races are unchanged** (wetness 0): identical results with the same seeds.

Usage:
    uv run python scripts/experiments/exp015_weather.py --seeds 20 \\
        --json docs/experiments/exp015.json
"""

import argparse
import functools
import hashlib
import json
import shutil
import statistics
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
import yaml
from scipy.sparse.linalg import lsqr
from scipy.stats import spearmanr

from f1ar.calibration.lap_model_fit import LIFE_FILE
from f1ar.calibration.reference_metrics import real_metrics, red_flag_periods
from f1ar.calibration.simulated_metrics import sim_metrics, spearman
from f1ar.calibration.tyre_life_fit import green_laps
from f1ar.calibration.wet_race_fit import WET_EVENTS, WET_FILE
from f1ar.simulation.scenarios import event_data_dir, load_event, setup_from_codes

print = functools.partial(print, flush=True)  # noqa: A001 (progress through pipes)
ROOT = Path(__file__).resolve().parents[2]
SPLITS = ROOT / "data" / "manifests" / "dataset_laps_2024_v1.yaml"
DATASET = ROOT / "data" / "processed" / "datasets" / "laps_2024_v1.parquet"
WHAT_IF = (
    "2024_bahrain_gp",
    "2024_spanish_gp",
    "2024_hungarian_gp",
    "2024_japanese_gp",
    "2024_italian_gp",
)


def event_seed(event: str, k: int) -> int:
    return int(hashlib.sha256(event.encode()).hexdigest()[:6], 16) * 100 + k


def temperature_within(laps: pd.DataFrame, events: list[str]) -> list[dict[str, Any]]:
    rows = []
    for ev in events:
        g = green_laps(laps[laps["event_id"] == ev]).reset_index(drop=True)
        temp = g["track_temp_c"].to_numpy(float)
        d_t = np.nan_to_num(temp - np.nanmean(temp))
        n = len(g)
        idx = np.arange(n)
        lap = pd.factorize(g["lap_number"])[0]
        stint = pd.factorize(g["driver_code"].astype(str) + ":" + g["stint"].astype(str))[0]
        _, ci = np.unique(g["compound"].astype(str), return_inverse=True)
        age = g["age"].to_numpy(float)
        x = sp.hstack(
            [
                sp.csr_matrix((np.ones(n), (idx, lap))),
                sp.csr_matrix((np.ones(n), (idx, stint)))[:, 1:],
                sp.csr_matrix((age, (idx, ci))),
                sp.csr_matrix((age * d_t, (idx, np.zeros(n, int))), shape=(n, 1)),
            ]
        ).tocsr()
        b = lsqr(x, g["lap_time_s"].to_numpy(float), atol=1e-12, btol=1e-12, iter_lim=50_000)[0]
        rows.append(
            {
                "event": ev,
                "track_temp_range_c": round(float(np.nanmax(temp) - np.nanmin(temp)), 1),
                "wear_per_lap_per_c_s": round(float(b[-1]), 5),
            }
        )
    return rows


def temperature_across(laps: pd.DataFrame) -> dict[str, Any]:
    life = json.loads((ROOT / "configs" / LIFE_FILE).read_text(encoding="utf-8"))["events"]
    temp = laps.groupby("event_id")["track_temp_c"].mean()
    pairs = [
        (float(temp[e]), float(v["life_raw"]["MEDIUM"]))
        for e, v in life.items()
        if e in temp.index and v["life_raw"].get("MEDIUM")
    ]
    x, y = zip(*pairs, strict=True)
    return {
        "spearman_temp_vs_medium_life": round(float(spearmanr(x, y).statistic), 3),
        "races": len(pairs),
    }


@functools.cache
def _setup(event: str, configs: str, random_rain: bool) -> Any:
    roster, profiles = load_event(ROOT / "configs", event)
    return setup_from_codes(
        roster,
        profiles,
        None,
        data_dir=event_data_dir(ROOT, event),
        configs_dir=Path(configs),
        random_rain=random_rain,
    )


def race(job: tuple[str, str, str, str, bool, int]) -> dict[str, Any]:
    kind, event, variant, configs, random_rain, seed = job
    m = sim_metrics(_setup(event, configs, random_rain), seed)
    ec = m.extra["event_counts"]
    stops = list(m.stops.values())
    return {
        "kind": kind,
        "event": event,
        "variant": variant,
        "random_rain": random_rain,
        "neutralisations": m.safety_cars + m.vscs + ec.get("RED_FLAG", 0),
        "retirements": m.retirements,
        "mistakes": ec.get("MISTAKE", 0),
        "rain_starts": ec.get("RAIN_STARTED", 0),
        "stops": statistics.fmean(stops) if stops else float("nan"),
        "order": m.order,
    }


def mean(rows: list[dict[str, Any]], key: str) -> float:
    return round(statistics.fmean(r[key] for r in rows), 3)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--workers", type=int, default=14)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    splits = yaml.safe_load(SPLITS.read_text(encoding="utf-8"))["splits"]
    dry = list(splits["dry_train"]) + list(splits["dry_test"])
    laps = pd.read_parquet(DATASET)
    within = temperature_within(laps, dry)
    rel = [r["wear_per_lap_per_c_s"] for r in within]
    temp = {
        "within_races": within,
        "within_median_s_per_lap_per_c": round(float(np.median(rel)), 5),
        "within_positive_share": round(float(np.mean(np.array(rel) > 0)), 3),
        "across_tracks": temperature_across(laps),
    }
    print("temperature", json.dumps({k: v for k, v in temp.items() if k != "within_races"}))
    old = Path(tempfile.mkdtemp(prefix="exp015_configs_"))
    shutil.copytree(ROOT / "configs", old, dirs_exist_ok=True)
    (old / WET_FILE).unlink()
    variants = (("w6", str(ROOT / "configs")), ("v0.13", str(old)))
    seeds = range(1000, 1000 + args.seeds)
    jobs = [
        ("wet", e, v, c, False, event_seed(e, k))
        for e in WET_EVENTS
        for v, c in variants
        for k in seeds
    ]
    jobs += [
        ("what_if", e, v, c, True, event_seed(e, k))
        for e in WHAT_IF
        for v, c in variants
        for k in seeds
    ]
    jobs += [
        ("dry", e, v, c, False, event_seed(e, k))
        for e in WHAT_IF[:3]
        for k in range(3)
        for v, c in variants
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = []
        for i, row in enumerate(pool.map(race, jobs, chunksize=2), 1):
            rows.append(row)
            if i % 50 == 0 or i == len(jobs):
                print(f"{i}/{len(jobs)} races")
    shutil.rmtree(old, ignore_errors=True)
    wet: dict[str, Any] = {}
    for e in WET_EVENTS:
        data = ROOT / "data" / "processed" / e
        rm = real_metrics(data)
        red = len(red_flag_periods(pd.read_parquet(data / "track_status.parquet")))
        wet[e] = {
            "real": {
                "neutralisations": rm.safety_cars + rm.vscs + red,
                "retirements": rm.retirements,
                "stops": round(statistics.fmean(rm.stops.values()), 3),
            }
        }
        for v, _ in variants:
            mine = [r for r in rows if r["kind"] == "wet" and r["event"] == e and r["variant"] == v]
            wet[e][v] = {
                "neutralisations": mean(mine, "neutralisations"),
                "retirements": mean(mine, "retirements"),
                "mistakes": mean(mine, "mistakes"),
                "stops": mean(mine, "stops"),
                "spearman_vs_real": round(
                    statistics.fmean(spearman(r["order"], rm.order) for r in mine), 3
                ),
            }
    print("wet", json.dumps(wet))
    what_if = {}
    for v, _ in variants:
        mine = [r for r in rows if r["variant"] == v and r["kind"] == "what_if"]
        what_if[v] = {
            k: mean(mine, k) for k in ("neutralisations", "retirements", "mistakes", "rain_starts")
        }
    print("random rain what-if", json.dumps(what_if))
    dry_rows = [r for r in rows if r["kind"] == "dry"]
    w6 = [r for r in dry_rows if r["variant"] == "w6"]
    old_rows = [r for r in dry_rows if r["variant"] == "v0.13"]
    identical = all(
        a["order"] == b["order"] and a["neutralisations"] == b["neutralisations"]
        for a, b in zip(w6, old_rows, strict=True)
    )
    print("dry races identical:", identical)
    if args.json:
        args.json.write_text(
            json.dumps(
                {
                    "temperature": temp,
                    "wet_races": wet,
                    "random_rain_what_if": what_if,
                    "dry_identical": identical,
                },
                indent=1,
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )


if __name__ == "__main__":
    main()
