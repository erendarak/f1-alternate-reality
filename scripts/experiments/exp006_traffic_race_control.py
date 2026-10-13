"""EXP-006: overtaking and race control fitted on the 2024 season (M4 slice 5, D-035).

1. Overtaking: lap-level encounters on the 21 dry races (``calibration/traffic_fit``); the
   shrinkage of the track terms chosen by odd/even-lap cross-validation; a calibration table
   (observed vs predicted pass rate by expected advantage); a season-only fit on the 16 dry
   training races for the track-agnostic comparison.
2. Race control: retirement hazard, SC/VSC per retirement, SC length on all 21 dry races
   (engine); a training-only fit checked against the 5 dry test races.
3. Launch spread: lap 1 only, simulated on the dry training races, matched to their mean real
   lap-1 passes.
4. Simulation, every dry race x N seeds, three settings: v0.8 (editorial traffic and race
   control, no launch spread), season (fitted, one training-race threshold for every track)
   and per-track (the engine default). Passes by cause, retirements, SC, VSC vs the real race.

Writes configs/calibration/{traffic,race_control}_2024_v1.json and their manifests.

Usage:
    uv run python scripts/experiments/exp006_traffic_race_control.py --seeds 10 \\
        --json docs/experiments/exp006.json
"""

import argparse
import hashlib
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from f1ar.calibration.lap_model_fit import load_shape
from f1ar.calibration.race_control_fit import (
    RACE_CONTROL_FILE,
    fit_race_control,
    race_counts,
)
from f1ar.calibration.reference_metrics import PASS_CAUSES, real_metrics
from f1ar.calibration.simulated_metrics import sim_metrics
from f1ar.calibration.traffic_fit import (
    RIDGE,
    TRAFFIC_FILE,
    OvertakeFit,
    encounters,
    fit_overtaking,
    noise_sd,
    pass_probability_with_noise,
    traffic_params,
)
from f1ar.domain.race_state import RaceSetup
from f1ar.simulation.scenarios import (
    default_lap_model,
    default_race_control_params,
    default_traffic_params,
    event_data_dir,
    load_event,
    setup_from_codes,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "configs"
RIDGES = (1000.0, 150.0, 50.0, 30.0, 20.0, 8.0, 1.0)
SPREADS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0)


def dataset() -> tuple[pd.DataFrame, dict[str, str], str]:
    laps = pd.read_parquet(ROOT / "data" / "processed" / "datasets" / "laps_2024_v1.parquet")
    groups = laps[["event_id", "group"]].drop_duplicates().set_index("event_id")["group"]
    manifest = yaml.safe_load(
        (ROOT / "data" / "manifests" / "dataset_laps_2024_v1.yaml").read_text(encoding="utf-8")
    )
    dry = laps[laps["group"] != "wet"]
    return dry, {str(k): str(v) for k, v in groups.items()}, manifest["file"]["content_sha256"]


def noise_by_code(events: list[str]) -> dict[str, float]:
    lm = default_lap_model()
    out = {}
    for event_id in events:
        roster, profiles = load_event(CONFIGS, event_id)
        code = {e.driver_id: e.code for e in roster.entries}
        for p in profiles.profiles:
            out[code[p.driver_id]] = noise_sd(
                p.archetype.consistency, lm.noise_floor_s, lm.noise_inconsistency_s
            )
    return out


def log_likelihood(fit: OvertakeFit, enc: pd.DataFrame) -> float:
    margin = enc["advantage_s"].to_numpy() + fit.min_gap_s
    thr = enc["event_id"].map(lambda e: fit.threshold_for(str(e))).to_numpy()
    p = pass_probability_with_noise(margin, enc["sd_s"].to_numpy(), thr, fit.scale_s)
    p = np.clip(fit.eps + (1 - fit.eps) * p, 1e-9, 1 - 1e-9)
    y = enc["passed"].to_numpy().astype(float)
    return float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))


def ridge_cv(enc: pd.DataFrame) -> dict[str, float]:
    out = {}
    for ridge in RIDGES:
        total = 0.0
        for k in (0, 1):
            train, test = enc[enc["lap"] % 2 == k], enc[enc["lap"] % 2 != k]
            total += log_likelihood(fit_overtaking(train, ridge=ridge), test)
        out[str(ridge)] = round(total, 1)
    return out


def calibration_table(fit: OvertakeFit, enc: pd.DataFrame) -> list[dict[str, Any]]:
    margin = enc["advantage_s"].to_numpy() + fit.min_gap_s
    thr = enc["event_id"].map(lambda e: fit.threshold_for(str(e))).to_numpy()
    p = pass_probability_with_noise(margin, enc["sd_s"].to_numpy(), thr, fit.scale_s)
    p = fit.eps + (1 - fit.eps) * p
    bins = pd.cut(enc["advantage_s"], [-9, -1, -0.5, 0, 0.5, 1, 1.5, 2, 9])
    frame = pd.DataFrame({"bin": bins, "y": enc["passed"].astype(float), "p": p})
    rows = []
    for b, g in frame.groupby("bin", observed=True):
        rows.append(
            {
                "advantage_s": str(b),
                "encounters": len(g),
                "observed": round(float(g["y"].mean()), 3),
                "predicted": round(float(g["p"].mean()), 3),
            }
        )
    return rows


def write(path: Path, text: str) -> str:
    path.write_text(text, encoding="utf-8", newline="\n")
    return "sha256:" + hashlib.sha256(text.encode()).hexdigest()


def manifest(name: str, artifact: Path, sha: str, data_sha: str, extra: dict[str, Any]) -> None:
    body = {
        "model_name": name,
        "model_version": "v1",
        "artifact": artifact.as_posix(),
        "artifact_sha256": sha,
        "training_dataset_content_sha256": data_sha,
        **extra,
    }
    (CONFIGS / "models" / f"{name}_2024_v1.yaml").write_text(
        yaml.safe_dump(body, sort_keys=False, width=100), encoding="utf-8", newline="\n"
    )


def describe(values: list[float]) -> dict[str, float]:
    return {
        "mean": round(statistics.fmean(values), 2),
        "sd": round(statistics.pstdev(values), 2),
    }


def lap1_passes(setups: dict[str, RaceSetup], spread: float, seeds: int) -> float:
    values = []
    for setup in setups.values():
        traffic = setup.traffic.model_copy(update={"start_spread_s": spread})
        s = setup.model_copy(update={"traffic": traffic})
        values += [
            sim_metrics(s, seed, until_lap=1).extra["pass_causes"]["lap1"]
            for seed in range(1, seeds + 1)
        ]
    return statistics.fmean(values)


def interpolate(xs: list[float], ys: list[float], target: float) -> float:
    """The spread whose lap-1 passes hit ``target`` (the curve rises with the spread)."""
    return float(np.interp(target, ys, xs))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    t0 = time.perf_counter()

    # --- 1. overtaking ---------------------------------------------------------------------
    laps, groups, data_sha = dataset()
    events = sorted(laps["event_id"].unique())
    train = [e for e in events if groups[e] == "dry_train"]
    test = [e for e in events if groups[e] == "dry_test"]
    shape, _ = load_shape(CONFIGS)
    tp = default_traffic_params()
    enc = encounters(
        laps,
        shape,
        noise_by_code(events),
        dirty_air_window_s=tp.dirty_air_window_s,
        dirty_air_loss_s=tp.dirty_air_loss_s,
    )
    cv = ridge_cv(enc)
    best = max(cv, key=lambda k: cv[k])
    print("ridge CV (odd/even laps):", cv, "best", best, "used", RIDGE, flush=True)
    note = "EXP-006: lap-level encounters, 21 dry 2024 races; time losses editorial"
    fit = fit_overtaking(enc, note=note)
    season = fit_overtaking(enc[enc["event_id"].isin(train)], note="dry_train only")
    print("fit:", fit.min_gap_s, fit.threshold_s, fit.scale_s, fit.eps, flush=True)
    print("season (train) threshold:", season.threshold_s, season.scale_s, flush=True)

    # --- 2. race control ---------------------------------------------------------------------
    counts = {e: race_counts(ROOT / "data" / "processed" / e) for e in events}
    rc = fit_race_control(list(counts.values()), note="EXP-006: 21 dry 2024 races")
    rc_train = fit_race_control([counts[e] for e in train], note="dry_train only")
    rc_sha = write(CONFIGS / RACE_CONTROL_FILE, rc.to_json() + "\n")
    manifest(
        "race_control",
        RACE_CONTROL_FILE,
        rc_sha,
        data_sha,
        {
            "kind": "method of moments: retirement hazard, SC/VSC per retirement, SC length",
            "source_type": "estimated_from_data (crash/failure split editorial)",
            "training_races": events,
            "status": "experimental (21 races, small counts)",
        },
    )
    test_counts = [counts[e] for e in test]
    test_exposure = sum(c.car_laps_after_lap1 for c in test_counts)
    rc_check = {
        "lap1_retirements": {
            "predicted": round(
                rc_train.lap1_retirement_per_starter * sum(c.starters for c in test_counts), 2
            ),
            "observed": sum(c.lap1_retirements for c in test_counts),
        },
        "later_retirements": {
            "predicted": round(rc_train.retirement_per_car_lap * test_exposure, 2),
            "observed": sum(c.later_retirements for c in test_counts),
        },
    }
    print("race control:", rc, "\ntrain->test:", rc_check, flush=True)

    # --- 3. launch spread (lap 1 only, dry training races) ------------------------------------
    tr_sha = write(CONFIGS / TRAFFIC_FILE, fit.to_json() + "\n")
    setups = {}
    for e in events:
        roster, profiles = load_event(CONFIGS, e)
        setups[e] = setup_from_codes(
            roster, profiles, None, data_dir=event_data_dir(ROOT, e), configs_dir=CONFIGS
        )
    real = {e: real_metrics(ROOT / "data" / "processed" / e) for e in events}
    target = statistics.fmean(real[e].extra["pass_causes"]["lap1"] for e in train)
    curve = [lap1_passes({e: setups[e] for e in train}, s, args.seeds) for s in SPREADS]
    spread = round(interpolate(list(SPREADS), curve, target), 3)
    print("lap-1 passes by spread:", dict(zip(SPREADS, curve, strict=True)), flush=True)
    print(f"target {target:.2f} -> spread {spread}", flush=True)
    fit = OvertakeFit(**{**fit.__dict__, "start_spread_s": spread})
    tr_sha = write(CONFIGS / TRAFFIC_FILE, fit.to_json() + "\n")
    manifest(
        "traffic",
        TRAFFIC_FILE,
        tr_sha,
        data_sha,
        {
            "kind": "maximum likelihood on lap-level encounters (engine noise convolved), "
            "ridge-shrunk track thresholds; launch spread matched by simulation",
            "source_type": "estimated_from_data (passing/dirty-air/blue-flag losses editorial)",
            "training_races": events,
            "encounters": fit.encounters,
            "passes": fit.passes,
            "ridge": RIDGE,
            "status": "experimental (one race per track)",
        },
    )

    # --- 4. simulation: three settings ---------------------------------------------------------
    results: dict[str, Any] = {}
    for e in events:
        base = setup_from_codes(
            *load_event(CONFIGS, e), None, data_dir=event_data_dir(ROOT, e), configs_dir=CONFIGS
        )
        settings = {
            "v0.8": base.model_copy(
                update={
                    "traffic": default_traffic_params(),
                    "race_control": default_race_control_params(),
                }
            ),
            "season": base.model_copy(
                update={
                    "traffic": traffic_params(
                        OvertakeFit(
                            **{
                                **fit.__dict__,
                                "track_threshold_s": {},
                                "threshold_s": season.threshold_s,
                                "scale_s": season.scale_s,
                            }
                        ),
                        tr_sha,
                        e,
                        default_traffic_params(),
                    )
                }
            ),
            "per_track": base,
        }
        r = real[e]
        row: dict[str, Any] = {
            "group": groups[e],
            "real": {
                "passes": r.overtakes,
                **r.extra["pass_causes"],
                "retirements": r.retirements,
                "safety_cars": r.safety_cars,
                "vscs": r.vscs,
            },
            "threshold_s": fit.threshold_for(e),
        }
        for name, setup in settings.items():
            sims = [sim_metrics(setup, seed) for seed in range(1, args.seeds + 1)]
            row[name] = {
                "passes": describe([s.overtakes for s in sims]),
                **{
                    c: round(statistics.fmean(s.extra["pass_causes"][c] for s in sims), 2)
                    for c in PASS_CAUSES
                },
                "retirements": round(statistics.fmean(s.retirements for s in sims), 2),
                "safety_cars": round(statistics.fmean(s.safety_cars for s in sims), 2),
                "vscs": round(statistics.fmean(s.vscs for s in sims), 2),
            }
        results[e] = row
        print(
            f"{e:28} real {r.overtakes:3} | "
            + " | ".join(f"{n} {row[n]['passes']['mean']:6.1f}" for n in settings),
            flush=True,
        )

    summary = {}
    for name in ("v0.8", "season", "per_track"):
        by_group = {}
        for grp, evs in (("dry_train", train), ("dry_test", test), ("all_dry", events)):
            real_p = [results[e]["real"]["passes"] for e in evs]
            sim_p = [results[e][name]["passes"]["mean"] for e in evs]
            by_group[grp] = {
                "pass_mae": round(
                    statistics.fmean(abs(a - b) for a, b in zip(real_p, sim_p, strict=True)), 2
                ),
                "pass_correlation": round(float(np.corrcoef(real_p, sim_p)[0, 1]), 3)
                if len(evs) > 2
                else math.nan,
                "real_mean_passes": round(statistics.fmean(real_p), 2),
                "sim_mean_passes": round(statistics.fmean(sim_p), 2),
                "real_mean_retirements": round(
                    statistics.fmean(results[e]["real"]["retirements"] for e in evs), 2
                ),
                "sim_mean_retirements": round(
                    statistics.fmean(results[e][name]["retirements"] for e in evs), 2
                ),
                "real_mean_sc": round(
                    statistics.fmean(results[e]["real"]["safety_cars"] for e in evs), 2
                ),
                "sim_mean_sc": round(
                    statistics.fmean(results[e][name]["safety_cars"] for e in evs), 2
                ),
                "real_mean_vsc": round(
                    statistics.fmean(results[e]["real"]["vscs"] for e in evs), 2
                ),
                "sim_mean_vsc": round(statistics.fmean(results[e][name]["vscs"] for e in evs), 2),
                **{
                    f"real_{c}": round(statistics.fmean(results[e]["real"][c] for e in evs), 2)
                    for c in PASS_CAUSES
                },
                **{
                    f"sim_{c}": round(statistics.fmean(results[e][name][c] for e in evs), 2)
                    for c in PASS_CAUSES
                },
            }
        summary[name] = by_group
    print(json.dumps(summary, indent=1))

    out = {
        "seeds": args.seeds,
        "overtaking": {
            "fit": {k: v for k, v in fit.__dict__.items() if k != "fitted_on"},
            "season_train_only": {
                "threshold_s": season.threshold_s,
                "scale_s": season.scale_s,
                "min_gap_s": season.min_gap_s,
            },
            "ridge_cv": cv,
            "calibration": calibration_table(fit, enc),
            "lap1": {
                "target_mean_real_dry_train": round(target, 2),
                "by_spread": {str(s): round(v, 2) for s, v in zip(SPREADS, curve, strict=True)},
                "spread_s": spread,
            },
        },
        "race_control": {
            "fit": {k: v for k, v in rc.__dict__.items() if k != "fitted_on"},
            "train_only": {k: v for k, v in rc_train.__dict__.items() if k != "fitted_on"},
            "train_to_test": rc_check,
        },
        "races": results,
        "summary": summary,
        "seconds": round(time.perf_counter() - t0, 1),
    }
    if args.json:
        args.json.write_text(json.dumps(out, indent=1, default=str) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
