"""Fit the W4 race events (engine v0.11.0, D-042).

Writes configs/calibration/race_events_2024_v1.json and its registry manifest.

1. From data (all 24 races; debris on the dry ones): pit crews and slow stops per team,
   failure multipliers per team, neutralisations without a retirement, red flags, penalties
   (``f1ar.calibration.race_events_fit``).
2. By simulation on the dry_train races only (the rule field, ``--seeds`` per race):
   ``risk_scale`` so simulated retirements match the real ones (crashes are scaled, failures
   aren't), then ``red_flag_per_crash`` so red flags per race match 2024's 3 in 24.
   A second run with the fitted values checks both.

Usage:
    uv run python scripts/fit_race_events.py --seeds 5     # ~5-10 min
"""

import argparse
import functools
import hashlib
import json
import statistics
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import yaml

from f1ar.calibration.race_events_fit import EVENTS_FILE, fit
from f1ar.calibration.reference_metrics import real_metrics

print = functools.partial(print, flush=True)  # noqa: A001 (progress through pipes)
ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "data" / "manifests" / "dataset_laps_2024_v1.yaml"


def simulate(job: tuple[str, int, dict[str, float]]) -> dict[str, Any]:
    from f1ar.domain.race_state import WorldStatus
    from f1ar.simulation import lightweight_engine as engine
    from f1ar.simulation.scenarios import event_data_dir, load_event, setup_from_codes

    event, seed, overrides = job
    roster, profiles = load_event(ROOT / "configs", event)
    setup = setup_from_codes(
        roster,
        profiles,
        None,
        data_dir=event_data_dir(ROOT, event),
        configs_dir=ROOT / "configs",
        race_events_overrides=overrides,
    )
    state = engine.start(engine.create_world(setup, seed=seed, world_id="fit"))
    counts: Counter[str] = Counter()
    while state.world_status is WorldStatus.RUNNING:
        res = engine.step(state)
        state = res.state
        for e in res.events:
            if e.type == "CAR_RETIRED":
                counts[f"retired_{e.payload['reason']}"] += 1
            elif e.type in ("RED_FLAG", "TIME_PENALTY", "SAFETY_CAR_DEPLOYED", "VSC_DEPLOYED"):
                counts[e.type] += 1
    rc = setup.race_control
    used = {"risk_scale": rc.risk_scale, "red_flag_per_crash": rc.red_flag_per_crash}
    return {"event": event, "seed": seed, "used": used, **counts}


def run(jobs: list[tuple[str, int, dict[str, float]]], workers: int) -> list[dict[str, Any]]:
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(simulate, jobs, chunksize=1))


RATE_KEYS = (
    "collision_per_pass",
    "false_start_per_starter",
    "pit_speeding_per_stop",
    "other_per_car_lap",
)


def event_seed(event: str, k: int) -> int:
    """A different seed per race: keyed draws don't include the event, so seeds 1..N on every
    track would repeat the same luck (lap-1 crashes, red flags) everywhere (D-042)."""
    return int(hashlib.sha256(event.encode()).hexdigest()[:6], 16) * 100 + k


def per_race(rows: list[dict[str, Any]], key: str) -> float:
    return statistics.fmean(float(r.get(key, 0)) for r in rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--workers", type=int, default=14)
    args = parser.parse_args()
    t0 = time.perf_counter()
    manifest = yaml.safe_load(SPLITS.read_text(encoding="utf-8"))
    splits = manifest["splits"]
    train, test, wet = list(splits["dry_train"]), list(splits["dry_test"]), list(splits["wet"])
    events = train + test + wet
    real = {e: real_metrics(ROOT / "data" / "processed" / e) for e in events}
    cfg = fit(ROOT, events, train + test, {e: m.overtakes for e in events for m in [real[e]]})
    out = ROOT / "configs" / EVENTS_FILE

    def write() -> None:
        out.write_text(json.dumps(cfg, indent=1) + "\n", encoding="utf-8", newline="\n")

    write()
    print(f"data fit written ({(time.perf_counter() - t0) / 60:.1f} min); simulating dry_train")
    seeds = range(1, args.seeds + 1)
    base = run(
        [
            (e, s, {"risk_scale": 1.0, "red_flag_per_crash": 0.0})
            for e in train
            for s in [event_seed(e, k) for k in seeds]
        ],
        args.workers,
    )
    crashes = per_race(base, "retired_CRASH")
    failures = per_race(base, "retired_FAILURE")
    real_ret = statistics.fmean(real[e].retirements for e in train)
    scale = max(0.2, (real_ret - failures) / crashes) if crashes > 0 else 1.0
    reds_per_race = cfg["red_flags"]["per_race"]
    red = min(1.0, reds_per_race / (scale * crashes)) if crashes > 0 else 0.0
    print(
        f"risk 1.0: crashes {crashes:.2f}, failures {failures:.2f} per race vs real retirements "
        f"{real_ret:.2f} -> risk_scale {scale:.3f}, red_flag_per_crash {red:.3f}"
    )
    cfg["fitted"] = {
        "risk_scale": round(scale, 4),
        "red_flag_per_crash": round(red, 4),
        "fit": {
            "races": train,
            "seeds": args.seeds,
            "sim_crashes_per_race_at_1": round(crashes, 3),
            "sim_failures_per_race": round(failures, 3),
            "real_retirements_per_race": round(real_ret, 3),
            "target_red_flags_per_race": reds_per_race,
        },
    }
    write()
    check = run(
        [(e, s, {}) for e in train for s in [event_seed(e, k) for k in seeds]], args.workers
    )
    cfg["fitted"]["check"] = {
        "retirements_per_race": round(
            per_race(check, "retired_CRASH") + per_race(check, "retired_FAILURE"), 3
        ),
        "red_flags_per_race": round(per_race(check, "RED_FLAG"), 3),
        "safety_cars_per_race": round(per_race(check, "SAFETY_CAR_DEPLOYED"), 3),
        "vscs_per_race": round(per_race(check, "VSC_DEPLOYED"), 3),
        "penalties_per_race": round(per_race(check, "TIME_PENALTY"), 3),
    }
    # penalties: one scale on all four rates so the simulated train field matches the real one
    # (passes, laps and penalty-prone drivers differ between the data and the simulation)
    from f1ar.calibration.race_events_fit import event_counts

    real_pens = statistics.fmean(
        sum(event_counts(ROOT / "data" / "processed" / e, ROOT / "configs", e).penalties.values())
        for e in train
    )
    sim_pens = cfg["fitted"]["check"]["penalties_per_race"]
    pen_scale = real_pens / sim_pens if sim_pens > 0 else 1.0
    pens = cfg["penalties"]
    pens["raw"] = {k: pens[k] for k in RATE_KEYS}
    for k in RATE_KEYS:
        pens[k] = round(pens[k] * pen_scale, 6)
    pens["scale"] = round(pen_scale, 4)
    pens["scale_note"] = (
        f"x{pen_scale:.3f}: simulated dry_train {sim_pens:.2f} vs real {real_pens:.2f} per race"
    )
    write()
    again = run(
        [(e, s, {}) for e in train for s in [event_seed(e, k) for k in seeds]], args.workers
    )
    cfg["fitted"]["check"]["penalties_per_race_after_scale"] = round(
        per_race(again, "TIME_PENALTY"), 3
    )
    write()
    print("check:", cfg["fitted"]["check"], "used:", check[0]["used"], base[0]["used"])
    body = {
        "model_name": "race_events",
        "model_version": "v1",
        "kind": "rates and shrunk team multipliers from 2024 timing; 2 values fitted by simulation",
        "source_type": "estimated_from_data",
        "artifact": out.relative_to(ROOT).as_posix(),
        "artifact_sha256": "sha256:" + hashlib.sha256(out.read_bytes()).hexdigest(),
        "feature_schema_version": 1,
        "training_dataset_manifest": SPLITS.relative_to(ROOT).as_posix(),
        "training_dataset_content_sha256": manifest["file"]["content_sha256"],
        "training_races": sorted(events),
        "metrics": {"dry_train_check": cfg["fitted"]["check"], "held_out_check": "EXP-011"},
        "status": "validated_for_engine_use (W4, engine v0.11.0, D-042)",
    }
    (ROOT / "configs" / "models" / "race_events_2024_v1.yaml").write_text(
        yaml.safe_dump(body, sort_keys=False, width=100), encoding="utf-8", newline="\n"
    )
    print(f"done in {(time.perf_counter() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
