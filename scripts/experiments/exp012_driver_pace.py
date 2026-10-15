"""EXP-012: driver race pace and lap noise (W2, engine v0.12.0, D-043) vs the 2024 races.

1. **Lap level:** simulated laps (rule field, 3 per-race seeds per race) go through the same
   same-lap regression as the real laps (``driver_pace_fit.race_fit``): lap-to-lap residual SD
   per driver (clean air) and teammate gaps per race, real vs simulated, on the dry_train races
   (fitted) and the dry_test races (not fitted).
2. **Race level:** every dry race x ``--seeds`` per-race seeds, ``w2`` (the engine default) vs
   ``v0.11`` (the same engine without the driver-pace file: persona pace and noise): passes,
   stops per finisher, retirements, finishing order vs real, winner time.

Usage:
    uv run python scripts/experiments/exp012_driver_pace.py --seeds 10 \\
        --json docs/experiments/exp012.json
"""

import argparse
import functools
import hashlib
import json
import shutil
import statistics
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from f1ar.calibration.driver_pace_fit import CLEAN_GAP_S, MAX_RESIDUAL_S, PACE_FILE, race_fit
from f1ar.calibration.reference_metrics import real_metrics
from f1ar.calibration.simulated_metrics import sim_metrics, spearman
from f1ar.calibration.tyre_life_fit import green_laps

print = functools.partial(print, flush=True)  # noqa: A001 (progress through pipes)
ROOT = Path(__file__).resolve().parents[2]
SPLITS = ROOT / "data" / "manifests" / "dataset_laps_2024_v1.yaml"
DATASET = ROOT / "data" / "processed" / "datasets" / "laps_2024_v1.parquet"


def event_seed(event: str, k: int) -> int:
    return int(hashlib.sha256(event.encode()).hexdigest()[:6], 16) * 100 + k


@functools.cache
def _setup(event: str, configs: str) -> Any:
    from f1ar.simulation.scenarios import event_data_dir, load_event, setup_from_codes

    roster, profiles = load_event(ROOT / "configs", event)
    return setup_from_codes(
        roster, profiles, None, data_dir=event_data_dir(ROOT, event), configs_dir=Path(configs)
    )


def race_level(job: tuple[str, str, str, int]) -> dict[str, Any]:
    event, variant, configs, seed = job
    m = sim_metrics(_setup(event, configs), seed)
    stops = list(m.stops.values())
    return {
        "event": event,
        "variant": variant,
        "passes": m.overtakes,
        "stops": statistics.fmean(stops) if stops else float("nan"),
        "retirements": m.retirements,
        "winner_time_min": m.winner_time_s / 60,
        "order": m.order,
    }


def sim_laps(job: tuple[str, int]) -> list[dict[str, Any]]:
    """Simulated laps in the lap dataset's columns (clean air: no traffic loss on the lap)."""
    from f1ar.domain.race_state import RacePhase, WorldStatus
    from f1ar.simulation import lightweight_engine as engine

    event, seed = job
    setup = _setup(event, str(ROOT / "configs"))
    state = engine.start(engine.create_world(setup, seed=seed, world_id="laps"))
    entry = {d.driver_id: d for d in setup.drivers}
    stint = {d: 1 for d in entry}
    stint_lap = {d: 0 for d in entry}
    rows: list[dict[str, Any]] = []
    while state.world_status is WorldStatus.RUNNING:
        before = state
        res = engine.step(state)
        state = res.state
        pits = {e.driver_id for e in res.events if e.type == "PIT_STOP"}
        for e in res.events:
            if e.type != "LAP_COMPLETED" or e.driver_id is None:
                continue
            d = e.driver_id
            car = before.car(d)
            stint_lap[d] += 1
            green = before.race_control.phase is RacePhase.GREEN and (
                state.race_control.phase is RacePhase.GREEN
            )
            rows.append(
                {
                    "event_id": event,
                    "driver_code": entry[d].code,
                    "team_id": entry[d].team_id,
                    "lap_number": int(e.payload["lap"]),
                    "lap_time_s": float(e.payload["lap_time_s"]),
                    "compound": car.compound.value,
                    "tyre_age_start": 0,
                    "stint": stint[d],
                    "stint_lap": stint_lap[d],
                    "pit_in_lap": d in pits,
                    "pit_out_lap": stint_lap[d] == 1 and stint[d] > 1,
                    "track_status_start": "1" if green else "4",
                    "gap_ahead_s": 99.0 if car.traffic_loss_s <= 0.0 else 0.5,
                }
            )
            if d in pits:
                stint[d] += 1
                stint_lap[d] = 0
    return rows


def lap_level(laps: pd.DataFrame, events: list[str]) -> dict[str, Any]:
    """Residual SD per driver (clean air) and teammate gaps per race."""
    noise: dict[str, list[float]] = {}
    gaps: dict[tuple[str, str], float] = {}
    for ev in events:
        g = green_laps(laps[laps["event_id"] == ev])
        if len(g) < 200:
            continue
        terms, res = race_fit(g)
        clean = res[(res["gap_ahead_s"] > CLEAN_GAP_S) & (res["residual"].abs() < MAX_RESIDUAL_S)]
        for d, r in clean.groupby("driver_code"):
            noise.setdefault(str(d), []).extend(r["residual"].tolist())
        teams = g.groupby("driver_code")["team_id"].first()
        counts = g.groupby("driver_code").size()
        for team, members in teams.groupby(teams):
            pair = sorted(str(d) for d in members.index if counts.get(d, 0) >= 10)
            if len(pair) == 2:
                gaps[(ev, str(team))] = terms[pair[0]] - terms[pair[1]]
    return {
        "noise": {d: float(np.std(v)) for d, v in noise.items() if len(v) >= 30},
        "gaps": gaps,
    }


def compare(real: dict[str, Any], sim: dict[str, Any]) -> dict[str, Any]:
    common = sorted(set(real["noise"]) & set(sim["noise"]))
    rn = np.array([real["noise"][d] for d in common])
    sn = np.array([sim["noise"][d] for d in common])
    keys = sorted(set(real["gaps"]) & set(sim["gaps"]))
    rg = np.array([real["gaps"][k] for k in keys])
    sg = np.array([sim["gaps"][k] for k in keys])
    return {
        "noise_drivers": len(common),
        "noise_real_median": round(float(np.median(rn)), 3),
        "noise_sim_median": round(float(np.median(sn)), 3),
        "noise_corr": round(float(np.corrcoef(rn, sn)[0, 1]), 2) if len(common) > 2 else None,
        "gap_pairs": len(keys),
        "gap_real_abs_median": round(float(np.median(np.abs(rg))), 3),
        "gap_sim_abs_median": round(float(np.median(np.abs(sg))), 3),
        "gap_corr": round(float(np.corrcoef(rg, sg)[0, 1]), 2) if len(keys) > 2 else None,
        "gap_sign_agreement": round(float(np.mean(np.sign(rg) == np.sign(sg))), 2),
        "per_driver_noise": {
            d: [round(real["noise"][d], 3), round(sim["noise"][d], 3)] for d in common
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--workers", type=int, default=14)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    t0 = time.perf_counter()
    splits = yaml.safe_load(SPLITS.read_text(encoding="utf-8"))["splits"]
    train, test = list(splits["dry_train"]), list(splits["dry_test"])
    events = train + test
    old = Path(tempfile.mkdtemp(prefix="exp012_configs_"))
    shutil.copytree(ROOT / "configs", old, dirs_exist_ok=True)
    (old / PACE_FILE).unlink()
    real_laps = pd.read_parquet(DATASET)
    real_rm = {e: real_metrics(ROOT / "data" / "processed" / e) for e in events}
    jobs = [
        (e, v, cfg, event_seed(e, k))
        for e in events
        for v, cfg in (("w2", str(ROOT / "configs")), ("v0.11", str(old)))
        for k in range(1, args.seeds + 1)
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = []
        for i, row in enumerate(pool.map(race_level, jobs, chunksize=2), start=1):
            rows.append(row)
            if i % 60 == 0 or i == len(jobs):
                el = time.perf_counter() - t0
                print(f"{i}/{len(jobs)} races, {el / 60:.1f} min")
        lap_jobs = [(e, event_seed(e, k)) for e in events for k in (1, 2, 3)]
        sim_rows = [r for rs in pool.map(sim_laps, lap_jobs) for r in rs]
    shutil.rmtree(old, ignore_errors=True)
    sim_df = pd.DataFrame(sim_rows)
    laps_out: dict[str, Any] = {}
    for part, evs in (("train", train), ("test", test)):
        real = lap_level(real_laps, evs)
        sim = lap_level(sim_df, evs)
        laps_out[part] = compare(real, sim)
    races: dict[str, Any] = {}
    for part, evs in (("train", train), ("test", test), ("all_dry", events)):
        races[part] = {
            "real": {
                "passes": round(statistics.fmean(real_rm[e].overtakes for e in evs), 1),
                "retirements": round(statistics.fmean(real_rm[e].retirements for e in evs), 2),
            }
        }
        for v in ("w2", "v0.11"):
            mine = [r for r in rows if r["variant"] == v and r["event"] in evs]
            races[part][v] = {
                "passes": round(statistics.fmean(r["passes"] for r in mine), 1),
                "stops": round(statistics.fmean(r["stops"] for r in mine), 2),
                "retirements": round(statistics.fmean(r["retirements"] for r in mine), 2),
                "spearman": round(
                    statistics.fmean(spearman(r["order"], real_rm[r["event"]].order) for r in mine),
                    3,
                ),
                "winner_time_error_min": round(
                    statistics.fmean(
                        r["winner_time_min"] - real_rm[r["event"]].winner_time_s / 60 for r in mine
                    ),
                    2,
                ),
            }
    result = {
        "seeds": args.seeds,
        "laps": laps_out,
        "races": races,
        "seconds": round(time.perf_counter() - t0, 1),
    }
    for part in ("train", "test"):
        print("laps", part, {k: v for k, v in laps_out[part].items() if k != "per_driver_noise"})
    for part, r in races.items():
        print("races", part, r)
    if args.json:
        args.json.write_text(
            json.dumps(result, indent=1, default=float) + "\n", encoding="utf-8", newline="\n"
        )


if __name__ == "__main__":
    main()
