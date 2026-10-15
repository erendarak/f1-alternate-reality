"""EXP-013: racecraft (W3, engine v0.13.0, D-044) vs the 2024 races.

Every dry 2024 race x ``--seeds`` per-race seeds, ``w3`` (the engine default) vs ``v0.12`` (the
same engine without the racecraft file: W2's global threshold shift, no driver terms, uniform
lap-1 risk). Measured against the real races:

- passes per race (and per track: correlation with the real counts), finishing order, stops;
- per driver: share of the season's passes made and suffered (real: the lap-level encounters'
  passes; simulated: OVERTAKE events), correlation real vs simulated;
- lap 1: mean positions changed, gains by grid third, per-driver mean lap-1 gain vs real;
  lap-1 retirements by grid third.

Usage:
    uv run python scripts/experiments/exp013_racecraft.py --seeds 10 \\
        --json docs/experiments/exp013.json
"""

import argparse
import functools
import hashlib
import json
import shutil
import statistics
import tempfile
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from f1ar.calibration.racecraft_fit import RACECRAFT_FILE, load_racecraft
from f1ar.calibration.reference_metrics import real_metrics
from f1ar.calibration.simulated_metrics import spearman

print = functools.partial(print, flush=True)  # noqa: A001 (progress through pipes)
ROOT = Path(__file__).resolve().parents[2]
SPLITS = ROOT / "data" / "manifests" / "dataset_laps_2024_v1.yaml"
DATASET = ROOT / "data" / "processed" / "datasets" / "laps_2024_v1.parquet"


def event_seed(event: str, k: int) -> int:
    return int(hashlib.sha256(event.encode()).hexdigest()[:6], 16) * 100 + k


def third(slot: int) -> str:
    return "front" if slot <= 6 else "mid" if slot <= 13 else "back"


@functools.cache
def _setup(event: str, configs: str) -> Any:
    from f1ar.simulation.scenarios import event_data_dir, load_event, setup_from_codes

    roster, profiles = load_event(ROOT / "configs", event)
    return setup_from_codes(
        roster, profiles, None, data_dir=event_data_dir(ROOT, event), configs_dir=Path(configs)
    )


def run(job: tuple[str, str, str, int]) -> dict[str, Any]:
    """One race: passes by driver, lap-1 positions, lap-1 retirements by grid third, result."""
    from f1ar.calibration.reference_metrics import count_overtakes
    from f1ar.domain.race_state import CarStatus, WorldStatus
    from f1ar.simulation import lightweight_engine as engine

    event, variant, configs, seed = job
    setup = _setup(event, configs)
    code = {d.driver_id: d.code for d in setup.drivers}
    grid = {d.code: d.grid_position for d in setup.drivers}
    state = engine.start(engine.create_world(setup, seed=seed, world_id="x"))
    made: Counter[str] = Counter()
    suffered: Counter[str] = Counter()
    lap1_out: Counter[str] = Counter()
    crossings: dict[int, dict[str, float]] = {}
    stopped: dict[str, set[int]] = {}
    while state.world_status is WorldStatus.RUNNING:
        res = engine.step(state)
        state = res.state
        for e in res.events:
            if e.driver_id is None:
                continue
            c = code[e.driver_id]
            if e.type == "OVERTAKE" and int(e.payload["lap"]) > 1:
                made[c] += 1
                suffered[code[str(e.payload["passed"])]] += 1
            elif e.type == "CAR_RETIRED" and int(e.payload["lap"]) == 1:
                lap1_out[third(grid[c])] += 1
            elif e.type == "LAP_COMPLETED":
                crossings.setdefault(int(e.payload["lap"]), {})[c] = float(
                    e.payload["crossing_time_s"]
                )
            elif e.type == "PIT_STOP":
                stopped.setdefault(c, set()).add(int(e.payload["lap"]))
    first = sorted(crossings.get(1, {}), key=lambda c: crossings[1][c])
    order = [code[d] for d in engine.standings(state)]
    finishers = [code[c.driver_id] for c in state.cars if c.status is CarStatus.FINISHED]
    stops = [state.car(d).pit_stops for d in code if code[d] in finishers]
    order_grid = [d.code for d in sorted(setup.drivers, key=lambda d: d.grid_position)]
    return {
        "event": event,
        "variant": variant,
        "passes": count_overtakes(crossings, order_grid, stopped),
        "made": dict(made),
        "suffered": dict(suffered),
        "lap1_gain": {c: grid[c] - (i + 1) for i, c in enumerate(first)},
        "lap1_out": dict(lap1_out),
        "order": order,
        "stops": statistics.fmean(stops) if stops else float("nan"),
    }


def real_lap1(laps: pd.DataFrame, events: list[str]) -> dict[str, Any]:
    r = laps[
        laps["event_id"].isin(events) & (laps["lap_number"] == 1) & (laps["grid_position"] > 0)
    ]
    r = r[r["position_end"].notna()]
    gain = r["grid_position"] - r["position_end"]
    thirds = r["grid_position"].map(lambda g: third(int(g)))
    return {
        "mean_abs_change": round(float(gain.abs().mean()), 3),
        "gain_by_third": {k: round(float(v), 3) for k, v in gain.groupby(thirds).mean().items()},
        "per_driver": {str(k): float(v) for k, v in gain.groupby(r["driver_code"]).mean().items()},
    }


def corr(a: dict[str, float], b: dict[str, float]) -> float | None:
    keys = sorted(set(a) & set(b))
    if len(keys) < 3:
        return None
    return round(float(np.corrcoef([a[k] for k in keys], [b[k] for k in keys])[0, 1]), 3)


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
    old = Path(tempfile.mkdtemp(prefix="exp013_configs_"))
    shutil.copytree(ROOT / "configs", old, dirs_exist_ok=True)
    (old / RACECRAFT_FILE).unlink()
    laps = pd.read_parquet(DATASET)
    real = {e: real_metrics(ROOT / "data" / "processed" / e) for e in events}
    cfg = load_racecraft(ROOT / "configs")
    assert cfg is not None
    real_made = {d: float(v["passes_made"]) for d, v in cfg["overtaking"]["drivers"].items()}
    real_suffered = {
        d: float(v["passes_suffered"]) for d, v in cfg["overtaking"]["drivers"].items()
    }
    jobs = [
        (e, v, c, event_seed(e, k))
        for e in events
        for v, c in (("w3", str(ROOT / "configs")), ("v0.12", str(old)))
        for k in range(1, args.seeds + 1)
    ]
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, row in enumerate(pool.map(run, jobs, chunksize=2), start=1):
            rows.append(row)
            if i % 60 == 0 or i == len(jobs):
                print(f"{i}/{len(jobs)} races, {(time.perf_counter() - t0) / 60:.1f} min")
    shutil.rmtree(old, ignore_errors=True)
    rl = real_lap1(laps, events)
    result: dict[str, Any] = {
        "seeds": args.seeds,
        "real": {
            "passes": round(statistics.fmean(real[e].overtakes for e in events), 1),
            "lap1": {k: v for k, v in rl.items() if k != "per_driver"},
        },
    }
    for v in ("w3", "v0.12"):
        mine = [r for r in rows if r["variant"] == v]
        made: Counter[str] = Counter()
        suffered: Counter[str] = Counter()
        gains: dict[str, list[int]] = {}
        out: Counter[str] = Counter()
        for r in mine:
            made.update(r["made"])
            suffered.update(r["suffered"])
            out.update(r["lap1_out"])
            for c, g in r["lap1_gain"].items():
                gains.setdefault(c, []).append(g)
        per_race = {
            e: statistics.fmean(r["passes"] for r in mine if r["event"] == e) for e in events
        }
        all_gains = [g for gs in gains.values() for g in gs]
        result[v] = {
            "passes": round(statistics.fmean(per_race.values()), 1),
            "passes_per_track_corr": corr(per_race, {e: float(real[e].overtakes) for e in events}),
            "driver_made_corr": corr({k: float(n) for k, n in made.items()}, real_made),
            "driver_suffered_corr": corr({k: float(n) for k, n in suffered.items()}, real_suffered),
            "lap1_mean_abs_change": round(float(np.mean(np.abs(all_gains))), 3),
            "lap1_gain_per_driver_corr": corr(
                {c: float(np.mean(g)) for c, g in gains.items()}, rl["per_driver"]
            ),
            "lap1_retirements_by_third_per_race": {
                k: round(n / len(mine), 3) for k, n in out.items()
            },
            "spearman": round(
                statistics.fmean(spearman(r["order"], real[r["event"]].order) for r in mine), 3
            ),
            "stops": round(statistics.fmean(r["stops"] for r in mine), 2),
        }
        print(v, result[v])
    print("real", result["real"])
    result["seconds"] = round(time.perf_counter() - t0, 1)
    if args.json:
        args.json.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
