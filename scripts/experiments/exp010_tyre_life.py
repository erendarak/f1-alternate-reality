"""EXP-010: does the tyre-life world (W1, engine v0.10.0, D-041) race like 2024? (all 21 dry races)

For every dry 2024 race x ``--seeds`` the rule field is simulated twice on identical seeds:
``life`` (the W1 tyre model, the engine default) and ``v0.9`` (the same engine without the
tyre-life config: the M4 tyre model, which reproduces engine v0.9.0's behaviour). Each is
compared with the real race, using the measurements of EXP-004/006 (``reference_metrics``):
stops per finisher and 1/2/3-stop shares, compound sequences, passes, retirements, safety cars,
winner time, finishing order (Spearman). Train (16) and test (5) races are reported apart: the
tyre-life globals were fitted on the training races only; each race's own life and wear
inputs are in-sample (like the car pace offsets).

Also: the same-lap regression of ``tyre_life_fit`` on simulated laps (3 seeds per race) vs on
the real laps: on race laps the compounds should look as close as in reality.

Usage:
    uv run python scripts/experiments/exp010_tyre_life.py --seeds 10 \\
        --json docs/experiments/exp010.json
"""

import argparse
import functools
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

from f1ar.calibration.lap_model_fit import LIFE_FILE
from f1ar.calibration.reference_metrics import real_metrics
from f1ar.calibration.simulated_metrics import sim_metrics, spearman
from f1ar.calibration.tyre_life_fit import fit_wear, green_laps

print = functools.partial(print, flush=True)  # noqa: A001 (progress through pipes)
ROOT = Path(__file__).resolve().parents[2]
SPLITS = ROOT / "data" / "manifests" / "dataset_laps_2024_v1.yaml"
DATASET = ROOT / "data" / "processed" / "datasets" / "laps_2024_v1.parquet"


@functools.cache
def _setup(event: str, configs: str) -> Any:
    from f1ar.simulation.scenarios import event_data_dir, load_event, setup_from_codes

    roster, profiles = load_event(ROOT / "configs", event)
    return setup_from_codes(
        roster, profiles, None, data_dir=event_data_dir(ROOT, event), configs_dir=Path(configs)
    )


def run(job: tuple[str, str, str, int]) -> dict[str, Any]:
    event, variant, configs, seed = job
    m = sim_metrics(_setup(event, configs), seed)
    stops = list(m.stops.values())
    return {
        "event": event,
        "variant": variant,
        "seed": seed,
        "stops": statistics.fmean(stops) if stops else float("nan"),
        "stop_counts": dict(Counter(stops)),
        "sequences": dict(Counter(m.sequences.values())),
        "passes": m.overtakes,
        "retirements": m.retirements,
        "safety_cars": m.safety_cars,
        "vscs": m.vscs,
        "winner_time_s": m.winner_time_s,
        "order": m.order,
    }


def sim_laps(job: tuple[str, int]) -> list[dict[str, Any]]:
    """Lap rows of a simulated race in the lap dataset's columns (for ``green_laps``)."""
    from f1ar.domain.race_state import RacePhase, WorldStatus
    from f1ar.simulation import lightweight_engine as engine

    event, seed = job
    setup = _setup(event, str(ROOT / "configs"))
    state = engine.start(engine.create_world(setup, seed=seed, world_id="laps"))
    code = {d.driver_id: d.code for d in setup.drivers}
    rows: list[dict[str, Any]] = []
    stint = {d.driver_id: 1 for d in setup.drivers}
    stint_lap = {d.driver_id: 0 for d in setup.drivers}
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
                    "driver_code": code[d],
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


def summarise(rows: list[dict[str, Any]], real: dict[str, Any], events: list[str]) -> Any:
    out: dict[str, Any] = {}
    for variant in ("life", "v0.9"):
        mine = [r for r in rows if r["variant"] == variant and r["event"] in events]
        per_race: dict[str, dict[str, Any]] = {}
        for ev in events:
            rs = [r for r in mine if r["event"] == ev]
            counts: Counter[int] = Counter()
            for r in rs:
                counts.update({int(k): v for k, v in r["stop_counts"].items()})
            n = sum(counts.values()) or 1
            per_race[ev] = {
                "stops": round(statistics.fmean(r["stops"] for r in rs), 2),
                "share_1_2_3": [
                    round(counts[1] / n, 2),
                    round(counts[2] / n, 2),
                    round(sum(v for k, v in counts.items() if k >= 3) / n, 2),
                ],
                "passes": round(statistics.fmean(r["passes"] for r in rs), 1),
                "retirements": round(statistics.fmean(r["retirements"] for r in rs), 2),
                "winner_time_min": round(statistics.fmean(r["winner_time_s"] for r in rs) / 60, 2),
                "spearman": round(
                    statistics.fmean(spearman(r["order"], real[ev]["order"]) for r in rs), 3
                ),
                "top_sequences": dict(
                    sum((Counter(r["sequences"]) for r in rs), Counter()).most_common(4)
                ),
            }
        sim = np.array([per_race[e]["stops"] for e in events])
        rl = np.array([real[e]["stops"] for e in events])
        out[variant] = {
            "stops_mean": round(float(sim.mean()), 2),
            "stops_mae": round(float(np.abs(sim - rl).mean()), 2),
            "stops_corr": round(float(np.corrcoef(sim, rl)[0, 1]), 2) if sim.std() > 0 else None,
            "passes_mean": round(statistics.fmean(per_race[e]["passes"] for e in events), 1),
            "retirements_mean": round(
                statistics.fmean(per_race[e]["retirements"] for e in events), 2
            ),
            "spearman_mean": round(statistics.fmean(per_race[e]["spearman"] for e in events), 3),
            "winner_time_error_min": round(
                statistics.fmean(
                    per_race[e]["winner_time_min"] - real[e]["winner_time_min"] for e in events
                ),
                2,
            ),
            "per_race": per_race,
        }
    out["real"] = {
        "stops_mean": round(statistics.fmean(real[e]["stops"] for e in events), 2),
        "passes_mean": round(statistics.fmean(real[e]["passes"] for e in events), 1),
        "retirements_mean": round(statistics.fmean(real[e]["retirements"] for e in events), 2),
        "per_race": {e: real[e] for e in events},
    }
    return out


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
    # v0.9 condition: the same configs without the tyre-life file
    old = Path(tempfile.mkdtemp(prefix="exp010_configs_"))
    shutil.copytree(ROOT / "configs", old, dirs_exist_ok=True)
    (old / LIFE_FILE).unlink()
    real: dict[str, Any] = {}
    for ev in events:
        m = real_metrics(ROOT / "data" / "processed" / ev)
        stops = list(m.stops.values())
        counts = Counter(stops)
        n = len(stops) or 1
        real[ev] = {
            "stops": round(statistics.fmean(stops), 2),
            "share_1_2_3": [
                round(counts[1] / n, 2),
                round(counts[2] / n, 2),
                round(sum(v for k, v in counts.items() if k >= 3) / n, 2),
            ],
            "passes": m.overtakes,
            "retirements": m.retirements,
            "winner_time_min": round(m.winner_time_s / 60, 2),
            "order": m.order,
            "top_sequences": dict(Counter(m.sequences.values()).most_common(4)),
        }
    jobs = [
        (ev, variant, cfg, seed)
        for ev in events
        for variant, cfg in (("life", str(ROOT / "configs")), ("v0.9", str(old)))
        for seed in range(1, args.seeds + 1)
    ]
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, row in enumerate(pool.map(run, jobs, chunksize=2), start=1):
            rows.append(row)
            if i % 40 == 0 or i == len(jobs):
                el = time.perf_counter() - t0
                eta = el / i * (len(jobs) - i)
                print(f"{i}/{len(jobs)} races, {el / 60:.1f} min, ETA {eta / 60:.1f} min")
        lap_rows = [
            r for rs in pool.map(sim_laps, [(e, s) for e in events for s in (1, 2, 3)]) for r in rs
        ]
    shutil.rmtree(old, ignore_errors=True)
    # same-lap compound gaps and wear, simulated vs real
    real_laps = pd.read_parquet(DATASET)
    real_fit = fit_wear(green_laps(real_laps[real_laps["event_id"].isin(events)]))
    sim_fit = fit_wear(green_laps(pd.DataFrame(lap_rows)))
    same_lap = {
        "real": {"offsets": real_fit.offsets, "lin": real_fit.lin, "fresh": real_fit.fresh_bonus_s},
        "sim": {"offsets": sim_fit.offsets, "lin": sim_fit.lin, "fresh": sim_fit.fresh_bonus_s},
    }
    result = {
        "seeds": args.seeds,
        "train": summarise(rows, real, train),
        "test": summarise(rows, real, test),
        "same_lap": same_lap,
        "seconds": round(time.perf_counter() - t0, 1),
    }
    for part in ("train", "test"):
        r = result[part]
        print(part, {k: {kk: vv for kk, vv in v.items() if kk != "per_race"} for k, v in r.items()})
    print("same-lap", json.dumps(same_lap, default=float))
    if args.json:
        args.json.write_text(
            json.dumps(result, indent=1, default=float) + "\n", encoding="utf-8", newline="\n"
        )


if __name__ == "__main__":
    main()
