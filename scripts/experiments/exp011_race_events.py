"""EXP-011: race control and random events (W4, engine v0.11.0, D-042) vs the 2024 races.

For every dry 2024 race x ``--seeds`` the rule field is simulated twice on identical seeds: ``w4``
(the engine default) and ``v0.10`` (the same engine without the race-events file: W1 only).
Compared with the real race: retirements, safety cars, VSCs, red flags, time penalties, and the
W1 measures (stops per finisher, passes, finishing order, winner time), train (16) and test (5)
apart. The race-events values were fitted on all 24 races' data except ``risk_scale`` and
``red_flag_per_crash`` (fitted on the simulated dry_train field).

Seeds differ per race (keyed draws don't include the event, so seeds 1..N everywhere would
repeat the same luck on every track; D-042).

Usage:
    uv run python scripts/experiments/exp011_race_events.py --seeds 10 \\
        --json docs/experiments/exp011.json
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

import yaml

from f1ar.calibration.race_events_fit import EVENTS_FILE, event_counts
from f1ar.calibration.reference_metrics import real_metrics
from f1ar.calibration.simulated_metrics import sim_metrics, spearman

print = functools.partial(print, flush=True)  # noqa: A001 (progress through pipes)
ROOT = Path(__file__).resolve().parents[2]
SPLITS = ROOT / "data" / "manifests" / "dataset_laps_2024_v1.yaml"
MEASURES = ("stops", "retirements", "safety_cars", "vscs", "red_flags", "penalties", "passes")


def event_seed(event: str, k: int) -> int:
    return int(hashlib.sha256(event.encode()).hexdigest()[:6], 16) * 100 + k


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
    counts = m.extra.get("event_counts", {})
    stops = list(m.stops.values())
    return {
        "event": event,
        "variant": variant,
        "stops": statistics.fmean(stops) if stops else float("nan"),
        "retirements": m.retirements,
        "safety_cars": m.safety_cars,
        "vscs": m.vscs,
        "red_flags": counts.get("RED_FLAG", 0),
        "penalties": counts.get("TIME_PENALTY", 0),
        "passes": m.overtakes,
        "winner_time_min": m.winner_time_s / 60,
        "order": m.order,
    }


def real_race(event: str) -> dict[str, Any]:
    data = ROOT / "data" / "processed" / event
    m = real_metrics(data)
    c = event_counts(data, ROOT / "configs", event)
    stops = list(m.stops.values())
    return {
        "stops": statistics.fmean(stops),
        "retirements": m.retirements,
        "safety_cars": sum(1 for _, k, _ in c.neutralisations if k == "SC"),
        "vscs": sum(1 for _, k, _ in c.neutralisations if k == "VSC"),
        "red_flags": c.red_flags,
        "penalties": sum(c.penalties.values()),
        "passes": m.overtakes,
        "winner_time_min": m.winner_time_s / 60,
        "order": m.order,
    }


def summarise(rows: list[dict[str, Any]], real: dict[str, Any], events: list[str]) -> Any:
    out: dict[str, Any] = {
        "real": {k: round(statistics.fmean(real[e][k] for e in events), 3) for k in MEASURES}
    }
    for variant in ("w4", "v0.10"):
        mine = [r for r in rows if r["variant"] == variant and r["event"] in events]
        per_race = {
            e: {
                k: round(statistics.fmean(r[k] for r in mine if r["event"] == e), 3)
                for k in MEASURES
            }
            for e in events
        }
        entry: dict[str, Any] = {
            k: round(statistics.fmean(per_race[e][k] for e in events), 3) for k in MEASURES
        }
        entry["spearman"] = round(
            statistics.fmean(spearman(r["order"], real[r["event"]]["order"]) for r in mine), 3
        )
        entry["share_races_with_sc"] = round(
            statistics.fmean(1.0 if r["safety_cars"] > 0 else 0.0 for r in mine), 3
        )
        entry["share_races_with_sc_or_vsc"] = round(
            statistics.fmean(1.0 if r["safety_cars"] + r["vscs"] > 0 else 0.0 for r in mine), 3
        )
        entry["stops_mae"] = round(
            statistics.fmean(abs(per_race[e]["stops"] - real[e]["stops"]) for e in events), 3
        )
        entry["per_race"] = per_race
        out[variant] = entry
    out["real"]["share_races_with_sc"] = round(
        statistics.fmean(1.0 if real[e]["safety_cars"] > 0 else 0.0 for e in events), 3
    )
    out["real"]["share_races_with_sc_or_vsc"] = round(
        statistics.fmean(
            1.0 if real[e]["safety_cars"] + real[e]["vscs"] > 0 else 0.0 for e in events
        ),
        3,
    )
    out["real"]["per_race"] = {e: {k: real[e][k] for k in MEASURES} for e in events}
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
    old = Path(tempfile.mkdtemp(prefix="exp011_configs_"))
    shutil.copytree(ROOT / "configs", old, dirs_exist_ok=True)
    (old / EVENTS_FILE).unlink()
    real = {e: real_race(e) for e in events}
    jobs = [
        (e, variant, cfg, event_seed(e, k))
        for e in events
        for variant, cfg in (("w4", str(ROOT / "configs")), ("v0.10", str(old)))
        for k in range(1, args.seeds + 1)
    ]
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, row in enumerate(pool.map(run, jobs, chunksize=2), start=1):
            rows.append(row)
            if i % 40 == 0 or i == len(jobs):
                el = time.perf_counter() - t0
                eta = el / i * (len(jobs) - i)
                print(f"{i}/{len(jobs)} races, {el / 60:.1f} min, ETA {eta / 60:.1f} min")
    shutil.rmtree(old, ignore_errors=True)
    result = {
        "seeds": args.seeds,
        "train": summarise(rows, real, train),
        "test": summarise(rows, real, test),
        "all_dry": summarise(rows, real, events),
        "seconds": round(time.perf_counter() - t0, 1),
    }
    for part in ("train", "test", "all_dry"):
        r = result[part]
        print(part, {k: {kk: vv for kk, vv in v.items() if kk != "per_race"} for k, v in r.items()})
    if args.json:
        text = json.dumps(result, indent=1, default=float) + "\n"
        args.json.write_text(text, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
