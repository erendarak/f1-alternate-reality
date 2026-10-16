"""EXP-020: undercuts and out-laps, real 2024 vs simulation (W7, D-051).

The same undercut detector (``calibration/undercut``) on the 21 dry 2024 races and on simulated
races of the same tracks (rule field, per-race seeds), plus the out-lap and the first laps on
new tyres relative to the car's laps around them. ``--configs`` compares another configs
directory (e.g. without the W7 file).

Usage:
    uv run python scripts/experiments/exp020_undercut.py --seeds 6 \
        --json docs/experiments/exp020.json
"""

import argparse
import functools
import hashlib
import json
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from f1ar.calibration.undercut import Attempt, real_laps, summary, undercut_attempts
from f1ar.domain.race_state import RacePhase, WorldStatus
from f1ar.simulation import lightweight_engine as engine
from f1ar.simulation.scenarios import event_data_dir, load_event, setup_from_codes

print = functools.partial(print, flush=True)  # noqa: A001 (progress through pipes)
ROOT = Path(__file__).resolve().parents[2]
SPLITS = ROOT / "data" / "manifests" / "dataset_laps_2024_v1.yaml"
DATASET = ROOT / "data" / "processed" / "datasets" / "laps_2024_v1.parquet"
OPTIONAL = ("calibration/out_laps_2024_v1.json",)


def event_seed(event: str, k: int) -> int:
    return int(hashlib.sha256(event.encode()).hexdigest()[:6], 16) * 100 + k


@functools.cache
def _setup(event: str, configs: str) -> Any:
    roster, profiles = load_event(ROOT / "configs", event)
    return setup_from_codes(
        roster, profiles, None, data_dir=event_data_dir(ROOT, event), configs_dir=Path(configs)
    )


def sim_laps(job: tuple[str, str, int]) -> pd.DataFrame:
    event, configs, seed = job
    setup = _setup(event, configs)
    code = {d.driver_id: d.code for d in setup.drivers}
    state = engine.start(engine.create_world(setup, seed=seed, world_id="exp020"))
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    neutral_laps: set[int] = set()
    while state.world_status is WorldStatus.RUNNING:
        before = state
        result = engine.step(state)
        state = result.state
        if state.race_control.phase is not RacePhase.GREEN:
            neutral_laps.add(max(c.laps_completed for c in state.cars) + 1)
        for e in result.events:
            if e.driver_id is None:
                continue
            c = code[e.driver_id]
            if e.type == "LAP_COMPLETED":
                lap = int(e.payload["lap"])
                rows[(c, lap)] = {
                    "driver": c,
                    "lap": lap,
                    "end_s": float(e.payload["crossing_time_s"]),
                    "pit_in": False,
                    "green": before.race_control.phase is RacePhase.GREEN
                    and state.race_control.phase is RacePhase.GREEN,
                }
            elif e.type == "PIT_STOP":
                key = (c, int(e.payload["lap"]))
                if key in rows:
                    rows[key]["pit_in"] = True
    df = pd.DataFrame(list(rows.values()))
    df["green"] = df["green"] & ~df["lap"].isin(neutral_laps)
    return df.assign(event=event)


def attempts_of(df: pd.DataFrame) -> list[Attempt]:
    return undercut_attempts(df)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, default=6)
    parser.add_argument("--workers", type=int, default=14)
    parser.add_argument("--without-w7", action="store_true", help="also run without the W7 file")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    splits = yaml.safe_load(SPLITS.read_text(encoding="utf-8"))["splits"]
    events = list(splits["dry_train"]) + list(splits["dry_test"])
    laps = pd.read_parquet(DATASET)
    real = {e: undercut_attempts(real_laps(laps[laps["event_id"] == e])) for e in events}
    result: dict[str, Any] = {
        "real": summary([a for v in real.values() for a in v]),
        "real_test": summary([a for e in splits["dry_test"] for a in real[e]]),
    }
    print("real", json.dumps(result["real"]))
    variants = {"current": str(ROOT / "configs")}
    tmp = None
    if args.without_w7:
        tmp = Path(tempfile.mkdtemp(prefix="exp020_configs_"))
        shutil.copytree(ROOT / "configs", tmp, dirs_exist_ok=True)
        for f in OPTIONAL:
            (tmp / f).unlink(missing_ok=True)
        variants["without_w7"] = str(tmp)
    for name, configs in variants.items():
        jobs = [(e, configs, event_seed(e, k)) for e in events for k in range(1, args.seeds + 1)]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            frames = list(pool.map(sim_laps, jobs, chunksize=2))
        att = [a for f in frames for a in undercut_attempts(f)]
        test = [
            a
            for f in frames
            if f["event"].iloc[0] in splits["dry_test"]
            for a in undercut_attempts(f)
        ]
        result[name] = {
            "all": {**summary(att), "attempts_per_race": round(len(att) / len(frames), 2)},
            "test": summary(test),
        }
        print(name, json.dumps(result[name]))
    if tmp is not None:
        shutil.rmtree(tmp, ignore_errors=True)
    result["real"]["attempts_per_race"] = round(float(result["real"]["attempts"]) / len(events), 2)
    if args.json:
        args.json.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
