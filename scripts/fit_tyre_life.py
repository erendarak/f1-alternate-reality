"""Fit the W1 tyre-life model on a season (engine v0.10.0, D-041) and write
configs/calibration/tyre_life_<season>_v1.json (2024 by default; H2a: any season 2018+).

1. Per race: raw tyre life per compound (stint lengths), wear multiplier; season same-lap wear
   curve and fresh-tyre peak (``f1ar.calibration.tyre_life_fit``).
2. The globals ``kappa`` (life scale) and ``delta`` (potential pace step between compounds) by
   grid search on the dry_train races only: the rule field is simulated and its stops per
   finisher and compound shares of laps are compared with the real race (squared errors).
   The dry_test races are never used here (EXP-010 checks them).

Usage:
    uv run python scripts/fit_tyre_life.py                      # data + grid search (~15-30 min)
    uv run python scripts/fit_tyre_life.py --kappa 1.0 --delta 0.35 --no-grid
    uv run python scripts/fit_tyre_life.py --season 2019        # H2a
"""

import argparse
import functools
import hashlib
import json
import statistics
import time
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from f1ar.calibration.lap_model_fit import LIFE_FILE
from f1ar.calibration.season_files import dataset_files, season_path
from f1ar.calibration.tyre_life_fit import fit_events, to_json

print = functools.partial(print, flush=True)  # noqa: A001 (progress through pipes)
ROOT = Path(__file__).resolve().parents[1]
KAPPAS = (0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.45)
"""0.6 and 0.7 added in H2b: four 2010-2017 seasons hit 0.8, the old edge."""
DELTAS = (0.0, 0.1, 0.2, 0.35, 0.5, 0.65)
SEEDS = (1, 2, 3)
FIRST_COMPOUND_SEASON = 2018
SHARE_WEIGHT = 4.0
"""Weight of the compound-share error (shares are 0-1; stops ~1-3)."""


def real_targets(laps: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Per race: stops per finisher (within a lap of the distance) and compound shares of
    green racing laps."""
    out = {}
    for ev, r in laps.groupby("event_id"):
        fin = r.groupby("driver_code").agg(
            stops=("pit_in_lap", "sum"),
            laps=("lap_number", "max"),
            sched=("scheduled_laps", "first"),
        )
        fin = fin[fin["laps"] >= fin["sched"] - 1]
        g = r[
            (r["track_status_start"].astype(str) == "1")
            & r["compound"].isin(["SOFT", "MEDIUM", "HARD"])
        ]
        shares = g["compound"].value_counts(normalize=True).to_dict()
        out[str(ev)] = {
            "stops": float(fin["stops"].mean()),
            "shares": {c: float(shares.get(c, 0.0)) for c in ("SOFT", "MEDIUM", "HARD")},
        }
    return out


def simulate(job: tuple[str, float, float, tuple[int, ...]]) -> dict[str, Any]:
    """The rule field on ``event`` with ``kappa``/``delta``: stops per finisher and compound
    shares of laps, averaged over ``seeds``."""
    from f1ar.domain.race_state import CarStatus, WorldStatus
    from f1ar.simulation import lightweight_engine as engine
    from f1ar.simulation.scenarios import event_data_dir, load_event, setup_from_codes

    event, kappa, delta, seeds = job
    roster, profiles = load_event(ROOT / "configs", event)
    setup = setup_from_codes(
        roster,
        profiles,
        None,
        data_dir=event_data_dir(ROOT, event),
        configs_dir=ROOT / "configs",
        tyre_life_overrides={"kappa": kappa, "delta": delta},
    )
    total = setup.track.total_laps
    stops, laps_on = [], {"SOFT": 0, "MEDIUM": 0, "HARD": 0}
    modes = {"PUSH": 0, "CONSERVE": 0, "BALANCED": 0}
    for seed in seeds:
        state = engine.start(engine.create_world(setup, seed=seed, world_id=f"fit_{event}"))
        changes: dict[str, list[tuple[int, str]]] = {}
        while state.world_status is WorldStatus.RUNNING:
            res = engine.step(state)
            state = res.state
            for e in res.events:
                if e.type == "PIT_STOP" and e.driver_id is not None:
                    changes.setdefault(e.driver_id, []).append(
                        (int(e.payload["lap"]), str(e.payload["to_compound"]))
                    )
                elif e.type == "LAP_COMPLETED" and e.driver_id is not None:
                    modes[state.car(e.driver_id).lap_pace_mode.value] += 1
        for d in setup.drivers:
            car = state.car(d.driver_id)
            if car.status is CarStatus.FINISHED and car.laps_completed >= total - 1:
                stops.append(car.pit_stops)
            cur, lap0 = d.start_compound.value, 0
            for lap, comp in [*changes.get(d.driver_id, []), (car.laps_completed, "")]:
                if cur in laps_on:
                    laps_on[cur] += max(0, lap - lap0)
                cur, lap0 = comp, lap
    n = sum(laps_on.values()) or 1
    m = sum(modes.values()) or 1
    return {
        "event": event,
        "kappa": kappa,
        "delta": delta,
        "stops": statistics.fmean(stops) if stops else float("nan"),
        "shares": {c: v / n for c, v in laps_on.items()},
        "modes": {k: round(v / m, 3) for k, v in modes.items()},
    }


def content_sha(splits_file: Path) -> str:
    manifest = yaml.safe_load(splits_file.read_text(encoding="utf-8"))
    return str(manifest["file"]["content_sha256"])


def write_manifest(
    artifact: Path,
    *,
    season: int,
    splits_file: Path,
    splits_sha: str,
    train: list[str],
    best: tuple[float, float],
    fit: dict[str, Any],
) -> None:
    """The model registry entry (configs/models/tyre_life_<season>_v1.yaml)."""
    body = {
        "model_name": "tyre_life",
        "model_version": "v1",
        "kind": "stint-length life + same-lap wear regression; globals fitted on behaviour",
        "source_type": "estimated_from_data",
        "artifact": artifact.relative_to(ROOT).as_posix(),
        "artifact_sha256": "sha256:" + hashlib.sha256(artifact.read_bytes()).hexdigest(),
        "feature_schema_version": 1,
        "training_dataset_manifest": splits_file.relative_to(ROOT).as_posix(),
        "training_dataset_content_sha256": splits_sha,
        "training_races": sorted(train),
        "metrics": {
            "dry_train_fit": {
                "kappa": best[0],
                "delta_s": best[1],
                "loss": round(float(fit["loss"]), 4),
                "stops_sim": round(float(fit["stops_sim"]), 3),
                "stops_real": round(float(fit["stops_real"]), 3),
                "share_sim": fit["share_sim"],
            },
            "held_out_check": "EXP-010 (docs/experiments/exp010.json)",
        },
        "status": "validated_for_engine_use (W1, engine v0.10.0, D-041)"
        if season == 2024
        else f"fitted for {season} (H2a, D-054); held-out check EXP-023",
    }
    (ROOT / "configs" / "models" / f"tyre_life_{season}_v1.yaml").write_text(
        yaml.safe_dump(body, sort_keys=False, width=100), encoding="utf-8", newline="\n"
    )


def loss(sim: dict[str, Any], real: dict[str, Any], share_weight: float = SHARE_WEIGHT) -> float:
    share = sum((sim["shares"][c] - real["shares"][c]) ** 2 for c in real["shares"])
    return float((sim["stops"] - real["stops"]) ** 2 + share_weight * share)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--kappa", type=float, default=1.0)
    parser.add_argument("--delta", type=float, default=0.35)
    parser.add_argument("--no-grid", action="store_true")
    parser.add_argument("--workers", type=int, default=14)
    parser.add_argument("--report", type=Path, help="write the grid results as JSON")
    parser.add_argument("--season", type=int, default=2024)
    parser.add_argument(
        "--reference-season",
        type=int,
        help="compound ratios from this season's fit (default: 2018 for seasons before 2018, "
        "which have no compound data; H2b)",
    )
    args = parser.parse_args()
    ref_season = args.reference_season or (FIRST_COMPOUND_SEASON if args.season < 2018 else None)
    reference = None
    if ref_season is not None:
        ref_file = ROOT / "configs" / season_path(LIFE_FILE, ref_season)
        reference = json.loads(ref_file.read_text(encoding="utf-8"))
    # compound shares are meaningless with the MEDIUM placeholder: fit stops only
    share_weight = 0.0 if reference is not None else SHARE_WEIGHT
    t0 = time.perf_counter()
    dataset, splits_file = dataset_files(ROOT, args.season)
    laps = pd.read_parquet(dataset)
    splits = yaml.safe_load(splits_file.read_text(encoding="utf-8"))["splits"]
    train, test = list(splits["dry_train"]), list(splits["dry_test"])
    season, ratios, events = fit_events(laps, train + test, train, reference)
    sha = "sha256:" + hashlib.sha256(dataset.read_bytes()).hexdigest()
    out_path = ROOT / "configs" / season_path(LIFE_FILE, args.season)

    def write(kappa: float, delta: float) -> None:
        cfg = to_json(
            season,
            ratios,
            events,
            train,
            kappa=kappa,
            delta=delta,
            data_sha=sha,
            year=args.season,
            reference=None
            if ref_season is None
            else f"compounds unknown: soft/hard wear and life ratios to the medium from "
            f"{season_path(LIFE_FILE, ref_season).as_posix()} (borrowed); stops-only fit",
        )
        out_path.write_text(json.dumps(cfg, indent=1) + "\n", encoding="utf-8", newline="\n")

    write(args.kappa, args.delta)
    print(
        f"wrote {out_path.relative_to(ROOT)} (kappa {args.kappa}, delta {args.delta})", flush=True
    )
    if args.no_grid:
        return
    targets = real_targets(laps)
    jobs = [(ev, k, d, SEEDS) for k, d in product(KAPPAS, DELTAS) for ev in train]
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, row in enumerate(pool.map(simulate, jobs, chunksize=1), start=1):
            rows.append(row)
            if i % 16 == 0 or i == len(jobs):
                el = time.perf_counter() - t0
                eta = el / i * (len(jobs) - i)
                print(
                    f"{i}/{len(jobs)} race sets, {el / 60:.1f} min, ETA {eta / 60:.1f} min",
                    flush=True,
                )
    grid: dict[tuple[float, float], dict[str, Any]] = {}
    for k, d in product(KAPPAS, DELTAS):
        mine = [r for r in rows if r["kappa"] == k and r["delta"] == d]
        grid[(k, d)] = {
            "loss": statistics.fmean(loss(r, targets[r["event"]], share_weight) for r in mine),
            "stops_sim": statistics.fmean(r["stops"] for r in mine),
            "stops_real": statistics.fmean(targets[r["event"]]["stops"] for r in mine),
            "share_sim": {
                c: round(statistics.fmean(r["shares"][c] for r in mine), 3)
                for c in ("SOFT", "MEDIUM", "HARD")
            },
            "modes": {
                m: round(statistics.fmean(r["modes"][m] for r in mine), 3)
                for m in ("PUSH", "BALANCED", "CONSERVE")
            },
        }
        print(f"kappa {k:.2f} delta {d:.2f}: {grid[(k, d)]}", flush=True)
    best = min(grid, key=lambda kd: float(grid[kd]["loss"]))
    print(f"best: kappa {best[0]}, delta {best[1]}: {grid[best]}", flush=True)
    write(*best)
    write_manifest(
        out_path,
        season=args.season,
        splits_file=splits_file,
        splits_sha=content_sha(splits_file),
        train=train,
        best=best,
        fit=grid[best],
    )
    if args.report:
        report = {
            "real_share_train": {
                c: round(statistics.fmean(targets[e]["shares"][c] for e in train), 3)
                for c in ("SOFT", "MEDIUM", "HARD")
            },
            "grid": [{"kappa": k, "delta": d, **v} for (k, d), v in grid.items()],
            "best": {"kappa": best[0], "delta": best[1]},
            "per_race_best": [
                {**r, "real": targets[r["event"]]} for r in rows if (r["kappa"], r["delta"]) == best
            ],
            "seconds": round(time.perf_counter() - t0, 1),
        }
        args.report.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"done in {(time.perf_counter() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
