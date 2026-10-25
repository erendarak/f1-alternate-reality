"""EXP-027: what knowing the race-day truth is worth to a rule team (H3 gate, D-058).

Held-out races 2000-2025 (each season's dry_test races with a race-day prior), ``--seeds``
per-race seeds, two truths:

- ``historical``: the race as it was (its own tyre-life fit); the teams know the prior;
- ``drawn``: a new truth drawn around the prior with the era's measured surprise (truth seed =
  the race seed).

Per race and seed: the **prior world** (every car's rule planner on the prior), then for each car
the same race with only that car an **oracle** (planning on the truth), on identical luck
(keyed RNG). The car's paired gain, oracle - prior, is the value of information. Also the **truth
world** (every car an oracle = v0.18's world) and the real race's stops per finisher.

Scores: ``score`` = the M5 POINTS objective (points / 25 + 0.02 x place, as EXP-016..019);
``score_season`` = points / the season's points for a win + 0.02 x place (the 2000-2009 point
scales are smaller); points; finishing position. CIs: bootstrap over races (95 %). The
surprise tertiles split races by the *historical* surprise (in ``drawn`` mode they don't
describe that mode's own surprise).

Gate (``docs/PLAN_HIDDEN_INFO_ERAS.md`` §6): a mean gain under ~0.02 score per race means the
information isn't worth enough to build H4-H5 on.

Usage:
    uv run python scripts/experiments/exp027_value_of_information.py --seeds 3 \\
        --json docs/experiments/exp027.json
"""

import argparse
import functools
import hashlib
import json
import random
import statistics
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from f1ar.calibration.reference_metrics import real_metrics

print = functools.partial(print, flush=True)  # noqa: A001 (progress through pipes)
ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "configs"
MODES = ("historical", "drawn")
METRICS = ("score", "score_season", "points", "position")


def event_seed(event: str, k: int) -> int:
    return int(hashlib.sha256(event.encode()).hexdigest()[:6], 16) * 100 + k


@functools.cache
def _base(event: str) -> Any:
    from f1ar.simulation.scenarios import event_data_dir, load_event, setup_from_codes

    roster, profiles = load_event(CONFIGS, event)
    return setup_from_codes(
        roster, profiles, None, data_dir=event_data_dir(ROOT, event), configs_dir=CONFIGS
    )


def _run(setup: Any, seed: int) -> Any:
    from f1ar.domain.race_state import WorldStatus
    from f1ar.simulation import lightweight_engine as engine

    s = engine.start(engine.create_world(setup, seed=seed, world_id="exp027"))
    while s.world_status is WorldStatus.RUNNING:
        s = engine.step(s).state
    return s


def _scores(state: Any) -> dict[str, dict[str, float]]:
    from f1ar.domain.race_state import CarStatus
    from f1ar.training.objectives import (
        POSITION_TIEBREAK,
        Objective,
        TeamOrder,
        objective_value,
        race_results,
    )

    results = race_results(state)
    n = len(state.setup.drivers)
    win = state.setup.rules.points_for(1, fastest_lap=False, final_round=False)
    order = TeamOrder(Objective.POINTS)
    out = {}
    for d, r in results.items():
        place = (n - r.position) / max(1, n - 1)
        car = state.car(d)
        out[d] = {
            "score": objective_value(order, d, results, n),
            "score_season": r.points / win + POSITION_TIEBREAK * place,
            "points": float(r.points),
            "position": float(r.position),
            "stops": float(car.pit_stops),
            "finished": float(car.status is CarStatus.FINISHED),
        }
    return out


def _stops_per_finisher(state: Any) -> float:
    from f1ar.domain.race_state import CarStatus

    stops = [c.pit_stops for c in state.cars if c.status is CarStatus.FINISHED]
    return statistics.fmean(stops) if stops else float("nan")


def job(args: tuple[str, str, int]) -> dict[str, Any]:
    from f1ar.simulation.race_day import TruthMode, with_hidden_truth, with_oracle

    event, mode, seed = args
    hidden = with_hidden_truth(_base(event), CONFIGS, mode=TruthMode(mode), truth_seed=seed)
    drivers = tuple(d.driver_id for d in hidden.drivers)
    prior_state = _run(hidden, seed)
    truth_state = _run(with_oracle(hidden, drivers), seed)
    prior = _scores(prior_state)
    cars = []
    for d in drivers:
        oracle = _scores(_run(with_oracle(hidden, (d,)), seed))[d]
        cars.append(
            {
                "driver_id": d,
                "grid": hidden.driver(d).grid_position,
                "prior": prior[d],
                "oracle": oracle,
                "gain": {k: oracle[k] - prior[d][k] for k in METRICS},
            }
        )
    return {
        "event": event,
        "mode": mode,
        "seed": seed,
        "note": hidden.inputs["race_day_truth"],
        "stops_prior_world": _stops_per_finisher(prior_state),
        "stops_truth_world": _stops_per_finisher(truth_state),
        "cars": cars,
    }


def bootstrap_ci(per_race: dict[str, float], n: int = 2000) -> tuple[float, float]:
    keys = sorted(per_race)
    rng = random.Random(27)
    means = sorted(statistics.fmean(per_race[rng.choice(keys)] for _ in keys) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n) - 1]


def summarise(rows: list[dict[str, Any]], prior_cfg: dict[str, Any]) -> dict[str, Any]:
    """Mean gain per car-race (each race weighted equally), CIs over races, breakdowns."""
    per_race: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    changed = better = worse = 0
    for r in rows:
        for c in r["cars"]:
            for k in METRICS:
                per_race[r["event"]][k].append(c["gain"][k])
            g = c["gain"]["score"]
            changed += c["oracle"]["stops"] != c["prior"]["stops"] or g != 0.0
            better += g > 1e-12
            worse += g < -1e-12
    cases = sum(len(r["cars"]) for r in rows)

    def block(events: list[str]) -> dict[str, Any]:
        out: dict[str, Any] = {"races": len(events)}
        for k in METRICS:
            means = {e: statistics.fmean(per_race[e][k]) for e in events}
            lo, hi = bootstrap_ci(means) if len(events) > 1 else (float("nan"),) * 2
            out[k] = {
                "mean": round(statistics.fmean(means.values()), 4),
                "ci95": [round(lo, 4), round(hi, 4)],
            }
        return out

    events = sorted(per_race)
    ev = prior_cfg["events"]
    by_era = defaultdict(list)
    for e in events:
        by_era[ev[e]["era"]].append(e)
    size = sorted(events, key=lambda e: abs(ev[e]["surprise"]["life_shared_log"]))
    third = len(size) // 3
    surprise = {
        "small": size[:third],
        "middle": size[third : 2 * third],
        "large": size[2 * third :],
    }
    world = defaultdict(list)
    for r in rows:
        world[r["event"]].append(r)
    return {
        "all": block(events),
        "eras": {k: block(v) for k, v in sorted(by_era.items())},
        "life_surprise_tertiles": {
            k: {
                **block(v),
                "abs_life_log_range": [
                    round(abs(ev[v[0]]["surprise"]["life_shared_log"]), 3),
                    round(abs(ev[v[-1]]["surprise"]["life_shared_log"]), 3),
                ],
            }
            for k, v in surprise.items()
        },
        "cases": cases,
        "oracle_better": better,
        "oracle_worse": worse,
        "oracle_changed_something": changed,
        "stops_prior_world": round(
            statistics.fmean(
                statistics.fmean(x["stops_prior_world"] for x in v) for v in world.values()
            ),
            3,
        ),
        "stops_truth_world": round(
            statistics.fmean(
                statistics.fmean(x["stops_truth_world"] for x in v) for v in world.values()
            ),
            3,
        ),
    }


def per_race_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Per race and truth mode: mean gains over cars and seeds, the two worlds' stops, and the
    largest single-car gain / loss (the per-car rows are too large to keep)."""
    out: dict[str, Any] = {}
    for r in sorted(rows, key=lambda r: (r["event"], r["mode"], r["seed"])):
        race = out.setdefault(r["event"], {})
        if r["mode"] == "historical":
            race["note"] = r["note"]
        race.setdefault(r["mode"], []).append(r)
    for race in out.values():
        for m in MODES:
            if m not in race:
                continue
            mine = race[m]
            gains = [c["gain"] for r in mine for c in r["cars"]]
            race[m] = {
                **{k: round(statistics.fmean(g[k] for g in gains), 4) for k in METRICS},
                "best_score_gain": round(max(g["score"] for g in gains), 3),
                "worst_score_gain": round(min(g["score"] for g in gains), 3),
                "stops_prior_world": round(
                    statistics.fmean(r["stops_prior_world"] for r in mine), 2
                ),
                "stops_truth_world": round(
                    statistics.fmean(r["stops_truth_world"] for r in mine), 2
                ),
            }
    return out


def real_stops(event: str) -> float:
    m = real_metrics(ROOT / "data" / "processed" / event)
    stops = [m.stops.get(d, 0) for d in m.finishers]
    return statistics.fmean(stops) if stops else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--workers", type=int, default=14)
    parser.add_argument("--modes", nargs="+", default=list(MODES), choices=MODES)
    parser.add_argument("--limit", type=int, help="first N races only (smoke test)")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    from f1ar.simulation.race_day import load_prior

    prior_cfg = load_prior(CONFIGS)
    assert prior_cfg is not None
    events = sorted(e for e, v in prior_cfg["events"].items() if v["split"] == "dry_test")
    events = [e for e in events if (ROOT / "data" / "processed" / e).exists()]
    if args.limit:
        events = events[: args.limit]
    jobs = [
        (e, m, event_seed(e, k))
        for m in args.modes
        for e in events
        for k in range(1, args.seeds + 1)
    ]
    print(f"EXP-027: {len(events)} held-out races x {args.seeds} seeds x {args.modes}")
    print(f"{len(jobs)} jobs")
    t0 = time.perf_counter()
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(job, j) for j in jobs]
        for i, f in enumerate(as_completed(futures), 1):
            rows.append(f.result())
            if i % 20 == 0 or i == len(jobs):
                el = time.perf_counter() - t0
                eta = el / i * (len(jobs) - i) / 60
                print(f"  {i}/{len(jobs)} jobs, {el / 60:.1f} min, ETA {eta:.1f} min")
    out: dict[str, Any] = {"seeds": args.seeds, "events": events, "modes": {}}
    for m in args.modes:
        mine = [r for r in rows if r["mode"] == m]
        out["modes"][m] = summarise(mine, prior_cfg)
    if "historical" in args.modes:
        real = {e: real_stops(e) for e in events}
        out["real_stops_per_finisher"] = round(statistics.fmean(real.values()), 3)
        hist = [r for r in rows if r["mode"] == "historical"]
        for world in ("prior", "truth"):
            errs = [
                abs(
                    statistics.fmean(r[f"stops_{world}_world"] for r in hist if r["event"] == e)
                    - real[e]
                )
                for e in events
            ]
            out[f"stops_mae_{world}_world"] = round(statistics.fmean(errs), 3)
    out["races"] = per_race_summary(rows)
    for m in args.modes:
        a = out["modes"][m]["all"]
        print(
            f"{m}: gain score {a['score']['mean']:+.4f} {a['score']['ci95']}, season-scaled "
            f"{a['score_season']['mean']:+.4f}, points {a['points']['mean']:+.3f}, position "
            f"{a['position']['mean']:+.3f}"
        )
        print(json.dumps({k: v for k, v in out["modes"][m].items() if k != "all"}, indent=1))
    for k in ("real_stops_per_finisher", "stops_mae_prior_world", "stops_mae_truth_world"):
        if k in out:
            print(k, out[k])
    print(f"EXP-027 done in {(time.perf_counter() - t0) / 60:.1f} min")
    if args.json:
        args.json.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
