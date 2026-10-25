"""EXP-031 (part 3): how much is better pit timing worth when rivals react? (A1b, D-062)

For every rule car, the race is forked 3 laps before its planned first stop (lap ``P``) and that
stop is forced at ``P + d`` (``d`` in -2..+3; the car is handed to a USER for those laps, then
back to the rules after the stop; every branch uses the same mechanism, so ``d = 0`` is the
reference) on identical luck. Rivals keep their own rules (with ``--world a1b``: undercut and
cover, per-team knowledge).

- **hindsight**: mean over cars of the best branch minus ``d = 0``: an upper bound on what
  better first-stop timing is worth (it knows how the race turned out);
- **one simple rule**: the single offset that is best on average on half of the races (split by
  event), applied to the other half: a lower bound on what is learnable from nothing at all.

Worlds: ``a1`` (in-race learning, shared estimate, no rival rules) vs ``a1b`` (per-team
practice + private data, rival rules). Held-out 2018-2025 races.

Usage:
    uv run python scripts/experiments/exp031_decision_headroom.py --seeds 2 \\
        --json docs/experiments/exp031_decisions.json
"""

import argparse
import functools
import hashlib
import json
import random
import statistics
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

print = functools.partial(print, flush=True)  # noqa: A001 (progress through pipes)
ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "configs"
OFFSETS = (-2, -1, 0, 1, 2, 3)
LEAD = 3


def event_seed(event: str, k: int) -> int:
    return int(hashlib.sha256(event.encode()).hexdigest()[:6], 16) * 100 + k


def _world(event: str, seed: int, world: str) -> Any:
    from f1ar.simulation.race_day import with_hidden_truth, with_rival_rules
    from f1ar.simulation.scenarios import event_data_dir, load_event, setup_from_codes

    roster, profiles = load_event(CONFIGS, event)
    base = setup_from_codes(
        roster, profiles, None, data_dir=event_data_dir(ROOT, event), configs_dir=CONFIGS
    )
    setup = with_hidden_truth(
        base,
        CONFIGS,
        truth_seed=seed,
        learning=True,
        track_trend=True,
        per_team=world == "a1b",
    )
    return with_rival_rules(setup, CONFIGS) if world == "a1b" else setup


def _score(state: Any, driver_id: str) -> float:
    from f1ar.training.objectives import Objective, TeamOrder, objective_value, race_results

    results = race_results(state)
    return objective_value(
        TeamOrder(Objective.POINTS), driver_id, results, len(state.setup.drivers)
    )


def _branch(state: Any, driver_id: str, stop_lap: int, compound: Any) -> float:
    from f1ar.domain.race_state import (
        RequestPitStop,
        SetStrategyController,
        StrategyController,
        WorldStatus,
    )
    from f1ar.simulation import lightweight_engine as engine

    def car() -> Any:
        return s.car(driver_id)

    s = engine.apply_action(
        state, SetStrategyController(driver_id=driver_id, controller=StrategyController.USER)
    ).state
    requested = handed_back = False
    while s.world_status is WorldStatus.RUNNING:
        c = car()
        if not requested and c.laps_completed >= stop_lap - 1 and c.status.value == "RUNNING":
            s = engine.apply_action(s, RequestPitStop(driver_id=driver_id, compound=compound)).state
            requested = True
        if requested and not handed_back and c.pit_stops >= 1:
            s = engine.apply_action(
                s,
                SetStrategyController(driver_id=driver_id, controller=StrategyController.BASELINE),
            ).state
            handed_back = True
        s = engine.step(s).state
    return _score(s, driver_id)


def job(args: tuple[str, int, str]) -> dict[str, Any]:
    from f1ar.domain.race_state import CarStatus, StrategyController, WorldStatus
    from f1ar.simulation import lightweight_engine as engine

    event, seed, world = args
    setup = _world(event, seed, world)
    s = engine.start(engine.create_world(setup, seed=seed, world_id="exp031"))
    forks: dict[str, tuple[Any, int, Any]] = {}
    total = setup.track.total_laps
    while s.world_status is WorldStatus.RUNNING:
        s = engine.step(s).state
        for c in s.cars:
            if c.driver_id in forks or c.status is not CarStatus.RUNNING or c.pit_stops > 0:
                continue
            if c.controller is not StrategyController.BASELINE or not c.planned_stops:
                continue
            p = c.planned_stops[0]
            if c.laps_completed == p.lap - LEAD and p.lap + max(OFFSETS) < total:
                forks[c.driver_id] = (s, p.lap, p.compound)
    cars = []
    for driver_id, (state, lap, compound) in forks.items():
        scores = {d: _branch(state, driver_id, lap + d, compound) for d in OFFSETS}
        cars.append({"driver_id": driver_id, "planned_lap": lap, "scores": scores})
    return {"event": event, "seed": seed, "world": world, "cars": cars}


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    events = sorted({r["event"] for r in rows})
    cars = [(r["event"], c) for r in rows for c in r["cars"]]
    gain = {d: [c["scores"][d] - c["scores"][0] for _, c in cars] for d in OFFSETS}
    hind = [max(c["scores"].values()) - c["scores"][0] for _, c in cars]
    rng = random.Random(31)
    half = set(rng.sample(events, len(events) // 2)) or set(events)  # one race: no split
    fit = {
        d: statistics.fmean(c["scores"][d] - c["scores"][0] for e, c in cars if e in half)
        for d in OFFSETS
    }
    best = max(fit, key=lambda d: fit[d])
    held = [c["scores"][best] - c["scores"][0] for e, c in cars if e not in half]
    changed = sum(len(set(c["scores"].values())) > 1 for _, c in cars)
    return {
        "races": len(events),
        "cars": len(cars),
        "cars_where_timing_changed_the_score": changed,
        "hindsight_gain": round(statistics.fmean(hind), 4),
        "mean_gain_by_offset": {str(d): round(statistics.fmean(v), 4) for d, v in gain.items()},
        "simple_rule": {
            "offset": best,
            "fit_half_gain": round(fit[best], 4),
            "held_out_half_gain": round(statistics.fmean(held), 4) if held else 0.0,
        },
    }


def main() -> None:
    from f1ar.simulation.race_day import load_prior

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--workers", type=int, default=14)
    parser.add_argument("--from-season", type=int, default=2018)
    parser.add_argument("--worlds", nargs="+", default=["a1", "a1b"])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    prior = load_prior(CONFIGS)
    assert prior is not None
    events = sorted(
        e
        for e, v in prior["events"].items()
        if v["split"] == "dry_test" and v["season"] >= args.from_season
    )
    events = [e for e in events if (ROOT / "data" / "processed" / e).exists()]
    if args.limit:
        events = events[: args.limit]
    jobs = [
        (e, event_seed(e, k), w)
        for w in args.worlds
        for e in events
        for k in range(1, args.seeds + 1)
    ]
    print(f"EXP-031 decisions: {len(events)} races x {args.seeds} seeds x {args.worlds}")
    t0 = time.perf_counter()
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(job, j) for j in jobs]
        for i, f in enumerate(as_completed(futures), 1):
            rows.append(f.result())
            if i % 10 == 0 or i == len(jobs):
                el = time.perf_counter() - t0
                eta = el / i * (len(jobs) - i) / 60
                print(f"  {i}/{len(jobs)} jobs, {el / 60:.1f} min, ETA {eta:.1f} min")
    out: dict[str, Any] = {"seeds": args.seeds, "events": events, "worlds": {}}
    for w in args.worlds:
        out["worlds"][w] = summarise([r for r in rows if r["world"] == w])
        print(w, json.dumps(out["worlds"][w], indent=1))
    print(f"EXP-031 decisions done in {(time.perf_counter() - t0) / 60:.1f} min")
    if args.json:
        args.json.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
