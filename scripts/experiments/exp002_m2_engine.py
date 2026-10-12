"""EXP-002: how close is the full M2 engine to real races? Monza 2024 (in-sample) and Spa 2024
(out-of-sample), real vs a distribution of simulated races.

For each event: real metrics from the fetched timing data, simulated metrics from N seeds of the
full engine (pit stops + baseline strategy, traffic, incidents, VSC/SC) with the event's
data-derived inputs (start tyres, car offsets, base lap, pit loss). Every simulated metric is
reported as mean ± sd and p10-p90 next to the real value.

Overtakes are counted the SAME way for both: positions = order of line crossings at the end of
each lap (lap 0 = grid); a pass is one car getting ahead of another between consecutive lap
ends; pairs where either car pitted at the end of either lap are excluded (stops, not passes).

Usage:
    uv run python scripts/experiments/exp002_m2_engine.py --seeds 30
    uv run python scripts/experiments/exp002_m2_engine.py --seeds 30 \\
        --json docs/experiments/exp002.json
"""

import argparse
import json
import statistics
import time
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from f1ar.calibration.reference_metrics import (
    RaceMetrics,
    count_overtakes,
    real_metrics,
    sequence_of,
)
from f1ar.domain.race_state import CarStatus, RaceSetup, WorldState, WorldStatus
from f1ar.simulation import lightweight_engine as engine
from f1ar.simulation.scenarios import (
    BELGIUM_2024,
    MONZA_2024,
    event_data_dir,
    load_event,
    setup_from_codes,
)

ROOT = Path(__file__).resolve().parents[2]


# --- a simulated race --------------------------------------------------------------------------


def sim_metrics(setup: RaceSetup, seed: int) -> RaceMetrics:
    state: WorldState = engine.start(engine.create_world(setup, seed=seed, world_id="exp002"))
    code = {d.driver_id: d.code for d in setup.drivers}
    crossings: dict[int, dict[str, float]] = {}
    stopped: dict[str, set[int]] = {}
    compounds: dict[str, list[str]] = {
        code[d.driver_id]: [d.start_compound.value] for d in setup.drivers
    }
    counts: Counter[str] = Counter()
    while state.world_status is WorldStatus.RUNNING:
        result = engine.step(state)
        state = result.state
        for e in result.events:
            counts[e.type] += 1
            if e.driver_id is None:
                continue
            c = code[e.driver_id]
            if e.type == "LAP_COMPLETED":
                crossings.setdefault(int(e.payload["lap"]), {})[c] = float(
                    e.payload["crossing_time_s"]
                )
            elif e.type == "PIT_STOP":
                stopped.setdefault(c, set()).add(int(e.payload["lap"]))
                compounds[c].append(str(e.payload["to_compound"]))
    order = [code[d] for d in engine.standings(state)]
    finished = {code[c.driver_id] for c in state.cars if c.status is CarStatus.FINISHED}
    winner_car = state.car(engine.standings(state)[0])
    grid = [d.code for d in sorted(setup.drivers, key=lambda d: d.grid_position)]
    return RaceMetrics(
        order=order,
        finishers=finished,
        winner=order[0],
        winner_time_s=winner_car.finish_time_s or 0.0,
        stops={code[c.driver_id]: c.pit_stops for c in state.cars if code[c.driver_id] in finished},
        sequences={d: sequence_of(compounds[d]) for d in finished},
        overtakes=count_overtakes(crossings, grid, stopped),
        retirements=sum(1 for c in state.cars if c.status is CarStatus.RETIRED),
        safety_cars=state.race_control.sc_count,
        vscs=state.race_control.vsc_count,
        lapped=sum(
            1
            for c in state.cars
            if c.status is CarStatus.FINISHED and c.laps_completed < winner_car.laps_completed
        ),
    )


# --- comparison --------------------------------------------------------------------------------


def spearman(a: list[str], b: list[str]) -> float:
    common = [d for d in a if d in b]
    rank_b = {d: i for i, d in enumerate(x for x in b if x in common)}
    n = len(common)
    d2 = sum((i - rank_b[d]) ** 2 for i, d in enumerate(common))
    return 1 - 6 * d2 / (n * (n * n - 1))


def position_error(sim: list[str], real: list[str], both: set[str]) -> float:
    s = [d for d in sim if d in both]
    r = [d for d in real if d in both]
    return statistics.fmean(abs(s.index(d) - r.index(d)) for d in both)


def describe(values: Sequence[float]) -> dict[str, float]:
    ordered = sorted(values)

    def pick(q: float) -> float:
        return ordered[min(len(ordered) - 1, int(q * len(ordered)))]

    return {
        "mean": statistics.fmean(values),
        "sd": statistics.pstdev(values),
        "p10": pick(0.1),
        "p90": pick(0.9),
    }


def run_event(event_id: str, seeds: int) -> dict[str, Any]:
    data_dir = event_data_dir(ROOT, event_id)
    if data_dir is None:
        raise SystemExit(f"fetch {event_id} first: scripts/fetch_event.py")
    roster, profiles = load_event(ROOT / "configs", event_id)
    setup = setup_from_codes(roster, profiles, None, data_dir=data_dir)
    real = real_metrics(data_dir)
    real_finish_order = [d for d in real.order if d in real.finishers]

    t0 = time.perf_counter()
    sims = [sim_metrics(setup, seed) for seed in range(1, seeds + 1)]
    elapsed = time.perf_counter() - t0

    rows: dict[str, tuple[float, list[float]]] = {
        "winner race time (min)": (real.winner_time_s / 60, [s.winner_time_s / 60 for s in sims]),
        "stops per finisher": (
            statistics.fmean(real.stops.values()),
            [statistics.fmean(s.stops.values()) for s in sims],
        ),
        "share of 1-stop finishers": (
            sum(v == 1 for v in real.stops.values()) / len(real.stops),
            [sum(v == 1 for v in s.stops.values()) / len(s.stops) for s in sims],
        ),
        "on-track overtakes": (real.overtakes, [s.overtakes for s in sims]),
        "retirements": (real.retirements, [s.retirements for s in sims]),
        "safety cars": (real.safety_cars, [s.safety_cars for s in sims]),
        "VSCs": (real.vscs, [s.vscs for s in sims]),
        "lapped finishers": (real.lapped, [s.lapped for s in sims]),
    }
    metrics = {}
    for name, (real_value, values) in rows.items():
        d = describe(values)
        metrics[name] = {
            "real": real_value,
            **d,
            "real_in_p10_p90": d["p10"] <= real_value <= d["p90"],
        }

    order_quality = []
    for s in sims:
        both = s.finishers & real.finishers
        sim_finish_order = [d for d in s.order if d in s.finishers]
        order_quality.append(
            (
                spearman(sim_finish_order, real_finish_order),
                position_error(s.order, real.order, both),
            )
        )
    team_of = {e.code: e.team_name for e in roster.entries}
    real_podium = real.order[:3]
    return {
        "event": event_id,
        "seeds": seeds,
        "engine": engine.ENGINE_VERSION,
        "seconds_per_race": elapsed / seeds,
        "inputs": dict(setup.inputs),
        "real": {
            "winner": real.winner,
            "podium": real_podium,
            "sequences": Counter(real.sequences.values()).most_common(5),
        },
        "metrics": metrics,
        "order": {
            "spearman": describe([q[0] for q in order_quality]),
            "position_error": describe([q[1] for q in order_quality]),
        },
        "winners": Counter(s.winner for s in sims).most_common(),
        "winner_teams": Counter(team_of[s.winner] for s in sims).most_common(),
        "real_winner_wins": sum(s.winner == real.winner for s in sims),
        "real_podium_exact": sum(s.order[:3] == real_podium for s in sims),
        "sim_sequences": Counter(seq for s in sims for seq in s.sequences.values()).most_common(5),
    }


def _row(name: str, real: str, d: dict[str, float], fmt: str, flag: str = "") -> str:
    cells = "".join(
        f"{d[k]:>{8 if k != 'mean' else 10}{fmt}}" for k in ("mean", "sd", "p10", "p90")
    )
    return f"{name:<28}{real:>8}{cells}  {flag}"


def print_report(r: dict[str, Any]) -> None:
    head = f"{r['event']} | {r['seeds']} seeds | {r['engine']} | {r['seconds_per_race']:.2f} s/race"
    print(f"\n=== {head}")
    print(f"real winner {r['real']['winner']}, podium {r['real']['podium']}")
    print(f"{'metric':<28}{'real':>8}{'sim mean':>10}{'+/- sd':>8}{'p10':>8}{'p90':>8}  in range")
    for name, m in r["metrics"].items():
        flag = "yes" if m["real_in_p10_p90"] else "NO"
        print(_row(name, f"{m['real']:.2f}", m, ".2f", flag))
    print(_row("Spearman vs real order", "", r["order"]["spearman"], ".3f"))
    print(_row("mean |position error|", "", r["order"]["position_error"], ".2f"))
    print(f"sim winners: {r['winners']}")
    seeds = r["seeds"]
    print(
        f"real winner won {r['real_winner_wins']}/{seeds}; "
        f"exact real podium {r['real_podium_exact']}/{seeds}"
    )
    print(f"tyre sequences  real: {r['real']['sequences']}")
    print(f"                sim:  {r['sim_sequences']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="EXP-002: M2 engine vs real races")
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--events", nargs="+", default=[MONZA_2024, BELGIUM_2024])
    parser.add_argument("--json", type=Path, help="write the full results here")
    args = parser.parse_args()
    reports = [run_event(e, args.seeds) for e in args.events]
    for r in reports:
        print_report(r)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(reports, indent=2, default=str) + "\n", encoding="utf-8", newline="\n"
        )
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
