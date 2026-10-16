"""EXP-018: do goals and character change what the learned strategist does, for learned reasons,
and does it beat the rules at its own goal? (M5 slice 4, D-048)

Cases: the 5 held-out dry races x grid P1/P5/P10/P15 x ``--seeds`` (seeds 1..N). Each case is
played under five scenarios (same race, same seed: paired):

- ``assigned``: synthetic standings drawn per case (seeded), the team principal's objective and
  the driver's persona risk appetite (what the AI meets in a world);
- ``chaser`` / ``leader``: CHAMPIONSHIP vs the car next to it on the grid (the title rival),
  15 points behind / ahead with 3 races left; persona risk appetite;
- ``careful`` / ``daring``: POINTS with risk appetite 0.1 / 0.9.

Methods: ``rules`` (B1: always NORMAL), ``learned`` (the policy's most likely action),
``learned_sampled`` (actions drawn from the policy, seeded per race: still reproducible), and on
``assigned`` also the hand-written ``racecraft_rule``. Every method is scored under the
scenario's own objective and risk utility. Logged per AI decision: the intent chosen, whether a
car is within 1 s ahead / behind and whether that car is the title rival.

Usage:
    uv run python scripts/experiments/exp018_goals.py --policy strategist_m5s4b_v1 --seeds 6 \\
        --json docs/experiments/exp018.json
"""

import argparse
import functools
import json
import statistics
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from f1ar.domain.race_state import RaceSetup
from f1ar.policy.strategists import LearnedStrategist, RacecraftRule, Strategist
from f1ar.simulation.lightweight_engine import ENGINE_VERSION
from f1ar.training.actions import intent_of
from f1ar.training.evaluate import evaluate
from f1ar.training.features import RL_OBSERVATION_SCHEMA, _gaps, _neighbours
from f1ar.training.objectives import (
    ChampionshipContext,
    Objective,
    Scoring,
    TeamOrder,
    scoring_for,
)
from f1ar.training.policy_registry import NumpyPolicy, load_policy
from f1ar.training.regulations import REGULATIONS_2024
from f1ar.training.scenarios import repo_setup_factory, synthetic_standings
from f1ar.training.session import Decision, RaceSession

print = functools.partial(print, flush=True)  # noqa: A001 (progress through pipes)
ROOT = Path(__file__).resolve().parents[2]
EVENTS = (
    "2024_italian_gp",
    "2024_japanese_gp",
    "2024_monaco_gp",
    "2024_mexico_city_gp",
    "2024_abu_dhabi_gp",
)
SCENARIOS = ("assigned", "chaser", "leader", "careful", "daring")
CLOSE_S = 1.0


@functools.cache
def _setup(event_id: str) -> RaceSetup:
    return repo_setup_factory(ROOT)(event_id, False)


@functools.cache
def _policy(policy_id: str) -> NumpyPolicy:
    if policy_id.endswith(".npz"):
        return NumpyPolicy.load(Path(policy_id))
    return load_policy(
        ROOT, policy_id, regulations=REGULATIONS_2024.version, engine_version=ENGINE_VERSION
    ).policy


def with_risk(setup: RaceSetup, driver_id: str, appetite: float) -> RaceSetup:
    drivers = []
    for d in setup.drivers:
        if d.driver_id == driver_id:
            arche = d.profile.archetype.model_copy(update={"risk_appetite": appetite})
            d = d.model_copy(update={"profile": d.profile.model_copy(update={"archetype": arche})})
        drivers.append(d)
    return setup.model_copy(update={"drivers": tuple(drivers)})


def scenario(
    setup: RaceSetup, driver_id: str, seed: int, name: str
) -> tuple[RaceSetup, Scoring, ChampionshipContext | None, str | None]:
    """(setup, scoring, context, rival) for one scenario of a case."""
    grid = sorted(setup.drivers, key=lambda d: d.grid_position)
    i = next(k for k, d in enumerate(grid) if d.driver_id == driver_id)
    neighbour = grid[i - 1 if i > 0 else 1].driver_id
    risk = setup.driver(driver_id).profile.archetype.risk_appetite
    if name == "assigned":
        rng = np.random.default_rng([seed, i, 18])
        context = synthetic_standings(setup, rng, int(rng.integers(4, 21)))
        scoring = scoring_for(setup, [driver_id], context)[driver_id]
        return setup, scoring, context, scoring.order.rival
    if name in ("chaser", "leader"):
        base = {d.driver_id: 20.0 for d in setup.drivers}
        lead = 15.0 if name == "leader" else -15.0
        base[driver_id], base[neighbour] = 200.0 + lead, 200.0
        context = ChampionshipContext(base, races_left=3)
        order = TeamOrder(
            Objective.CHAMPIONSHIP, rival=neighbour, rival_weight=1.0, reason=f"EXP-018 {name}"
        )
        return setup, Scoring(order, risk), context, neighbour
    appetite = 0.1 if name == "careful" else 0.9
    s = with_risk(setup, driver_id, appetite)
    return s, Scoring(TeamOrder(Objective.POINTS), appetite), None, None


class Logged:
    """Wraps a strategist and logs the racecraft situation of each decision."""

    def __init__(self, inner: Strategist, rival: str | None) -> None:
        self.inner = inner
        self.name = inner.name
        self.observation_schema = getattr(inner, "observation_schema", RL_OBSERVATION_SCHEMA)
        self.rival = rival
        self.log: Counter[str] = Counter()

    def start(self, session: RaceSession, driver_id: str) -> None:
        self.inner.start(session, driver_id)

    def decide(self, session: RaceSession, decision: Decision) -> int:
        action = self.inner.decide(session, decision)
        intent = intent_of(action).value.lower()
        state = session.state
        ahead_id, behind_id = _neighbours(state, decision.driver_id)
        ahead_s, behind_s = _gaps(state, decision.driver_id)
        close_ahead = ahead_s is not None and ahead_s < CLOSE_S
        close_behind = behind_s is not None and behind_s < CLOSE_S
        situation = (
            "rival_ahead"
            if close_ahead and ahead_id == self.rival
            else "rival_behind"
            if close_behind and behind_id == self.rival
            else "car_ahead"
            if close_ahead
            else "car_behind"
            if close_behind
            else "clear"
        )
        self.log[f"{situation}:{intent}"] += 1
        self.log[f"all:{intent}"] += 1
        return action


def run(job: tuple[str, int, int, str, str, str]) -> dict[str, Any]:
    event_id, slot, seed, name, method, policy_id = job
    base = _setup(event_id)
    driver_id = next(d.driver_id for d in base.drivers if d.grid_position == slot)
    setup, scoring, context, rival = scenario(base, driver_id, seed, name)
    inner: Strategist | None = None
    if method == "learned":
        inner = LearnedStrategist(_policy(policy_id), "learned")
    elif method == "learned_sampled":  # draws from the policy (seeded by case: reproducible)
        inner = LearnedStrategist(_policy(policy_id), "learned_sampled", sample_seed=18)
    elif method == "racecraft_rule":
        inner = RacecraftRule()
    s = Logged(inner, rival) if inner is not None else None
    row = asdict(evaluate(setup, driver_id, seed, scoring, s, context=context))
    row.update(
        scenario=name,
        method=method,
        slot=slot,
        objective=scoring.order.objective.value,
        risk_appetite=scoring.risk_appetite,
        log=dict(s.log) if s is not None else {},
    )
    return row


def bootstrap(diffs: list[float], n: int = 2000) -> list[float]:
    a = np.array(diffs)
    means = np.random.default_rng(0).choice(a, size=(n, len(a)), replace=True).mean(axis=1)
    return [round(float(np.quantile(means, 0.025)), 4), round(float(np.quantile(means, 0.975)), 4)]


def share(logs: list[dict[str, int]], situation: str, intent: str) -> float | None:
    total = sum(v for log in logs for k, v in log.items() if k.startswith(f"{situation}:"))
    if total == 0:
        return None
    return round(sum(log.get(f"{situation}:{intent}", 0) for log in logs) / total, 4)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--policy", required=True)
    parser.add_argument("--seeds", type=int, default=6)
    parser.add_argument("--workers", type=int, default=14)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--events", nargs="+", default=list(EVENTS))
    args = parser.parse_args()
    t0 = time.perf_counter()
    jobs = []
    for e in args.events:
        for slot in (1, 5, 10, 15):
            for seed in range(1, args.seeds + 1):
                for name in SCENARIOS:
                    methods = ["rules", "learned", "learned_sampled"] + (
                        ["racecraft_rule"] if name == "assigned" else []
                    )
                    jobs += [(e, slot, seed, name, m, args.policy) for m in methods]
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, row in enumerate(pool.map(run, jobs, chunksize=2), 1):
            rows.append(row)
            if i % 100 == 0 or i == len(jobs):
                print(f"{i}/{len(jobs)} races, {(time.perf_counter() - t0) / 60:.1f} min")
    by = {(r["event_id"], r["slot"], r["seed"], r["scenario"], r["method"]): r for r in rows}
    result: dict[str, Any] = {"policy": args.policy, "engine": ENGINE_VERSION, "scenarios": {}}
    for name in SCENARIOS:
        keys = sorted(
            {(r["event_id"], r["slot"], r["seed"]) for r in rows if r["scenario"] == name}
        )
        entry: dict[str, Any] = {"cases": len(keys)}
        methods = sorted({r["method"] for r in rows if r["scenario"] == name})
        for m in methods:
            mine = [by[(*k, name, m)] for k in keys]
            entry[m] = {
                "reward": round(statistics.fmean(r["reward"] for r in mine), 4),
                "points": round(statistics.fmean(r["points"] for r in mine), 2),
                "position": round(statistics.fmean(r["position"] for r in mine), 2),
                "position_p90": float(np.quantile([r["position"] for r in mine], 0.9)),
                "dnf": sum(not r["finished"] for r in mine),
                "stops": dict(sorted(Counter(r["stops"] for r in mine).items())),
                "objectives": dict(Counter(r["objective"] for r in mine)),
            }
            if m != "rules":
                diffs = [
                    by[(*k, name, m)]["reward"] - by[(*k, name, "rules")]["reward"] for k in keys
                ]
                entry[m]["vs_rules"] = {
                    "mean": round(statistics.fmean(diffs), 4),
                    "ci95": bootstrap(diffs),
                    "better": sum(d > 1e-9 for d in diffs),
                    "worse": sum(d < -1e-9 for d in diffs),
                }
                logs = [r["log"] for r in mine]
                entry[m]["intent"] = {
                    situation: {i: share(logs, situation, i) for i in ("attack", "defend")}
                    for situation in (
                        "all",
                        "rival_ahead",
                        "rival_behind",
                        "car_ahead",
                        "car_behind",
                        "clear",
                    )
                }
        result["scenarios"][name] = entry
        print(name, json.dumps(entry)[:1500])
    result["seconds"] = round(time.perf_counter() - t0, 1)
    if args.json:
        args.json.write_text(
            json.dumps({**result, "rows": rows}, indent=1) + "\n", encoding="utf-8", newline="\n"
        )


if __name__ == "__main__":
    main()
