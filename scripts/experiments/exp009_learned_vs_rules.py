"""EXP-009: does the first learned strategist beat the rule planner on held-out races? (M5 slice 3)

The policy (``--policy``, from the registry; hash-checked) plays one car vs the rule field on
the 5 dry *test* races, never seen in training, x 4 grid slots (P1/P5/P10/P15):

- **Set A, EXP-008's cases** (seeds 1-5, 100 cases): learned vs B1, recomputed and checked against
  EXP-008's stored B1 rows (same engine, same cases: must be identical), and vs EXP-008's B2/B3
  rows (the gap to MPC, without re-running it).
- **Set B, new seeds** (``--new-seeds``, default 6-15, 200 cases): learned vs B1.

Every case is paired (same setup, seed, other cars, scoring). Also ``learned+balanced``: the same
policy with pace forced to BALANCED (pit choices unchanged), which separates the gain from
pit timing / compounds from the gain through the editorial PUSH/CONSERVE effects.

Usage:
    uv run python scripts/experiments/exp009_learned_vs_rules.py --policy strategist_m5s3_v1 \\
        --workers 14 --json docs/experiments/exp009.json
"""

import argparse
import json
import statistics
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np

from f1ar.domain.race_state import PaceMode, RaceSetup
from f1ar.policy.strategists import LearnedStrategist
from f1ar.simulation.lightweight_engine import ENGINE_VERSION
from f1ar.training.actions import decode
from f1ar.training.evaluate import evaluate
from f1ar.training.objectives import scoring_for
from f1ar.training.policy_registry import NumpyPolicy, load_policy
from f1ar.training.regulations import REGULATIONS_2024
from f1ar.training.scenarios import repo_setup_factory
from f1ar.training.session import Decision, RaceSession

ROOT = Path(__file__).resolve().parents[2]
EVENTS = (
    "2024_italian_gp",
    "2024_japanese_gp",
    "2024_monaco_gp",
    "2024_mexico_city_gp",
    "2024_abu_dhabi_gp",
)
SLOTS = (1, 5, 10, 15)
EXP008 = ROOT / "docs" / "experiments" / "exp008.json"


@cache
def _setup(event_id: str) -> RaceSetup:
    return repo_setup_factory(ROOT)(event_id, False)


class Counting:
    """Wraps a strategist and counts the pace modes it chooses."""

    def __init__(self, inner: LearnedStrategist) -> None:
        self.inner = inner
        self.name = inner.name
        self.pace: Counter[str] = Counter()

    def start(self, session: RaceSession, driver_id: str) -> None:
        self.inner.start(session, driver_id)

    def decide(self, session: RaceSession, decision: Decision) -> int:
        action = self.inner.decide(session, decision)
        self.pace[decode(action)[1].value] += 1
        return action


def _policy(policy_id: str) -> NumpyPolicy:
    """A registered policy (hash-checked), or an unregistered ``.npz`` (diagnostics only, e.g.
    training checkpoints)."""
    if policy_id.endswith(".npz"):
        return NumpyPolicy.load(Path(policy_id))
    return load_policy(
        ROOT, policy_id, regulations=REGULATIONS_2024.version, engine_version=ENGINE_VERSION
    ).policy


def run_case(case: tuple[str, str, int, str, str]) -> dict[str, Any]:
    event_id, code, seed, method, policy_id = case
    setup = _setup(event_id)
    driver = next(d for d in setup.drivers if d.code == code)
    scoring = scoring_for(setup, [driver.driver_id], None)[driver.driver_id]
    s: Counting | None = None
    if method != "rules":
        policy = _policy(policy_id)
        pace = PaceMode.BALANCED if method == "learned+balanced" else None
        s = Counting(LearnedStrategist(policy, "learned", pace))
    row = asdict(evaluate(setup, driver.driver_id, seed, scoring, s))
    row["method"] = method
    row["code"] = code
    row["grid"] = driver.grid_position
    if s is not None:
        n = sum(s.pace.values())
        row["push_share"] = s.pace.get("PUSH", 0) / n if n else 0.0
        row["conserve_share"] = s.pace.get("CONSERVE", 0) / n if n else 0.0
    return row


def bootstrap(diffs: list[float], n: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    a = np.array(diffs)
    means = rng.choice(a, size=(n, len(a)), replace=True).mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def compare(rows: dict[tuple[Any, ...], dict[str, Any]], keys: list[Any], a: str, b: str) -> Any:
    out: dict[str, Any] = {}
    for metric in ("reward", "points", "position"):
        diffs = [rows[(*k, a)][metric] - rows[(*k, b)][metric] for k in keys]
        lo, hi = bootstrap(diffs)
        sign = -1 if metric == "position" else 1
        out[metric] = {
            "mean": round(statistics.fmean(diffs), 4),
            "ci95": [round(lo, 4), round(hi, 4)],
            "better": sum(sign * d > 1e-9 for d in diffs),
            "worse": sum(sign * d < -1e-9 for d in diffs),
        }
    return out


def summary(rows: list[dict[str, Any]], method: str) -> dict[str, Any]:
    mine = [r for r in rows if r["method"] == method]
    entry: dict[str, Any] = {
        "cases": len(mine),
        "reward": round(statistics.fmean(r["reward"] for r in mine), 4),
        "points": round(statistics.fmean(r["points"] for r in mine), 2),
        "position": round(statistics.fmean(r["position"] for r in mine), 2),
        "dsq": sum(r["disqualified"] for r in mine),
        "dnf": sum(not r["finished"] for r in mine),
        "stops": dict(sorted(Counter(r["stops"] for r in mine).items())),
        "sequences": dict(Counter(r["sequence"] for r in mine).most_common(6)),
    }
    if mine and "push_share" in mine[0]:
        entry["push_share"] = round(statistics.fmean(r["push_share"] for r in mine), 3)
        entry["conserve_share"] = round(statistics.fmean(r["conserve_share"] for r in mine), 3)
    return entry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--policy", required=True)
    parser.add_argument(
        "--new-seeds", type=int, nargs=2, default=[6, 15], metavar=("FIRST", "LAST")
    )
    parser.add_argument("--workers", type=int, default=14)
    parser.add_argument("--methods", nargs="+", default=["rules", "learned", "learned+balanced"])
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    t0 = time.perf_counter()
    if args.policy.endswith(".npz"):
        print(f"UNREGISTERED checkpoint {args.policy} (diagnostics only)", flush=True)
    else:
        loaded = load_policy(ROOT, args.policy, regulations=REGULATIONS_2024.version)
        print(f"policy {args.policy}: {loaded.manifest.artifact_sha256}", flush=True)

    methods: tuple[str, ...] = tuple(args.methods)
    seeds_a = list(range(1, 6))
    seeds_b = list(range(args.new_seeds[0], args.new_seeds[1] + 1))
    cases = []
    for event_id in EVENTS:
        by_grid = {d.grid_position: d.code for d in _setup(event_id).drivers}
        for slot in SLOTS:
            for seed in seeds_a + seeds_b:
                for m in methods:
                    cases.append((event_id, by_grid[slot], seed, m, args.policy))
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, row in enumerate(pool.map(run_case, cases, chunksize=2), start=1):
            rows.append(row)
            if i % 50 == 0 or i == len(cases):
                el = time.perf_counter() - t0
                eta = el / i * (len(cases) - i)
                print(
                    f"{i}/{len(cases)} races, {el / 60:.1f} min, ETA {eta / 60:.1f} min", flush=True
                )

    by = {(r["event_id"], r["code"], r["seed"], r["method"]): r for r in rows}
    result: dict[str, Any] = {"policy": args.policy, "sets": {}}
    for name, seeds in (("A_exp008_cases", seeds_a), ("B_new_seeds", seeds_b)):
        keys = sorted({(r["event_id"], r["code"], r["seed"]) for r in rows if r["seed"] in seeds})
        sub = [r for r in rows if r["seed"] in seeds]
        entry: dict[str, Any] = {m: summary(sub, m) for m in methods}
        entry["learned_vs_rules"] = compare(by, keys, "learned", "rules")
        if "learned+balanced" in methods:
            entry["learned+balanced_vs_rules"] = compare(by, keys, "learned+balanced", "rules")
            entry["learned_vs_learned+balanced"] = compare(by, keys, "learned", "learned+balanced")
        entry["per_race_reward"] = {
            e: {
                m: round(
                    statistics.fmean(
                        r["reward"] for r in sub if r["event_id"] == e and r["method"] == m
                    ),
                    4,
                )
                for m in methods
            }
            for e in EVENTS
        }
        result["sets"][name] = entry

    # set A against EXP-008's stored rows: B1 must match exactly; B2/B3 gaps
    if EXP008.exists():
        old = json.loads(EXP008.read_text(encoding="utf-8"))["rows"]
        old_by = {(r["event_id"], r["code"], r["seed"], r["method"]): r for r in old}
        keys = sorted({(r["event_id"], r["code"], r["seed"]) for r in rows if r["seed"] in seeds_a})
        mismatched = [
            k
            for k in keys
            if abs(old_by[(*k, "rules")]["reward"] - by[(*k, "rules")]["reward"]) > 1e-9
        ]
        merged = dict(by)
        for (e, c, s, m), r in old_by.items():
            if m in ("enumerator", "mpc"):
                merged[(e, c, s, m)] = r
        result["exp008_check"] = {
            "rules_rows_identical": not mismatched,
            "mismatched": [list(k) for k in mismatched],
            "learned_vs_enumerator": compare(merged, keys, "learned", "enumerator"),
            "learned_vs_mpc": compare(merged, keys, "learned", "mpc"),
        }
    result["seconds"] = round(time.perf_counter() - t0, 1)
    print(json.dumps(result, indent=1))
    if args.json:
        out = {**result, "rows": rows}
        args.json.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
