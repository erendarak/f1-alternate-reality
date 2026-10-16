"""Collect counterfactual racecraft targets on identical luck (M5 slice 4b, D-049).

Training scenarios as in slice 4 (the 16 dry_train races; championship context in 80 % of races,
the learning car's risk appetite redrawn per race, tyre randomisation 0.25), one AI car driven
by the rules teacher (pit/pace) with NORMAL racecraft. At sampled decisions (``--battle-share``
of the decisions with a car within 1.5 s ahead or behind, ``--clear-share`` of the others) the
race is forked and DEFEND / NORMAL / ATTACK are each held for ``--laps`` laps on the same luck
(``training.counterfactual``). Writes chunks ``<out>/part_<n>.npz`` (observation rl_obs_v3,
the two advantages vs NORMAL, the three rewards, metadata); a rerun skips finished chunks.

Usage:
    uv run python scripts/collect_counterfactuals.py --races 3000 --out artifacts/training/crn_v1
"""

import argparse
import functools
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from f1ar.domain.race_state import RacePhase, RaceSetup
from f1ar.policy.baseline_strategy import plan_for_car
from f1ar.policy.strategists import RulesTeacher
from f1ar.simulation.lightweight_engine import ENGINE_VERSION
from f1ar.training.counterfactual import _teacher_normal, decision_targets, pit_targets
from f1ar.training.features import RL_OBSERVATION_SCHEMA, _gaps
from f1ar.training.objectives import ChampionshipContext, Objective, Scoring, TeamOrder
from f1ar.training.scenarios import ScenarioSampler, repo_setup_factory
from f1ar.training.session import RaceSession

print = functools.partial(print, flush=True)  # noqa: A001 (progress through pipes)
ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "data" / "manifests" / "dataset_laps_2024_v1.yaml"
BATTLE_S = 1.5
PIT_WINDOW = 4


@functools.cache
def _sampler() -> ScenarioSampler:
    events = yaml.safe_load(SPLITS.read_text(encoding="utf-8"))["splits"]["dry_train"]
    return ScenarioSampler(
        tuple(events),
        repo_setup_factory(ROOT),
        agents=1,
        championship_probability=0.8,
        tyre_randomisation=0.25,
        character_randomisation=True,
    )


def title_fight(
    setup: RaceSetup, me: str, rng: np.random.Generator
) -> tuple[dict[str, Scoring], ChampionshipContext]:
    """A title fight against the car next to ``me`` on the grid (editorial scenario, as
    EXP-018's chaser/leader): -30..+30 points, 1-8 races left."""
    grid = sorted(setup.drivers, key=lambda d: d.grid_position)
    i = next(k for k, d in enumerate(grid) if d.driver_id == me)
    rival = grid[i - 1 if i > 0 else 1].driver_id
    points = {d.driver_id: 20.0 for d in setup.drivers}
    points[rival] = 200.0
    points[me] = 200.0 + float(rng.uniform(-30.0, 30.0))
    context = ChampionshipContext(points, races_left=int(rng.integers(1, 9)))
    order = TeamOrder(
        Objective.CHAMPIONSHIP,
        rival=rival,
        rival_weight=1.0,
        reason="collector",
        gap_before=points[me] - points[rival],
        races_after=context.races_left - 1,
    )
    risk = setup.driver(me).profile.archetype.risk_appetite
    return {me: Scoring(order, risk)}, context


def race(job: tuple[int, float, float, int, float, str]) -> dict[str, Any]:
    race_seed, battle_share, clear_share, laps, title_share, kind = job
    rng = np.random.default_rng(race_seed)
    scenario, setup = _sampler().sample(rng)
    me = scenario.agents[0]
    scoring, context = scenario.scoring, scenario.context
    if rng.random() < title_share:
        scoring, context = title_fight(setup, me, rng)
    session = RaceSession(
        setup,
        seed=scenario.seed,
        agents=[me],
        scoring=scoring,
        context=context,
        policy_id="crn_collector",
    )
    choose = _teacher_normal(RulesTeacher())
    keys = ("obs", "adv", "rewards", "battle", "lap", "rules_boxes", "neutral")
    rows: dict[str, list[Any]] = {k: [] for k in keys}
    while (d := session.next_decision()) is not None:
        ahead, behind = _gaps(session.state, me)
        battle = (ahead is not None and ahead < BATTLE_S) or (
            behind is not None and behind < BATTLE_S
        )
        neutral = session.state.race_control.phase is not RacePhase.GREEN
        if kind == "pit":
            car = session.state.car(me)
            plan = plan_for_car(setup, car)
            near = bool(plan) and abs(plan[0].lap - car.laps_completed - 1) <= PIT_WINDOW
            if near or neutral:
                p = pit_targets(session, d)
                if p is not None:
                    rows["obs"].append(d.observation)
                    rows["adv"].append([p.advantage])
                    rows["rewards"].append([p.rules, p.alternative])
                    rows["battle"].append(battle)
                    rows["lap"].append(d.lap)
                    rows["rules_boxes"].append(p.rules_boxes)
                    rows["neutral"].append(neutral)
        elif rng.random() < (battle_share if battle else clear_share):
            t = decision_targets(session, d, laps=laps)
            if t is not None:
                rows["obs"].append(d.observation)
                rows["adv"].append(t.advantages)
                rows["rewards"].append(t.rewards)
                rows["battle"].append(battle)
                rows["lap"].append(d.lap)
                rows["rules_boxes"].append(False)
                rows["neutral"].append(neutral)
        session.act(me, choose(session, d))
    order = scoring[me].order
    meta = {
        "event": scenario.event_id,
        "risk_appetite": setup.driver(me).profile.archetype.risk_appetite,
        "objective": order.objective.value,
        "grid": setup.driver(me).grid_position,
    }
    return {**rows, "meta": meta}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--races", type=int, default=3000)
    parser.add_argument("--chunk", type=int, default=140)
    parser.add_argument("--workers", type=int, default=14)
    parser.add_argument("--laps", type=int, default=5)
    parser.add_argument("--battle-share", type=float, default=0.35)
    parser.add_argument("--clear-share", type=float, default=0.04)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--kind",
        choices=("racecraft", "pit"),
        default="racecraft",
        help="racecraft: DEFEND/NORMAL/ATTACK held --laps laps; pit: the rules' pit choice vs "
        f"the one-lap alternative, within {PIT_WINDOW} laps of the planned stop or under SC/VSC",
    )
    parser.add_argument(
        "--title-share",
        type=float,
        default=0.0,
        help="share of races turned into a title fight vs the grid neighbour",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "config.json").write_text(
        json.dumps(
            {
                **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "engine": ENGINE_VERSION,
                "schema": RL_OBSERVATION_SCHEMA,
                "battle_s": BATTLE_S,
            },
            indent=1,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    t0 = time.perf_counter()
    done_races = 0
    n_chunks = (args.races + args.chunk - 1) // args.chunk
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for c in range(n_chunks):
            path = args.out / f"part_{c:03d}.npz"
            if path.exists():
                continue
            seeds = range(
                args.seed * 10_000_000 + c * args.chunk,
                args.seed * 10_000_000 + min((c + 1) * args.chunk, args.races),
            )
            jobs = [
                (s, args.battle_share, args.clear_share, args.laps, args.title_share, args.kind)
                for s in seeds
            ]
            out = list(pool.map(race, jobs, chunksize=2))
            rows = [r for r in out if r["obs"]]
            n = [len(r["obs"]) for r in rows]
            np.savez_compressed(
                path,
                obs=np.concatenate([np.array(r["obs"], dtype=np.float32) for r in rows]),
                adv=np.concatenate([np.array(r["adv"]) for r in rows]),
                rewards=np.concatenate([np.array(r["rewards"]) for r in rows]),
                battle=np.concatenate([np.array(r["battle"]) for r in rows]),
                lap=np.concatenate([np.array(r["lap"]) for r in rows]),
                rules_boxes=np.concatenate([np.array(r["rules_boxes"]) for r in rows]),
                neutral=np.concatenate([np.array(r["neutral"]) for r in rows]),
                race=np.concatenate([np.full(k, c * args.chunk + i) for i, k in enumerate(n)]),
                risk=np.concatenate(
                    [np.full(k, r["meta"]["risk_appetite"]) for r, k in zip(rows, n, strict=True)]
                ),
                objective=np.concatenate(
                    [np.full(k, r["meta"]["objective"]) for r, k in zip(rows, n, strict=True)]
                ),
                event=np.concatenate(
                    [np.full(k, r["meta"]["event"]) for r, k in zip(rows, n, strict=True)]
                ),
            )
            done_races += len(jobs)
            elapsed = time.perf_counter() - t0
            left = elapsed / done_races * (args.races - (c + 1) * args.chunk)
            adv = np.concatenate([np.array(r["adv"]) for r in rows])
            print(
                f"chunk {c + 1}/{n_chunks}: {sum(n)} decisions, "
                f"changes {np.mean(np.abs(adv[:, -1]) > 1e-9):.2f}, "
                f"{elapsed / 60:.1f} min, ~{max(0.0, left) / 60:.0f} min left"
            )


if __name__ == "__main__":
    main()
