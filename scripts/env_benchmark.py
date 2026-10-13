"""How fast is the M5 training environment? (decisions per second, races per hour)

Random legal actions (the policy's own cost is excluded), single process:
- one AI car vs the rule field (``RaceStrategyEnv``, the slice 3 setting),
- all 20 cars AI (``run_race``, the slice 5 setting).

Usage:
    uv run python scripts/env_benchmark.py --races 5 --events 2024_italian_gp 2024_monaco_gp
"""

import argparse
import statistics
import time
from pathlib import Path

import numpy as np

from f1ar.training.gym_env import RaceStrategyEnv, random_policy, run_race
from f1ar.training.scenarios import ScenarioSampler, repo_setup_factory
from f1ar.training.session import RaceSession

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--races", type=int, default=5)
    parser.add_argument("--events", nargs="+", default=["2024_italian_gp", "2024_monaco_gp"])
    args = parser.parse_args()
    factory = repo_setup_factory(ROOT)
    for e in args.events:  # build setups before timing
        factory(e, False)

    env = RaceStrategyEnv(ScenarioSampler(args.events, factory))
    rng = np.random.default_rng(0)
    times, decisions = [], 0
    for i in range(args.races):
        t0 = time.perf_counter()
        env.reset(seed=i)
        done = False
        while not done:
            action = int(rng.choice(np.flatnonzero(env.action_masks())))
            _, _, done, _, _ = env.step(action)
            decisions += 1
        times.append(time.perf_counter() - t0)
    one = statistics.fmean(times)
    print(
        f"one AI car:  {one:.2f} s/race, {decisions / sum(times):.0f} decisions/s, "
        f"{3600 / one:.0f} races/hour/core"
    )

    sampler = ScenarioSampler(args.events, factory, agents="all")
    policy = random_policy(rng)
    times, decisions = [], 0
    for i in range(args.races):
        t0 = time.perf_counter()
        scenario, setup = sampler.sample(np.random.default_rng(i))
        session = RaceSession(
            setup,
            seed=scenario.seed,
            agents=scenario.agents,
            scoring=scenario.scoring,
            context=scenario.context,
        )
        run_race(session, policy)
        decisions += session.decisions
        times.append(time.perf_counter() - t0)
    field = statistics.fmean(times)
    print(
        f"20 AI cars:  {field:.2f} s/race, {decisions / sum(times):.0f} decisions/s, "
        f"{3600 / field:.0f} races/hour/core"
    )


if __name__ == "__main__":
    main()
