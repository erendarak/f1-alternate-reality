"""M5 slice 2 (D-039): the baselines B0-B3, imagined futures and paired evaluation."""

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from f1ar.domain.race_state import (
    Compound,
    PaceMode,
    PlannedStop,
    RaceSetup,
)
from f1ar.policy.baseline_strategy import fresh_stint_costs, wear_table
from f1ar.policy.strategists import (
    MIN_STINT,
    STAY,
    Enumerator,
    FixedPlan,
    MonteCarlo,
    RandomStrategist,
    candidate_plans,
    historical_plan,
)
from f1ar.simulation import lightweight_engine as engine
from f1ar.training.evaluate import evaluate
from f1ar.training.objectives import race_results, scoring_for
from f1ar.training.rollouts import imagined_seed, imagined_world, rollout_reward
from f1ar.training.session import RaceSession
from tests.conftest import HAM, VER


def _scoring(setup: RaceSetup, driver: str = VER):  # type: ignore[no-untyped-def]
    return scoring_for(setup, [driver], None)[driver]


# --- B2: the plan enumeration -------------------------------------------------------------


def test_candidates_are_legal_cheapest_first_and_cover_more_stops(
    isolated_setup: RaceSetup,
) -> None:
    plans = candidate_plans(isolated_setup, VER)
    costs = [c for c, _ in plans]
    assert costs == sorted(costs)
    assert {len(p) for _, p in plans} == {1, 2, 3}  # 0 stops breaks the two-compound rule
    start = isolated_setup.driver(VER).start_compound
    laps = isolated_setup.track.total_laps
    for _, plan in plans:
        compounds = {start, *(s.compound for s in plan)}
        assert len(compounds) >= 2
        bounds = [0, *(s.lap for s in plan), laps]
        assert all(b - a >= MIN_STINT for a, b in zip(bounds, bounds[1:], strict=False))


def test_best_split_matches_brute_force(isolated_setup: RaceSetup) -> None:
    setup = isolated_setup
    driver = setup.driver(VER).profile.archetype
    laps = setup.track.total_laps
    pit = setup.pit.lane_loss_s + setup.pit.standstill_median_s
    fresh = {
        c: fresh_stint_costs(
            setup.lap_model.compounds[c].offset_s,
            wear_table(setup, driver, c, PaceMode.BALANCED, laps),
        )
        for c in (Compound.MEDIUM, Compound.HARD)
    }
    start = setup.driver(VER).start_compound
    assert start is Compound.MEDIUM
    brute = min(
        float(
            fresh[Compound.MEDIUM][a] + fresh[Compound.HARD][b - a] + fresh[Compound.HARD][laps - b]
        )
        + 2 * pit
        for a, b in combinations(range(MIN_STINT, laps - MIN_STINT + 1), 2)
        if b - a >= MIN_STINT and laps - b >= MIN_STINT
    )
    found = next(
        c
        for c, p in candidate_plans(setup, VER)
        if [s.compound for s in p] == [Compound.HARD, Compound.HARD]
    )
    assert found == pytest.approx(brute)


# --- B0 and fixed plans ---------------------------------------------------------------------


def test_historical_plan_skips_red_flag_stops(tmp_path: Path) -> None:
    rows = []
    for lap in range(1, 11):
        rows.append(
            {
                "driver_code": "VER",
                "lap_number": lap,
                "compound": "MEDIUM" if lap <= 3 else ("HARD" if lap <= 7 else "SOFT"),
                "pit_in_time_s": {3: 300.0, 7: 700.0}.get(lap, np.nan),
            }
        )
    pd.DataFrame(rows).to_parquet(tmp_path / "laps.parquet")
    status = pd.DataFrame(
        {"session_time_s": [0.0, 690.0, 710.0], "status": ["1", "5", "1"]}
    )  # a red flag around the second pit entry
    status.to_parquet(tmp_path / "track_status.parquet")
    assert historical_plan(tmp_path, "VER") == (PlannedStop(lap=3, compound=Compound.HARD),)


def test_a_fixed_plan_is_carried_out(isolated_setup: RaceSetup) -> None:
    plan = (
        PlannedStop(lap=12, compound=Compound.HARD),
        PlannedStop(lap=30, compound=Compound.SOFT),
    )
    r = evaluate(isolated_setup, VER, 3, _scoring(isolated_setup), FixedPlan(plan))
    assert r.stops == 2 and r.sequence == "M-H-S" and not r.disqualified


# --- imagined futures -------------------------------------------------------------------------


def test_imagined_futures_never_use_the_real_seed_or_real_weather(
    isolated_world: object,
) -> None:
    from f1ar.domain.race_state import WorldState

    assert isinstance(isolated_world, WorldState)
    real = isolated_world.rng.seed
    seeds = {imagined_seed(real, VER, lap, k) for lap in range(1, 30) for k in range(3)}
    assert real not in seeds and len(seeds) == 29 * 3
    rainy = isolated_world.setup.weather.model_copy(
        update={"rain_timeline": ((0.0, False), (3000.0, True))}
    )
    world = isolated_world.model_copy(
        update={"setup": isolated_world.setup.model_copy(update={"weather": rainy})}
    )
    imagined = imagined_world(world, 99)
    assert imagined.rng.seed == 99
    assert imagined.setup.weather.wetness_timeline == ()
    # the real rain at 3000 s (beyond the 10-minute forecast) is in no imagined future
    horizon = world.setup.weather.forecast_horizon_s
    for k in range(200):
        timeline = imagined_world(world, 1000 + k).setup.weather.rain_timeline
        starts = [t for t, rain in timeline if rain]
        assert all(t <= world.sim_clock_s + horizon for t in starts)
        assert (3000.0, True) not in timeline


def test_rollouts_are_reproducible_and_leave_the_real_race_alone(
    isolated_setup: RaceSetup,
) -> None:
    session = RaceSession(
        isolated_setup, seed=5, agents=[VER], scoring={VER: _scoring(isolated_setup)}
    )
    d = session.next_decision()
    assert d is not None
    before = session.state
    a = rollout_reward(session.state, VER, STAY, 1234, session.scoring, session.actor_id)
    b = rollout_reward(session.state, VER, STAY, 1234, session.scoring, session.actor_id)
    assert a == b and np.isfinite(a)
    assert session.state is before  # nothing happened to the real race


# --- B3 and the harness ---------------------------------------------------------------------


def test_mpc_is_deterministic_and_deliberates(isolated_setup: RaceSetup) -> None:
    def run() -> tuple[int, str, int]:
        mpc = MonteCarlo(futures=1, min_tyre_age=15)
        r = evaluate(isolated_setup, VER, 2, _scoring(isolated_setup), mpc)
        return r.position, r.sequence, mpc.deliberations

    first = run()
    assert first == run()
    assert first[2] > 0


def test_paired_evaluation_runs_every_method_on_the_same_race(isolated_setup: RaceSetup) -> None:
    sc = _scoring(isolated_setup)
    rules = evaluate(isolated_setup, VER, 4, sc, None)
    assert rules.method == "rules" and rules.decisions == 0 and rules.stops >= 1
    # the other car isn't touched by the method used for VER (no traffic, no incidents)
    s1 = RaceSession(isolated_setup, seed=4, agents=[], scoring={VER: sc})
    while s1.next_decision() is not None:
        pass
    results = race_results(s1.state)
    assert results[VER].position == rules.position
    rnd = evaluate(isolated_setup, VER, 4, sc, RandomStrategist(1))
    assert rnd.method == "random" and rnd.decisions > 0
    assert engine.ENGINE_VERSION  # the harness runs the current engine


def test_enumerator_picks_a_simulated_best_plan(isolated_setup: RaceSetup) -> None:
    enum = Enumerator(shortlist=3, futures=1)
    r = evaluate(isolated_setup, HAM, 6, _scoring(isolated_setup, HAM), enum)
    assert len(enum.scores) == 3
    assert enum.plan == enum.scores[0][1]
    assert r.stops == len(enum.plan) and not r.disqualified


def test_imagined_rain_follows_the_forecast(isolated_world: object) -> None:
    from f1ar.domain.race_state import WorldState
    from f1ar.simulation.weather import forecast_rain_probability

    assert isinstance(isolated_world, WorldState)
    w = isolated_world
    forecast = forecast_rain_probability(w.setup.weather, w.rng.seed, w.sim_clock_s, False)
    wet = [
        any(rain for _, rain in imagined_world(w, 5000 + k).setup.weather.rain_timeline)
        for k in range(1000)
    ]
    assert sum(wet) / len(wet) == pytest.approx(forecast, abs=0.05)
