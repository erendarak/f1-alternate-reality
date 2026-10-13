"""Overtaking and race control fitted on real races, and the launch spread (M4 slice 5, D-035)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from f1ar.calibration.pace_models import TyreShape
from f1ar.calibration.race_control_fit import (
    RaceCounts,
    fit_race_control,
    load_race_control,
    race_control_params,
)
from f1ar.calibration.simulated_metrics import sim_metrics
from f1ar.calibration.traffic_fit import (
    OvertakeFit,
    encounters,
    fit_overtaking,
    load_traffic,
    pass_probability,
    pass_probability_with_noise,
    traffic_params,
)
from f1ar.domain.driver_profiles import ProfileSet
from f1ar.domain.identities import RaceRoster
from f1ar.domain.provenance import SourceType
from f1ar.domain.race_state import RaceSetup
from f1ar.simulation import lightweight_engine as engine
from f1ar.simulation.rng_streams import normal
from f1ar.simulation.scenarios import (
    build_setup,
    default_race_control_params,
    default_traffic_params,
)
from tests.conftest import CONFIGS

# --- the pass curve ------------------------------------------------------------------------


def test_pass_curve_is_zero_below_the_threshold_and_rises() -> None:
    p = pass_probability(np.array([-1.0, 0.2, 0.5, 1.0, 3.0]), 0.2, 0.8)
    assert p[0] == p[1] == 0.0
    assert 0.0 < p[2] < p[3] < p[4] < 1.0
    assert p[3] == pytest.approx(1 - np.exp(-0.8 / 0.8))


def test_noise_convolution_matches_monte_carlo() -> None:
    margin, sd = np.array([0.0, 0.4, 1.2]), np.array([0.3, 0.3, 0.5])
    exact = pass_probability_with_noise(margin, sd, 0.2, 0.8)
    rng = np.random.default_rng(1)
    z = rng.standard_normal((200_000, 1))
    mc = pass_probability(margin + sd * z, 0.2, 0.8).mean(axis=0)
    assert exact == pytest.approx(mc, abs=0.005)
    assert pass_probability_with_noise(margin, np.zeros(3), 0.2, 0.8) == pytest.approx(
        pass_probability(margin, 0.2, 0.8)
    )


def _synthetic_encounters(n: int = 6000, seed: int = 3) -> pd.DataFrame:
    """Encounters drawn from the engine's own model: track ``easy`` threshold 0.0, ``hard``
    0.6, scale 0.9, min gap 0.4 (held cars end 0.4 s behind), engine noise sd 0.25."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        track = "easy" if i % 2 else "hard"
        thr = 0.0 if track == "easy" else 0.6
        adv = rng.uniform(-1.5, 2.5)
        noisy = adv + 0.4 + 0.25 * rng.standard_normal()
        passed = bool(rng.uniform() < pass_probability(np.array([noisy]), thr, 0.9)[0])
        rows.append(
            {
                "event_id": track,
                "lap": i % 50 + 2,
                "advantage_s": adv,
                "sd_s": 0.25,
                "passed": passed,
                "end_gap_s": -0.2 if passed else 0.4 + 0.05 * rng.standard_normal(),
            }
        )
    return pd.DataFrame(rows)


def test_fit_recovers_the_model_it_was_drawn_from() -> None:
    fit = fit_overtaking(_synthetic_encounters(), ridge=1.0)
    assert fit.min_gap_s == pytest.approx(0.4, abs=0.03)
    assert fit.scale_s == pytest.approx(0.9, rel=0.2)
    assert fit.threshold_for("easy") == pytest.approx(0.0, abs=0.15)
    assert fit.threshold_for("hard") == pytest.approx(0.6, abs=0.15)
    assert fit.threshold_for("never seen") == fit.threshold_s  # the season value
    # strong shrinkage pulls both tracks to one season threshold
    shrunk = fit_overtaking(_synthetic_encounters(), ridge=1e6)
    assert shrunk.threshold_for("hard") - shrunk.threshold_for("easy") < 0.01
    assert OvertakeFit.from_json(fit.to_json()) == fit


def _laps() -> pd.DataFrame:
    """Two cars, 6 laps; B (car behind, 0.8 s faster on paper) starts lap 4 0.5 s behind and
    passes; lap 5 is a pit lap for A, lap 6 under yellow."""
    rows = []
    for code, pace, order in (("AAA", 90.0, 1), ("BBB", 89.2, 2)):
        for lap in range(1, 7):
            start = 1000.0 + 90.0 * (lap - 1) + (0.5 if code == "BBB" else 0.0)
            passed = lap >= 4
            pos_start = order if lap <= 4 else 3 - order
            pos_end = (3 - order) if passed else order
            rows.append(
                {
                    "event_id": "race",
                    "driver_code": code,
                    "lap_number": lap,
                    "lap_status": "2" if lap == 6 else "1",
                    "pit_in_lap": code == "AAA" and lap == 5,
                    "pit_out_lap": False,
                    "position_start": pos_start,
                    "position_end": pos_end,
                    "lap_start_time_s": start,
                    "lap_end_time_s": start + pace,
                    "compound": "MEDIUM",
                    "tyre_age_start": float(lap),
                    "laps_remaining": float(6 - lap),
                    "lap_time_s": pace,
                    "representative": True,
                    "gap_ahead_s": 9.0,
                }
            )
    return pd.DataFrame(rows)


def test_encounters_take_clean_green_laps_only() -> None:
    flat = TyreShape(
        offset={"SOFT": 0.0, "MEDIUM": 0.0, "HARD": 0.0},
        wear_lin={"SOFT": 0.0, "MEDIUM": 0.0, "HARD": 0.0},
        wear_quad={"SOFT": 0.0, "MEDIUM": 0.0, "HARD": 0.0},
        fuel_s_per_lap=0.0,
        dirty_air_s=0.0,
    )
    enc = encounters(
        _laps(), flat, {"AAA": 0.2, "BBB": 0.2}, dirty_air_window_s=1.2, dirty_air_loss_s=0.15
    )
    # laps 2-4 are encounters (lap 1 is the start, lap 5 a pit lap, lap 6 not green)
    assert list(enc["lap"]) == [2, 3, 4]
    assert list(enc["passed"]) == [False, False, True]
    # expected advantage = 0.8 s pace - 0.5 s gap - 0.15 s dirty air
    assert enc["advantage_s"].iloc[0] == pytest.approx(0.8 - 0.5 - 0.15)
    assert enc["sd_s"].iloc[0] == pytest.approx(np.hypot(0.2, 0.2))


def test_traffic_params_use_the_tracks_own_threshold() -> None:
    fit, sha = load_traffic(CONFIGS)
    base = default_traffic_params()
    monaco = traffic_params(fit, sha, "2024_monaco_gp", base)
    vegas = traffic_params(fit, sha, "2024_las_vegas_gp", base)
    unknown = traffic_params(fit, sha, "2030_mars_gp", base)
    assert monaco.source_type is SourceType.ESTIMATED_FROM_DATA
    assert monaco.overtake_threshold_s > fit.threshold_s > vegas.overtake_threshold_s
    assert unknown.overtake_threshold_s == fit.threshold_s and "season" in unknown.note
    assert monaco.start_spread_s == fit.start_spread_s > 0.0
    # the time losses stay editorial
    assert monaco.attacker_loss_s == base.attacker_loss_s


# --- race control ----------------------------------------------------------------------------


def test_race_control_rates_by_moments() -> None:
    counts = [
        RaceCounts("a", 20, 1000, 1, 2, 1, 1, 0, [4.0]),
        RaceCounts("b", 20, 1000, 0, 1, 0, 0, 0, [6.0]),
    ]
    fit = fit_race_control(counts)
    assert fit.lap1_retirement_per_starter == pytest.approx(1 / 40)
    assert fit.retirement_per_car_lap == pytest.approx(3 / 2000)
    assert fit.crash_per_lap + fit.failure_per_lap == pytest.approx(3 / 2000)
    # lap 1 (crash x multiplier + failure) reproduces the lap-1 rate
    lap1 = fit.crash_per_lap * fit.first_lap_crash_multiplier + fit.failure_per_lap
    assert lap1 == pytest.approx(1 / 40, rel=0.01)
    assert fit.sc_per_retirement == pytest.approx(1 / 4, abs=1e-4)
    assert fit.vsc_per_retirement_without_sc == pytest.approx(1 / 3, abs=1e-4)  # rounded


def test_race_control_params_normalise_by_the_fields_skill() -> None:
    fit, sha = load_race_control(CONFIGS)
    base = default_race_control_params()
    p = race_control_params(fit, sha, base, mean_skill_factor=0.5)
    assert p.crash_per_lap == pytest.approx(fit.crash_per_lap / 0.5)
    assert p.crash_sc_probability == p.failure_sc_probability == fit.sc_per_retirement
    assert p.sc_min_laps <= p.sc_max_laps
    assert p.mistake_per_lap == base.mistake_per_lap  # still editorial


# --- the launch -------------------------------------------------------------------------------


def _grid_setup(roster: RaceRoster, profiles: ProfileSet, spread: float) -> RaceSetup:
    traffic = default_traffic_params().model_copy(update={"start_spread_s": spread})
    return build_setup(
        roster,
        profiles,
        roster.driver_ids(),
        traffic=traffic,
        race_control=default_race_control_params(enabled=False),
    )


def test_launch_spread_mixes_lap_one_and_is_keyed(
    monza_roster: RaceRoster, monza_profiles: ProfileSet
) -> None:
    still = _grid_setup(monza_roster, monza_profiles, 0.0)
    spread = _grid_setup(monza_roster, monza_profiles, 2.0)
    lap1 = [
        sum(sim_metrics(s, seed, until_lap=1).extra["pass_causes"]["lap1"] for seed in (1, 2, 3))
        for s in (still, spread)
    ]
    assert lap1[1] > lap1[0] + 5
    a = engine.create_world(spread, seed=4, world_id="a")
    b = engine.create_world(spread, seed=4, world_id="b")
    assert [c.lap_target_s for c in a.cars] == [c.lap_target_s for c in b.cars]
    # the launch only adds its own keyed draw to lap 1 (spread 0 = the lap 1 of v0.8.0)
    base = engine.create_world(still, seed=4, world_id="c")
    for with_launch, without in zip(a.cars, base.cars, strict=True):
        launch = 2.0 * normal(4, "launch", with_launch.driver_id)
        assert with_launch.lap_target_s - without.lap_target_s == pytest.approx(launch)


def test_fitted_files_exist_and_cover_every_dry_race() -> None:
    fit, _ = load_traffic(CONFIGS)
    rc, _ = load_race_control(CONFIGS)
    assert len(fit.track_threshold_s) == rc.races == 21
    assert fit.min_gap_s > 0 and fit.scale_s > 0
    assert Path(CONFIGS / "models" / "traffic_2024_v1.yaml").exists()


def test_worlds_with_fitted_traffic_survive_a_save_and_load(
    monza_roster: RaceRoster, monza_profiles: ProfileSet
) -> None:
    """Found by the M0 gate: a fitted negative threshold passed ``model_copy`` but failed when a
    save was loaded. Every fitted track must give a world that round-trips."""
    fit, sha = load_traffic(CONFIGS)
    assert min(fit.track_threshold_s.values()) < 0.0  # the case that broke
    for event_id in fit.track_threshold_s:
        traffic = traffic_params(fit, sha, event_id, default_traffic_params())
        setup = build_setup(monza_roster, monza_profiles, monza_roster.driver_ids()[:2])
        world = engine.create_world(
            setup.model_copy(update={"traffic": traffic}), seed=1, world_id="w"
        )
        assert type(world).model_validate_json(world.model_dump_json()) == world


def test_fitted_pass_curve_never_passes_cars_that_do_not_meet() -> None:
    # threshold below zero: the engine still needs the cars to meet (margin > 0)
    p = pass_probability(np.array([-0.3, -0.01, 0.01, 0.3]), -0.2, 1.0)
    assert p[0] == p[1] == 0.0 < p[2] < p[3]
