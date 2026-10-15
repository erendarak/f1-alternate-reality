"""W3 (engine v0.13.0, D-044): attack/defend skills, launch skill, lap-1 risk by grid slot, and
the racecraft fit."""

import numpy as np
import pandas as pd
import pytest

from f1ar.calibration.racecraft_fit import fit_overtaking_skills, lap1_grid_profile
from f1ar.domain.race_state import Compound, PaceMode, RaceSetup
from f1ar.simulation import lightweight_engine as engine
from f1ar.simulation.incidents import IncidentKind, lap_incident
from f1ar.simulation.traffic import Ahead, resolve
from tests.conftest import HAM, VER


def _passes(setup: RaceSetup, seeds: range) -> int:
    world = engine.create_world(setup, seed=1, world_id="w")
    me = world.car(VER).model_copy(update={"lap_target_s": 90.0, "laps_completed": 5})
    other = world.car(HAM).model_copy(update={"lap_target_s": 90.6, "laps_completed": 5})
    n = 0
    for s in seeds:
        out = resolve(setup, s, me, 0.3, [Ahead(car=other, lap_start_s=0.0, gap_s=0.3)])
        n += any(k == "OVERTAKE" for k, _ in out.events)
    return n


def test_attack_and_defend_move_the_pass_threshold(two_car_setup: RaceSetup) -> None:
    traffic = two_car_setup.traffic.model_copy(
        update={"overtake_threshold_s": 0.5, "overtake_scale_s": 0.4, "min_gap_s": 0.4}
    )
    base = two_car_setup.model_copy(update={"traffic": traffic})

    def with_skill(attack: float, defend: float) -> RaceSetup:
        ver = base.driver(VER).model_copy(update={"attack_s": attack})
        ham = base.driver(HAM).model_copy(update={"defend_s": defend})
        return base.model_copy(update={"drivers": (ver, ham)})

    seeds = range(1, 300)
    neutral = _passes(base, seeds)
    assert _passes(with_skill(0.3, 0.0), seeds) > neutral  # a strong attacker passes more
    assert _passes(with_skill(0.0, 0.3), seeds) < neutral  # a strong defender holds on more


def test_lap1_profile_shape_and_risk(two_car_setup: RaceSetup) -> None:
    profile = lap1_grid_profile()
    assert len(profile) == 20 and sum(profile) / 20 == pytest.approx(1.0, abs=1e-3)
    assert profile[0] < profile[5] and profile[19] < profile[5]  # front and back safest
    rc = two_car_setup.race_control.model_copy(
        update={"first_lap_grid_profile": profile, "crash_per_lap": 0.01, "mistake_per_lap": 0.0}
    )
    front = two_car_setup.driver(VER).model_copy(update={"grid_position": 1})
    mid = two_car_setup.driver(HAM).model_copy(update={"grid_position": 6})
    setup = two_car_setup.model_copy(update={"race_control": rc, "drivers": (front, mid)})
    world = engine.create_world(setup, seed=1, world_id="w")

    def crashes(driver: str) -> int:
        car = world.car(driver)
        hits = (lap_incident(setup, s, car, 1) for s in range(1, 3000))
        return sum(1 for i in hits if i is not None and i.kind is IncidentKind.CRASH)

    assert crashes(HAM) > crashes(VER)  # personas differ too, but x6 in the profile dominates


def test_launch_skill_is_part_of_lap_one(two_car_setup: RaceSetup) -> None:
    good = two_car_setup.driver(VER).model_copy(update={"launch_offset_s": -0.6})
    setup = two_car_setup.model_copy(update={"drivers": (good, two_car_setup.driver(HAM))})
    kw = {"compound": Compound.MEDIUM, "pace_mode": PaceMode.BALANCED, "tyre_wear_s": 0.0}
    lap1 = engine._sample_next_lap(setup, 3, VER, laps_completed=0, **kw)  # type: ignore[arg-type]
    base1 = engine._sample_next_lap(two_car_setup, 3, VER, laps_completed=0, **kw)  # type: ignore[arg-type]
    lap5 = engine._sample_next_lap(setup, 3, VER, laps_completed=4, **kw)  # type: ignore[arg-type]
    base5 = engine._sample_next_lap(two_car_setup, 3, VER, laps_completed=4, **kw)  # type: ignore[arg-type]
    assert lap1 - base1 == pytest.approx(-0.6) and lap5 == pytest.approx(base5)


def test_skill_fit_finds_a_strong_attacker() -> None:
    """Synthetic encounters: X passes more than its margins explain, Y is hard to pass."""
    rng = np.random.default_rng(3)
    rows = []
    for _ in range(4000):
        a, d = rng.choice(["X", "A", "B", "C"]), rng.choice(["Y", "D", "E", "F"])
        adv = rng.normal(0.3, 0.6)
        thr = 0.5 - (0.3 if a == "X" else 0.0) + (0.3 if d == "Y" else 0.0)
        m = adv + 0.45 + rng.normal(0, 0.4)
        p = 1 - np.exp(-(m - thr) / 0.4) if m > thr and m > 0 else 0.0
        rows.append(
            {
                "event_id": "t",
                "attacker": str(a),
                "defender": str(d),
                "advantage_s": adv,
                "sd_s": 0.4,
                "passed": bool(rng.random() < p),
                "end_gap_s": 0.45 if adv > 0.5 else 1.0,
            }
        )
    fit = fit_overtaking_skills(pd.DataFrame(rows), ridge_driver=5.0)
    d = fit["drivers"]
    assert d["X"]["attack_s"] > max(d[k]["attack_s"] for k in "ABC")
    assert d["Y"]["defend_s"] > max(d[k]["defend_s"] for k in "DEF")
