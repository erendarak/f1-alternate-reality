"""Pace models (M4 slice 2): the fitting code must recover known coefficients from laps generated
by a known shape, keep the fuel and wear terms apart through stops, and the placeholder shape
must be the engine's current coefficients."""

import numpy as np
import pandas as pd
import pytest

from f1ar.calibration.pace_models import (
    RaceAdjustment,
    TyreShape,
    fit_race_adjustment,
    fit_shape,
    predict_rolling_corrected,
    shape_residuals,
)
from f1ar.simulation.scenarios import default_lap_model

TRUE = TyreShape(
    offset={"SOFT": -0.3, "MEDIUM": 0.0, "HARD": 0.25},
    wear_lin={"SOFT": 0.12, "MEDIUM": 0.07, "HARD": 0.04},
    wear_quad={"SOFT": 0.002, "MEDIUM": 0.0005, "HARD": 0.0},
    fuel_s_per_lap=0.05,
    dirty_air_s=0.3,
)

PLANS = (  # (compound, stint length) per driver; every plan has at least one stop
    (("MEDIUM", 20), ("HARD", 30)),
    (("SOFT", 12), ("HARD", 38)),
    (("HARD", 28), ("MEDIUM", 22)),
    (("SOFT", 15), ("MEDIUM", 18), ("SOFT", 17)),
    (("MEDIUM", 25), ("SOFT", 25)),
    (("HARD", 32), ("SOFT", 18)),
)


def race(event: str, shape: TyreShape, *, noise: float, seed: int, k: float = 1.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for d, plan in enumerate(PLANS):
        pace = 90.0 + 0.3 * d + rng.normal(0, 0.2)
        lap = 1
        for compound, length in plan:
            for age in range(length):
                fuel = 50 - lap + 1
                dirty = float(rng.random() < 0.25)
                t = pace + float(
                    shape.offset[compound]
                    + k * (shape.wear_lin[compound] * age + shape.wear_quad[compound] * age * age)
                    + shape.fuel_s_per_lap * fuel
                    + shape.dirty_air_s * dirty
                )
                rows.append(
                    {
                        "event_id": event,
                        "driver_code": f"D{d}",
                        "representative": True,
                        "compound": compound,
                        "lap_time_s": t + rng.normal(0, noise),
                        "tyre_age_start": float(age),
                        "laps_remaining": fuel,
                        "gap_ahead_s": 0.5 if dirty else 3.0,
                    }
                )
                lap += 1
    return pd.DataFrame(rows)


def season(noise: float = 0.0) -> pd.DataFrame:
    return pd.concat(
        [race(f"r{i}", TRUE, noise=noise, seed=i) for i in range(4)], ignore_index=True
    )


def test_fit_recovers_the_true_shape_exactly_without_noise() -> None:
    fitted = fit_shape(season())
    assert fitted.fuel_s_per_lap == pytest.approx(0.05, abs=1e-6)
    assert fitted.dirty_air_s == pytest.approx(0.3, abs=1e-6)
    for c in ("SOFT", "MEDIUM", "HARD"):
        assert fitted.offset[c] == pytest.approx(TRUE.offset[c], abs=1e-6)
        assert fitted.wear_lin[c] == pytest.approx(TRUE.wear_lin[c], abs=1e-6)
        assert fitted.wear_quad[c] == pytest.approx(TRUE.wear_quad[c], abs=1e-6)
    assert fitted.fitted_on == ("r0", "r1", "r2", "r3")
    assert np.abs(shape_residuals(season(), fitted)).max() < 1e-6


def test_fit_is_close_with_noise_and_fuel_is_not_confused_with_wear() -> None:
    fitted = fit_shape(season(noise=0.3))
    assert fitted.fuel_s_per_lap == pytest.approx(0.05, abs=0.01)
    assert fitted.wear_lin["MEDIUM"] == pytest.approx(0.07, abs=0.03)
    # a wrong fuel value would show up as a much larger residual than the noise
    assert np.std(shape_residuals(season(noise=0.3), fitted)) < 0.35


def test_race_adjustment_recovers_a_high_degradation_track() -> None:
    hot = race("hot", TRUE, noise=0.05, seed=9, k=1.6)
    adj = fit_race_adjustment(hot, TRUE, ridge=1.0)
    assert adj.wear_multiplier == pytest.approx(1.6, abs=0.05)
    adjusted = adj.apply(TRUE)
    assert adjusted.wear_lin["SOFT"] == pytest.approx(TRUE.wear_lin["SOFT"] * adj.wear_multiplier)
    # strong prior: the estimate stays near the season shape
    assert fit_race_adjustment(hot, TRUE, ridge=1e6).wear_multiplier == pytest.approx(1.0, abs=0.01)
    assert RaceAdjustment(1.0, {}).apply(TRUE).offset == TRUE.offset


def test_placeholder_is_the_engine_lap_model() -> None:
    params = default_lap_model()
    shape = TyreShape.placeholder(params)
    assert shape.offset["MEDIUM"] == 0.0
    assert shape.offset["SOFT"] == pytest.approx(-0.6) and shape.offset["HARD"] == pytest.approx(
        0.4
    )
    assert shape.wear_lin["SOFT"] == pytest.approx(0.09)  # neutral wear factor = 1.0
    assert shape.fuel_s_per_lap == params.fuel_s_per_lap_remaining
    assert TyreShape.from_json(shape.to_json()) == shape


def test_rolling_correction_is_exact_when_the_shape_is_true() -> None:
    rows = pd.DataFrame(
        {
            "compound": ["HARD", "MEDIUM"],
            "age": [0.0, 12.0],
            "fuel": [30.0, 20.0],
            "drv_recent_compound": ["MEDIUM", "MEDIUM"],
            "drv_recent_age": [20.0, 10.0],
            "drv_recent_fuel": [32.0, 22.0],
        }
    )
    truth_now = 90.0 + TRUE.value(rows["compound"], rows["age"], rows["fuel"])
    truth_then = 90.0 + TRUE.value(
        rows["drv_recent_compound"], rows["drv_recent_age"], rows["drv_recent_fuel"]
    )
    rows["drv_recent_s"] = truth_then
    assert predict_rolling_corrected(rows, TRUE) == pytest.approx(truth_now)


def test_one_odd_lap_on_a_rare_compound_barely_moves_its_offset() -> None:
    """Regression (M4 slice 3): a race with a single, very fast SOFT lap must not get a large
    SOFT offset delta; the prior acts like ``ridge`` laps of that compound."""
    r = race("one_soft", TRUE, noise=0.0, seed=3)
    r = r[r["compound"] != "SOFT"].copy()
    lonely = r.iloc[[10]].copy()
    lonely["compound"] = "SOFT"
    lonely["lap_time_s"] = lonely["lap_time_s"] - 1.5  # e.g. a late fastest-lap attempt
    r = pd.concat([r, lonely], ignore_index=True)
    adj = fit_race_adjustment(r, TRUE)  # default ridge = 200 laps
    assert abs(adj.offset_delta["SOFT"]) < 0.02
    # 87 real SOFT laps 0.5 s faster: a light prior finds it, the 200-lap prior shrinks it
    many = race("many_soft", TRUE, noise=0.0, seed=4)
    many.loc[many["compound"] == "SOFT", "lap_time_s"] -= 0.5
    light = fit_race_adjustment(many, TRUE, ridge=2.0).offset_delta["SOFT"]
    assert light == pytest.approx(-0.5, abs=0.05)
    assert -0.5 < fit_race_adjustment(many, TRUE).offset_delta["SOFT"] < -0.05
