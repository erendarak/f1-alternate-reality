"""H3b (EXP-028): the same-lap trend recovers a known track trend, fuel effect and wear."""

import numpy as np
import pandas as pd

from f1ar.calibration.track_evolution import evolution_laps, physics_fuel_slope, race_trend


def _race(
    season: int, trend: float, fuel: float, wear: float = 0.05, seed: int = 0
) -> pd.DataFrame:
    """20 cars, 50 laps, stints of 12-25 laps; fuel on board resets at stops (refuelling)."""
    rng = np.random.default_rng(seed)
    rows = []
    for car in range(20):
        lengths: list[int] = []
        while sum(lengths) < 50:
            lengths.append(int(rng.integers(12, 26)))
        lap = 1
        for stint, length in enumerate(lengths, 1):
            for k in range(length):
                if lap > 50:
                    break
                left = length - k if season <= 2009 else 50 - lap + 1
                t = 90.0 + 0.1 * car + trend * lap + fuel * left + wear * k
                rows.append(
                    {
                        "event_id": f"{season}_test_gp",
                        "season": season,
                        "circuit_id": "test",
                        "driver_code": f"D{car:02d}",
                        "scheduled_laps": 50,
                        "lap_number": lap,
                        "laps_remaining": left,
                        "compound": "MEDIUM",
                        "tyre_age_start": 0.0,
                        "stint": stint,
                        "stint_lap": k + 1,
                        "pit_in_lap": False,
                        "pit_out_lap": False,
                        "track_status_start": "1",
                        "gap_ahead_s": 5.0,
                        "lap_time_s": t + rng.normal(0.0, 0.2),
                    }
                )
                lap += 1
    return pd.DataFrame(rows)


def test_refuelling_era_separates_evolution_from_fuel() -> None:
    fit = race_trend(evolution_laps(_race(2005, trend=-0.02, fuel=0.08)))
    assert fit is not None and fit.fuel_s_per_lap_of_fuel is not None
    assert abs(fit.evolution_s_per_lap + 0.02) < 0.005
    assert abs(fit.fuel_s_per_lap_of_fuel - 0.08) < 0.01
    assert abs(fit.wear_s_per_lap - 0.05) < 0.01
    fixed = race_trend(evolution_laps(_race(2005, trend=-0.02, fuel=0.08)), fuel_fixed=0.08)
    assert fixed is not None and abs(fixed.evolution_s_per_lap + 0.02) < 0.005


def test_after_2009_the_trend_is_fuel_plus_evolution() -> None:
    fuel = 0.03 * 110.0 / 50  # the physics value for 2019
    fit = race_trend(evolution_laps(_race(2019, trend=-0.01, fuel=fuel)), bootstrap=5)
    assert fit is not None and fit.fuel_s_per_lap_of_fuel is None
    assert abs(fit.trend_s_per_lap - (-0.01 - fuel)) < 0.005
    assert abs(physics_fuel_slope(2019, 50) + fuel) < 1e-12
    assert abs(fit.evolution_s_per_lap + 0.01) < 0.005
    assert 0.0 < fit.trend_se < 0.01
