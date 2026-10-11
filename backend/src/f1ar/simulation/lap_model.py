"""Minimal profile-conditioned lap-time model (plan §3.6 A, §7.1).

    lap_time = base_lap
             + skill_gap        * (1 - race_pace_skill)       # driver speed
             + compound offset                                # tyre choice
             + tyre degradation (age, compound, tire_management, pace_mode)
             + fuel effect      * laps_remaining               # car gets lighter
             + pace_mode offset                               # conserve / balanced / push
             + noise ~ N(0, sd(consistency))                   # lap-to-lap variance

Each profile field gets its own causal channel: ``risk_appetite`` is a *preference* and
does not enter the transition model here. It will shape strategy choices instead.
This is a modelling schema with placeholder coefficients, NOT physics and NOT yet
calibrated to real data (M1/M4).
"""

from f1ar.domain.driver_profiles import DriverArchetype
from f1ar.domain.race_state import Compound, LapModelParams, PaceMode
from f1ar.simulation.rng_streams import normal


def tyre_wear_factor(params: LapModelParams, driver: DriverArchetype) -> float:
    """Linear in tire_management: ``wear_max`` at 0 down to ``wear_min`` at 1."""
    span = params.tyre_mgmt_wear_min - params.tyre_mgmt_wear_max
    return params.tyre_mgmt_wear_max + span * driver.tire_management


def degradation_s(
    params: LapModelParams,
    driver: DriverArchetype,
    *,
    compound: Compound,
    tyre_age_laps: int,
    pace_mode: PaceMode,
) -> float:
    return (
        tyre_age_laps
        * params.compounds[compound].deg_s_per_lap
        * tyre_wear_factor(params, driver)
        * params.pace_modes[pace_mode].wear_multiplier
    )


def expected_lap_time_s(
    params: LapModelParams,
    driver: DriverArchetype,
    *,
    compound: Compound,
    tyre_age_laps: int,
    pace_mode: PaceMode,
    laps_remaining: int,
) -> float:
    return (
        params.base_lap_s
        + params.skill_gap_s * (1.0 - driver.race_pace_skill)
        + params.compounds[compound].offset_s
        + degradation_s(
            params, driver, compound=compound, tyre_age_laps=tyre_age_laps, pace_mode=pace_mode
        )
        + params.fuel_s_per_lap_remaining * laps_remaining
        + params.pace_modes[pace_mode].offset_s
    )


def noise_sd_s(params: LapModelParams, driver: DriverArchetype) -> float:
    return params.noise_floor_s + params.noise_inconsistency_s * (1.0 - driver.consistency)


def sample_lap_time_s(
    params: LapModelParams,
    driver: DriverArchetype,
    *,
    seed: int,
    driver_id: str,
    lap_number: int,
    compound: Compound,
    tyre_age_laps: int,
    pace_mode: PaceMode,
    laps_remaining: int,
) -> float:
    """Lap time for ``lap_number``. The noise is keyed by (driver, lap): common across forks."""
    mean = expected_lap_time_s(
        params,
        driver,
        compound=compound,
        tyre_age_laps=tyre_age_laps,
        pace_mode=pace_mode,
        laps_remaining=laps_remaining,
    )
    z = normal(seed, "lap_noise", driver_id, lap_number)
    return max(mean + noise_sd_s(params, driver) * z, params.min_lap_fraction * params.base_lap_s)
