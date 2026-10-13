"""Minimal profile-conditioned lap-time model (plan §3.6 A, §7.1).

    lap_time = base_lap
             + skill_gap        * (1 - race_pace_skill)       # driver speed
             + car_offset                                     # car speed (separate from driver)
             + compound offset                                # tyre choice
             + accumulated tyre wear                          # path-dependent, see below
             + fuel effect      * laps_remaining               # car gets lighter
             + pace_mode offset                               # conserve / balanced / push
             + noise ~ N(0, sd(consistency))                   # lap-to-lap variance

Tyre wear is STATE, not a formula of tyre age: each completed lap adds

    wear_per_lap = max(0, deg + quad * (2 * age + 1))       # the curve's step at this age
                   * wear_factor(tire_management) * pace_mode_wear_multiplier

for the pace mode actually driven on that lap (M4: a quadratic curve fitted on real laps; the
clamp means wear never decreases, even where a concave fit would turn down; quad = 0 gives the
old linear model). Changing pace mode therefore changes how
fast the tyres wear from now on. It never rewrites the wear already accumulated (engine
v0.1.0 got this wrong: it scaled the whole stint's wear by the current mode).

Each profile field gets its own causal channel: ``risk_appetite`` is a *preference* and
does not enter the transition model here. It will shape strategy choices instead.
Coefficients: placeholders (D-010) for synthetic setups; for data-backed races the tyre/fuel
part is fitted on the 2024 season (EXP-003, D-033). Skill, noise and pace modes stay editorial.
This is a statistical model of lap times, not physics.
"""

from f1ar.domain.driver_profiles import DriverArchetype
from f1ar.domain.race_state import Compound, LapModelParams, PaceMode
from f1ar.simulation.rng_streams import normal


def tyre_wear_factor(params: LapModelParams, driver: DriverArchetype) -> float:
    """Linear in tire_management: ``wear_max`` at 0 down to ``wear_min`` at 1."""
    span = params.tyre_mgmt_wear_min - params.tyre_mgmt_wear_max
    return params.tyre_mgmt_wear_max + span * driver.tire_management


def wear_step_s(params: LapModelParams, compound: Compound, tyre_age_laps: int) -> float:
    """The neutral curve's increase over one lap at ``tyre_age_laps`` (>= 0, D-033)."""
    c = params.compounds[compound]
    return max(0.0, c.deg_s_per_lap + c.deg_quad_s_per_lap2 * (2 * tyre_age_laps + 1))


def wear_per_lap_s(
    params: LapModelParams,
    driver: DriverArchetype,
    *,
    compound: Compound,
    pace_mode: PaceMode,
    tyre_age_laps: int = 0,
) -> float:
    """Lap time added to every later lap of this tyre set by one lap driven in ``pace_mode``
    on a set that was ``tyre_age_laps`` old when the lap started."""
    return (
        wear_step_s(params, compound, tyre_age_laps)
        * tyre_wear_factor(params, driver)
        * params.pace_modes[pace_mode].wear_multiplier
    )


def degradation_s(
    params: LapModelParams,
    driver: DriverArchetype,
    *,
    compound: Compound,
    tyre_age_laps: int,
    pace_mode: PaceMode,
) -> float:
    """Accumulated wear after ``tyre_age_laps`` laps all driven in the same mode (helper)."""
    return sum(
        wear_per_lap_s(params, driver, compound=compound, pace_mode=pace_mode, tyre_age_laps=a)
        for a in range(tyre_age_laps)
    )


def expected_lap_time_s(
    params: LapModelParams,
    driver: DriverArchetype,
    *,
    compound: Compound,
    tyre_wear_s: float,
    pace_mode: PaceMode,
    laps_remaining: int,
    car_offset_s: float = 0.0,
) -> float:
    return (
        params.base_lap_s
        + params.skill_gap_s * (1.0 - driver.race_pace_skill)
        + car_offset_s
        + params.compounds[compound].offset_s
        + tyre_wear_s
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
    tyre_wear_s: float,
    pace_mode: PaceMode,
    laps_remaining: int,
    car_offset_s: float = 0.0,
    extra_s: float = 0.0,
    noise_scale: float = 1.0,
) -> float:
    """Lap time for ``lap_number``. The noise is keyed by (driver, lap): common across forks.

    ``extra_s``: conditions on top of the dry lap (the weather penalty, M4); ``noise_scale``
    widens the lap-to-lap spread (more in the wet)."""
    mean = expected_lap_time_s(
        params,
        driver,
        compound=compound,
        tyre_wear_s=tyre_wear_s,
        pace_mode=pace_mode,
        laps_remaining=laps_remaining,
        car_offset_s=car_offset_s,
    )
    z = normal(seed, "lap_noise", driver_id, lap_number)
    noisy = mean + extra_s + noise_sd_s(params, driver) * noise_scale * z
    return max(noisy, params.min_lap_fraction * params.base_lap_s)
