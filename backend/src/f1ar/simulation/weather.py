"""Weather: rain, track wetness and what they cost each tyre (M4 slice 4, D-034).

**Wetness** ``w`` is a hidden track state in [0, 1]: 0 = dry, 1 = soaked after sustained rain.
Nothing in the data measures it; it is driven by the on/off rain signal:

    raining:  w -> 1 - (1 - w) exp(-wet_rate  dt)     (wets towards 1)
    dry:      w -> w exp(-dry_rate dt)                 (dries towards 0)

**Tyre penalty** (seconds per lap on top of the dry lap time), linear in ``w`` per tyre type:

    slick (S/M/H)   slick_per_w * w
    INTERMEDIATE    inter_base + inter_per_w * w
    WET             wet_base + wet_per_w * w

so each tyre is best in a band of wetness; the crossovers are where the lines meet. The wetting
and drying rates and the slick/inter lines are fitted on the 2024 wet races; the WET line is
editorial (14 usable full-wet laps in 2024). A driver's ``wet_skill`` scales the penalty
(editorial).

**Rain** comes from the real race's rain timeline (``HISTORICAL``: exogenous, identical in every
fork) or from a seeded two-state process per minute (``RANDOM``: keyed draws, also identical in
forks with the same seed). Pure functions; the engine stores ``WeatherState`` in the world.
"""

import math

from f1ar.domain.race_state import Compound, WeatherMode, WeatherParams
from f1ar.simulation.rng_streams import uniform

SLICKS = frozenset({Compound.SOFT, Compound.MEDIUM, Compound.HARD})
WET_TYRES = frozenset({Compound.INTERMEDIATE, Compound.WET})


def is_slick(compound: Compound) -> bool:
    return compound in SLICKS


def penalty_s(params: WeatherParams, compound: Compound, wetness: float) -> float:
    """Neutral driver's lap-time loss on ``compound`` at ``wetness``."""
    if compound is Compound.INTERMEDIATE:
        return params.inter_base_s + params.inter_s_per_wetness * wetness
    if compound is Compound.WET:
        return params.wet_base_s + params.wet_s_per_wetness * wetness
    return params.slick_s_per_wetness * wetness


def driver_factor(params: WeatherParams, wet_skill: float) -> float:
    """Penalty multiplier: 1 for a driver at the reference wet_skill (the field's mean in a data
    race), lower for better wet drivers (editorial effect size)."""
    return 1.0 + params.wet_skill_effect * (params.wet_skill_reference - wet_skill)


def best_tyre(params: WeatherParams, wetness: float) -> Compound:
    """The tyre type with the smallest penalty (MEDIUM stands for 'a slick')."""
    options = (Compound.MEDIUM, Compound.INTERMEDIATE, Compound.WET)
    return min(options, key=lambda c: penalty_s(params, c, wetness))


def crossovers(params: WeatherParams) -> tuple[float, float]:
    """Wetness where slicks stop being best, and where full wets start being best."""
    s_i = params.inter_base_s / max(params.slick_s_per_wetness - params.inter_s_per_wetness, 1e-9)
    i_w = (params.wet_base_s - params.inter_base_s) / max(
        params.inter_s_per_wetness - params.wet_s_per_wetness, 1e-9
    )
    return s_i, i_w


def next_wetness(params: WeatherParams, wetness: float, raining: bool, dt: float) -> float:
    if raining:
        return 1.0 - (1.0 - wetness) * math.exp(-params.wet_rate_per_s * dt)
    return wetness * math.exp(-params.dry_rate_per_s * dt)


def historical_wetness(params: WeatherParams, race_time_s: float) -> float:
    """Wetness from the race's inferred wetness timeline (linear between samples)."""
    timeline = params.wetness_timeline
    if race_time_s <= timeline[0][0]:
        return timeline[0][1]
    for (t0, w0), (t1, w1) in zip(timeline, timeline[1:], strict=False):
        if t0 <= race_time_s <= t1:
            return w0 + (w1 - w0) * (race_time_s - t0) / (t1 - t0) if t1 > t0 else w1
    return timeline[-1][1]


def historical_rain(params: WeatherParams, race_time_s: float) -> bool:
    """Rain flag of the real race's timeline at ``race_time_s`` (piecewise constant)."""
    raining = False
    for t, flag in params.rain_timeline:
        if t > race_time_s:
            break
        raining = flag
    return raining


def random_rain(params: WeatherParams, seed: int, minute: int, raining_before: bool) -> bool:
    """Two-state process per race minute, keyed so forks with the same seed share it."""
    u = uniform(seed, "rain", minute)
    if raining_before:
        return u >= params.random_rain_stop_per_min
    return u < params.random_rain_start_per_min


def raining_at(
    params: WeatherParams, seed: int, race_time_s: float, raining_now: bool, dt: float
) -> bool:
    """Whether it rains during the tick that ends at ``race_time_s``."""
    if params.mode is WeatherMode.HISTORICAL:
        return historical_rain(params, race_time_s)
    before, after = int((race_time_s - dt) // 60), int(race_time_s // 60)
    raining = raining_now
    for minute in range(before + 1, after + 1):  # a new minute started inside this tick
        raining = random_rain(params, seed, minute, raining)
    return raining


def rain_share_ahead(params: WeatherParams, race_time_s: float, horizon_s: float) -> float:
    """Share of the next ``horizon_s`` that the timeline says is wet (historical mode only;
    used ONLY to build a noisy forecast, never shown as is)."""
    steps = 20
    return (
        sum(
            historical_rain(params, race_time_s + horizon_s * (i + 0.5) / steps)
            for i in range(steps)
        )
        / steps
    )


def forecast_rain_probability(
    params: WeatherParams, seed: int, race_time_s: float, raining_now: bool
) -> float:
    """What a team's forecast says: P(rain in the next ``forecast_horizon_s``) (US-06).

    Historical: the true share ahead blurred by keyed noise (sd ``forecast_noise``), so it is
    informative but not the future. Random: the process's own probability, which is honest.
    """
    horizon = params.forecast_horizon_s
    if params.mode is WeatherMode.RANDOM:
        minutes = max(1, round(horizon / 60))
        if raining_now:
            return 1.0
        return 1.0 - (1.0 - params.random_rain_start_per_min) ** minutes
    truth = rain_share_ahead(params, race_time_s, horizon)
    minute = int(race_time_s // 60)
    noise = (uniform(seed, "forecast", minute) - 0.5) * 2 * math.sqrt(3) * params.forecast_noise
    return min(1.0, max(0.0, truth + noise))
