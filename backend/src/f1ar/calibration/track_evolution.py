"""Track evolution measured on real laps (H3b, EXP-028).

Per race, on green dry laps (``tyre_life_fit.green_laps`` filters, without its editorial fuel
correction), a same-lap regression:

    lap time = lap term + driver + compound offset + compound wear (age, age^2) + fresh peak
               [+ fuel x laps of fuel on board, refuelling era]

The **lap terms** hold everything every car shares on that lap. Their slope over the race
(least squares on the lap terms, weighted by cars per lap) is the **common trend**:

- 2000-2009: cars on the same lap carry different fuel (their own stints), so the fuel effect is
  estimated beside it and the trend is **track evolution** (plus changing conditions: air and
  track temperature, wind);
- 2010-2025: every car burns the same fuel, so the trend is **fuel burn-off + evolution**.
  Evolution = trend - ``FUEL_S_PER_KG`` x race fuel / race laps, with the race fuel per season
  from ``RACE_FUEL_KG`` (editorial, see there).

Our engine (D-033) puts this common trend into its fuel term since 2010 (it is fitted on the lap
terms), so the *average* evolution is already in the world there; before 2010 the fuel term
follows the fuel on board and has no evolution.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import lsqr

from f1ar.calibration.tyre_life_fit import DRY
from f1ar.domain.refuelling import FUEL_S_PER_KG, LAST_REFUELLING_SEASON

RACE_FUEL_KG = {
    **dict.fromkeys(range(2010, 2014), 160.0),
    **dict.fromkeys(range(2014, 2017), 100.0),
    **dict.fromkeys(range(2017, 2019), 105.0),
    **dict.fromkeys(range(2019, 2026), 110.0),
}
"""Fuel for a race (EDITORIAL): the FIA race-fuel limits 100 kg (2014), 105 kg (2017), 110 kg
(2019) as the fuel actually carried (teams often carried a little less); 2010-2013 had no limit:
~160 kg is the commonly quoted race load of that era (rule of thumb, not in our data)."""
MIN_LAPS_PER_TERM = 5
"""Lap terms with fewer green cars are left out of the trend."""


@dataclass(frozen=True)
class RaceTrend:
    event_id: str
    season: int
    circuit_id: str
    race_laps: int
    laps_used: int
    trend_s_per_lap: float
    """Slope of the common lap terms (negative = laps get faster)."""
    trend_se: float
    fuel_s_per_lap_of_fuel: float | None
    """Refuelling era: the fuel effect estimated beside the trend (None after 2009)."""
    evolution_s_per_lap: float
    """The trend net of fuel burn-off (2010+: physics estimate taken off)."""
    wear_s_per_lap: float
    """Lap time one more lap of tyre age adds on the race's most-used compound, at its mean age
    (the signal an in-race estimator reads)."""


def evolution_laps(laps: pd.DataFrame) -> pd.DataFrame:
    """``green_laps`` filters on raw lap times (no editorial fuel correction)."""
    g = laps[
        (laps["track_status_start"].astype(str) == "1")
        & ~laps["pit_in_lap"].astype(bool)
        & ~laps["pit_out_lap"].astype(bool)
        & (laps["lap_number"] > 1)
        & laps["compound"].isin(DRY)
        & (laps["gap_ahead_s"].fillna(99.0) > 1.0)
    ].copy()
    g["age"] = g["tyre_age_start"] + g["stint_lap"] - 1
    g = g[g["age"].notna() & g["lap_time_s"].notna()]
    med = g.groupby("event_id")["lap_time_s"].transform("median")
    g = g[g["lap_time_s"] < med * 1.07]
    g["fresh"] = ((g["stint_lap"] == 2) & (g["stint"] > 1)).astype(float)
    return g.reset_index(drop=True)


def physics_fuel_slope(season: int, race_laps: int) -> float:
    """Lap time gained per lap from burning fuel, 2010+ (s per lap; negative)."""
    return -FUEL_S_PER_KG * RACE_FUEL_KG[season] / race_laps


def race_trend(
    g: pd.DataFrame, *, fuel_fixed: float | None = None, bootstrap: int = 0, seed: int = 0
) -> RaceTrend | None:
    """The common trend of one race's ``evolution_laps`` (None: too few laps). Refuelling era:
    ``fuel_fixed`` takes a known fuel effect off instead of estimating it. ``bootstrap`` > 0:
    the SE is the spread over races rebuilt from resampled drivers (keeps each car's laps
    together), else the lap-term regression's SE (a floor)."""
    fit = _trend(g, fuel_fixed)
    if fit is None or bootstrap <= 0:
        return fit
    rng = np.random.default_rng(seed)
    drivers = g["driver_code"].astype(str).unique()
    slopes = []
    for _ in range(bootstrap):
        pick = rng.choice(drivers, size=len(drivers), replace=True)
        parts = [
            g[g["driver_code"] == d].assign(driver_code=f"{d}#{i}") for i, d in enumerate(pick)
        ]
        b = _trend(pd.concat(parts, ignore_index=True), fuel_fixed)
        if b is not None:
            slopes.append(b.trend_s_per_lap)
    if len(slopes) < 2:
        return fit
    return RaceTrend(**{**fit.__dict__, "trend_se": round(float(np.std(slopes, ddof=1)), 5)})


def _trend(g: pd.DataFrame, fuel_fixed: float | None) -> RaceTrend | None:
    if len(g) < 100:
        return None
    g = g.reset_index(drop=True)
    season = int(g["season"].iloc[0])
    refuel = season <= LAST_REFUELLING_SEASON
    if refuel and fuel_fixed is not None:
        fuel_off = fuel_fixed * g["laps_remaining"].to_numpy(dtype=float)
        g = g.assign(lap_time_s=g["lap_time_s"].to_numpy(dtype=float) - fuel_off)
    n = len(g)
    lap_key, lap_values = pd.factorize(g["lap_number"].astype(int), sort=True)
    drv_key = pd.factorize(g["driver_code"].astype(str))[0]
    names: list[str] = []
    cols: list[np.ndarray] = []
    present = [c for c in DRY if (g["compound"] == c).sum() > 30]
    age = g["age"].to_numpy(dtype=float)
    for c in present:
        m = (g["compound"] == c).to_numpy(dtype=float)
        if c != present[0]:
            cols.append(m)
            names.append(f"off:{c}")
        cols += [m * age, m * age**2]
        names += [f"lin:{c}", f"quad:{c}"]
    cols.append(g["fresh"].to_numpy(dtype=float))
    names.append("fresh")
    if refuel and fuel_fixed is None:
        cols.append(g["laps_remaining"].to_numpy(dtype=float))
        names.append("fuel")
    X = sp.hstack(
        [
            sp.csr_matrix((np.ones(n), (np.arange(n), lap_key))),
            sp.csr_matrix((np.ones(n), (np.arange(n), drv_key)))[:, 1:],
            sp.csr_matrix(np.column_stack(cols)),
        ]
    ).tocsr()
    b = lsqr(X, g["lap_time_s"].to_numpy(dtype=float), atol=1e-12, btol=1e-12, iter_lim=50_000)[0]
    terms = b[: len(lap_values)]
    counts = np.bincount(lap_key, minlength=len(lap_values))
    keep = counts >= MIN_LAPS_PER_TERM
    if keep.sum() < 10:
        return None
    x = np.asarray(lap_values, dtype=float)[keep]
    y = terms[keep]
    w = counts[keep].astype(float)
    xm = np.average(x, weights=w)
    ym = np.average(y, weights=w)
    sxx = float(np.sum(w * (x - xm) ** 2))
    slope = float(np.sum(w * (x - xm) * (y - ym)) / sxx)
    resid = y - ym - slope * (x - xm)
    # weights = cars per lap term (precision ~ count); the lap terms are autocorrelated, so
    # this SE is a floor
    s2 = float(np.sum(w * resid**2) / (len(x) - 2))
    se = float(np.sqrt(s2 / sxx))
    coef = dict(zip(names, b[-len(names) :], strict=True))
    race_laps = int(g["scheduled_laps"].iloc[0])
    fuel = (float(coef["fuel"]) if fuel_fixed is None else fuel_fixed) if refuel else None
    evolution = slope if refuel else slope - physics_fuel_slope(season, race_laps)
    main = g["compound"].value_counts().index[0]
    mean_age = float(g.loc[g["compound"] == main, "age"].mean())
    wear = float(coef[f"lin:{main}"] + 2.0 * coef[f"quad:{main}"] * mean_age)
    return RaceTrend(
        event_id=str(g["event_id"].iloc[0]),
        season=season,
        circuit_id=str(g["circuit_id"].iloc[0]),
        race_laps=race_laps,
        laps_used=n,
        trend_s_per_lap=round(slope, 5),
        trend_se=round(se, 5),
        fuel_s_per_lap_of_fuel=None if fuel is None else round(fuel, 4),
        evolution_s_per_lap=round(evolution, 5),
        wear_s_per_lap=round(wear, 5),
    )
