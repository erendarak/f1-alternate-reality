"""In-race pace context for the online pace predictor (M4, plan §8.2 ``recent_lap_pace``).

Features of a lap computed from the laps that had already ENDED when it starts (as-of, no
leakage; covered by the same audit as the lap dataset):

- ``drv_recent_s``: median of the driver's last ``RECENT`` representative laps.
- ``drv_recent_age`` / ``drv_recent_fuel``: mean tyre age / laps remaining of those laps.
- ``drv_recent_compound``: compound of the most recent of them.
- ``drv_recent_n``: how many there were (0 = no history yet).
- ``field_recent_s``: median of every car's representative laps that ended in the
  ``FIELD_WINDOW_S`` before the lap starts (track evolution, conditions).

Only representative laps feed the history (green, accurate, no pit in/out, not lap 1), and
only laps of the same race.
"""

import numpy as np
import pandas as pd

RECENT = 3
FIELD_WINDOW_S = 300.0

RECENT_FEATURES: tuple[str, ...] = (
    "drv_recent_s",
    "drv_recent_age",
    "drv_recent_fuel",
    "drv_recent_compound",
    "drv_recent_n",
    "field_recent_s",
)


def _one_race(race: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=race.index, columns=list(RECENT_FEATURES), dtype=object)
    rep = race[race["representative"].astype(bool) & race["lap_end_time_s"].notna()]
    rep = rep.sort_values("lap_end_time_s")
    rep_end = rep["lap_end_time_s"].to_numpy(dtype=float)
    rep_time = rep["lap_time_s"].to_numpy(dtype=float)
    starts = race["lap_start_time_s"].to_numpy(dtype=float)

    # field: representative laps that ended in [start - window, start]
    hi = np.searchsorted(rep_end, starts, side="right")
    lo = np.searchsorted(rep_end, starts - FIELD_WINDOW_S, side="left")
    field = [
        float(np.median(rep_time[a:b])) if b > a and not np.isnan(s) else np.nan
        for a, b, s in zip(lo, hi, starts, strict=True)
    ]
    out["field_recent_s"] = field

    by_driver = {int(str(k)): g for k, g in rep.groupby("driver_number")}
    recent_s: list[float] = []
    recent_age: list[float] = []
    recent_fuel: list[float] = []
    recent_cmp: list[str | None] = []
    recent_n: list[int] = []
    for number, start in zip(race["driver_number"], starts, strict=True):
        g = by_driver.get(int(number))
        if g is None or np.isnan(start):
            done = g.iloc[0:0] if g is not None else None
        else:
            ends = g["lap_end_time_s"].to_numpy(dtype=float)
            finished = int(np.searchsorted(ends, start, side="right"))  # ended by the start
            done = g.iloc[max(0, finished - RECENT) : finished]
        if done is None or len(done) == 0:
            recent_s.append(np.nan)
            recent_age.append(np.nan)
            recent_fuel.append(np.nan)
            recent_cmp.append(None)
            recent_n.append(0)
            continue
        recent_s.append(float(np.median(done["lap_time_s"].to_numpy(dtype=float))))
        recent_age.append(float(done["tyre_age_start"].astype(float).mean()))
        recent_fuel.append(float(done["laps_remaining"].astype(float).mean()))
        recent_cmp.append(str(done["compound"].iloc[-1]))
        recent_n.append(len(done))
    out["drv_recent_s"] = recent_s
    out["drv_recent_age"] = recent_age
    out["drv_recent_fuel"] = recent_fuel
    out["drv_recent_compound"] = recent_cmp
    out["drv_recent_n"] = recent_n
    return out


def add_recent_features(laps: pd.DataFrame) -> pd.DataFrame:
    """``laps`` (one or more races of the lap dataset) with the in-race context columns."""
    parts = [_one_race(g) for _, g in laps.groupby("event_id", sort=False)]
    ctx = pd.concat(parts).reindex(laps.index)
    out = laps.copy()
    for col in ("drv_recent_s", "drv_recent_age", "drv_recent_fuel", "field_recent_s"):
        out[col] = pd.to_numeric(ctx[col]).astype("Float64")
    out["drv_recent_compound"] = ctx["drv_recent_compound"].astype("string")
    out["drv_recent_n"] = pd.to_numeric(ctx["drv_recent_n"]).astype("Int64")
    return out
