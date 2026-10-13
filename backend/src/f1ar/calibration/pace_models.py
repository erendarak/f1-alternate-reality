"""Pace models for EXP-003 (M4 slice 2, plan §8.3).

**The shape** (``TyreShape``) is the part of the engine's lap model that should generalise
across races: compound offsets, tyre wear as a function of age (linear + quadratic), the fuel
effect per lap of fuel remaining and a dirty-air loss. Race base pace and car/driver pace are
*not* part of the shape: they differ per race and are estimated from each race (D-021, D-026).

Fitting: ordinary least squares on representative laps with one pace term per driver per race
(absorbed by demeaning). Inside one stint tyre age and laps remaining move in lockstep, so the
fuel and wear terms are only separated by laps of the same driver on different stints (a stop
resets the age, not the fuel). The fuel term therefore also contains track evolution
(rubbering in): both make laps faster as the race goes on and timing data can't tell them apart.

**Per-race adjustment** (``RaceAdjustment``): a wear multiplier and compound offset deltas for
one race, shrunk towards the season shape (ridge), estimated from that race's own laps, i.e.
in-sample like the engine's base lap and car offsets.

**Online predictors** predict a lap from what was known when it started (``pace_features``):
the driver's recent pace, corrected by the shape for what changed since (tyre age, fuel,
compound).
"""

import json
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from f1ar.domain.race_state import Compound, LapModelParams

DRY = ("SOFT", "MEDIUM", "HARD")
REFERENCE = "MEDIUM"
DIRTY_AIR_GAP_S = 1.2
"""Following within this many seconds at the line counts as dirty air (engine: TrafficParams
``dirty_air_window_s``)."""


@dataclass(frozen=True)
class TyreShape:
    offset: dict[str, float]
    """Fresh-tyre lap-time offset vs MEDIUM (MEDIUM = 0)."""
    wear_lin: dict[str, float]
    """Seconds added per lap of tyre age."""
    wear_quad: dict[str, float]
    """Seconds added per (lap of tyre age)^2."""
    fuel_s_per_lap: float
    """Seconds per lap of fuel remaining (includes track evolution)."""
    dirty_air_s: float
    label: str = ""
    fitted_on: tuple[str, ...] = field(default=())

    def wear(self, compound: Any, age: Any) -> Any:
        c = np.asarray(compound, dtype=object)
        a = np.asarray(age, dtype=float)
        lin = np.vectorize(lambda x: self.wear_lin.get(str(x), np.nan), otypes=[float])(c)
        quad = np.vectorize(lambda x: self.wear_quad.get(str(x), np.nan), otypes=[float])(c)
        return lin * a + quad * a * a

    def value(self, compound: Any, age: Any, fuel: Any, dirty: Any = 0.0) -> Any:
        """The shape's lap-time contribution (seconds; vectorised)."""
        c = np.asarray(compound, dtype=object)
        off = np.vectorize(lambda x: self.offset.get(str(x), np.nan), otypes=[float])(c)
        return (
            off
            + self.wear(c, age)
            + self.fuel_s_per_lap * np.asarray(fuel, dtype=float)
            + self.dirty_air_s * np.asarray(dirty, dtype=float)
        )

    def to_json(self) -> str:
        d = asdict(self)
        d["fitted_on"] = list(self.fitted_on)
        return json.dumps(d, indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "TyreShape":
        d = json.loads(text)
        d["fitted_on"] = tuple(d.get("fitted_on", ()))
        return cls(**d)

    @classmethod
    def placeholder(cls, params: LapModelParams, dirty_air_s: float = 0.15) -> "TyreShape":
        """The engine's current editorial coefficients (D-010) as a shape, for comparison: a
        neutral driver (tire_management 0.5) in BALANCED mode."""
        factor = (params.tyre_mgmt_wear_max + params.tyre_mgmt_wear_min) / 2
        ref = params.compounds[Compound.MEDIUM].offset_s
        return cls(
            offset={c: params.compounds[Compound(c)].offset_s - ref for c in DRY},
            wear_lin={c: params.compounds[Compound(c)].deg_s_per_lap * factor for c in DRY},
            wear_quad={c: 0.0 for c in DRY},
            fuel_s_per_lap=params.fuel_s_per_lap_remaining,
            dirty_air_s=dirty_air_s,
            label="placeholder (engine v0.6.0, editorial)",
        )


def training_laps(laps: pd.DataFrame) -> pd.DataFrame:
    """Representative dry-compound laps with the columns the shape needs."""
    r = laps[
        laps["representative"].astype(bool)
        & laps["compound"].isin(DRY)
        & laps["lap_time_s"].notna()
        & laps["tyre_age_start"].notna()
    ].copy()
    r["y"] = r["lap_time_s"].astype(float)
    r["age"] = r["tyre_age_start"].astype(float)
    r["fuel"] = r["laps_remaining"].astype(float)
    r["dirty"] = (r["gap_ahead_s"].astype(float) < DIRTY_AIR_GAP_S).astype(float).fillna(0.0)
    r["dr"] = r["event_id"].astype(str) + "|" + r["driver_code"].astype(str)
    return r


def _design(r: pd.DataFrame, *, quadratic: bool, dirty_air: bool) -> pd.DataFrame:
    cols: dict[str, pd.Series] = {}
    for c in DRY:
        if c != REFERENCE:
            cols[f"off_{c}"] = (r["compound"] == c).astype(float)
    for c in DRY:
        m = (r["compound"] == c).astype(float)
        cols[f"lin_{c}"] = m * r["age"]
        if quadratic:
            cols[f"quad_{c}"] = m * r["age"] ** 2
    cols["fuel"] = r["fuel"]
    if dirty_air:
        cols["dirty"] = r["dirty"]
    return pd.DataFrame(cols, index=r.index)


def _demean(x: pd.DataFrame | pd.Series, groups: pd.Series) -> Any:
    return x - x.groupby(groups).transform("mean")


def fit_shape(
    laps: pd.DataFrame, *, quadratic: bool = True, dirty_air: bool = True, label: str = ""
) -> TyreShape:
    """OLS with a pace term per driver per race (absorbed). ``laps``: the lap dataset."""
    r = training_laps(laps)
    x = _design(r, quadratic=quadratic, dirty_air=dirty_air)
    xd = _demean(x, r["dr"]).to_numpy()
    yd = _demean(r["y"], r["dr"]).to_numpy()
    beta, *_ = np.linalg.lstsq(xd, yd, rcond=None)
    b = dict(zip(x.columns, (float(v) for v in beta), strict=True))
    return TyreShape(
        offset={c: (0.0 if c == REFERENCE else b[f"off_{c}"]) for c in DRY},
        wear_lin={c: b[f"lin_{c}"] for c in DRY},
        wear_quad={c: b.get(f"quad_{c}", 0.0) for c in DRY},
        fuel_s_per_lap=b["fuel"],
        dirty_air_s=b.get("dirty", 0.0),
        label=label,
        fitted_on=tuple(sorted(r["event_id"].astype(str).unique())),
    )


@dataclass(frozen=True)
class RaceAdjustment:
    """One race's deviation from the season shape (estimated from that race, in-sample)."""

    wear_multiplier: float
    offset_delta: dict[str, float]

    def apply(self, shape: TyreShape) -> TyreShape:
        k = self.wear_multiplier
        return TyreShape(
            offset={c: shape.offset[c] + self.offset_delta.get(c, 0.0) for c in DRY},
            wear_lin={c: shape.wear_lin[c] * k for c in DRY},
            wear_quad={c: shape.wear_quad[c] * k for c in DRY},
            fuel_s_per_lap=shape.fuel_s_per_lap,
            dirty_air_s=shape.dirty_air_s,
            label=f"{shape.label} + race adjustment",
            fitted_on=shape.fitted_on,
        )


def fit_race_adjustment(
    race: pd.DataFrame, shape: TyreShape, *, ridge: float = 200.0
) -> RaceAdjustment:
    """Wear multiplier and S/H offset deltas for one race, ridge-shrunk to (1, 0, 0).

    ``ridge`` acts like that many laps of prior evidence for the season shape.
    """
    r = training_laps(race)
    if len(r) < 20:
        return RaceAdjustment(1.0, {})
    base = shape.value(r["compound"], r["age"], r["fuel"], r["dirty"])
    wear = shape.wear(r["compound"], r["age"])
    cols = {"k": wear}
    for c in DRY:
        if c != REFERENCE:
            cols[f"d_{c}"] = (r["compound"] == c).astype(float).to_numpy()
    x = pd.DataFrame(cols, index=r.index)
    # target: what the season shape leaves, with the wear term moved to the left (k = 1 prior)
    y = r["y"] - pd.Series(base, index=r.index) + pd.Series(wear, index=r.index)
    xd = _demean(x, r["dr"]).to_numpy()
    yd = _demean(y, r["dr"]).to_numpy()
    prior = np.array([1.0] + [0.0] * (xd.shape[1] - 1))
    # The prior weighs like ``ridge`` laps of the kind the column describes: scale by the
    # column's typical square on the laps where it applies. (Slice 2 averaged over ALL laps,
    # which made the prior vanish for a rarely used compound: Monza's single representative
    # SOFT lap, a late fastest-lap attempt, then set a -0.96 s SOFT offset; found in M4 slice 3.)
    raw = x.to_numpy()
    scale = np.array(
        [float((col[col != 0] ** 2).mean()) if (col != 0).any() else 1.0 for col in raw.T]
    )
    penalty = np.diag(ridge * scale)
    beta = np.linalg.solve(xd.T @ xd + penalty, xd.T @ yd + penalty @ prior)
    names = list(x.columns)
    return RaceAdjustment(
        wear_multiplier=float(beta[0]),
        offset_delta={n[2:]: float(v) for n, v in zip(names[1:], beta[1:], strict=True)},
    )


def shape_residuals(race: pd.DataFrame, shape: TyreShape) -> np.ndarray:
    """Residuals of a race's representative laps after the shape and a pace term per driver
    (the per-driver pace is fitted on the same race: in-sample, as in the engine)."""
    r = training_laps(race)
    pred = pd.Series(shape.value(r["compound"], r["age"], r["fuel"], r["dirty"]), index=r.index)
    res: np.ndarray = _demean(r["y"] - pred, r["dr"]).to_numpy()
    return res


# --- online predictors ------------------------------------------------------------------------


def online_rows(laps: pd.DataFrame) -> pd.DataFrame:
    """Laps the online predictors are evaluated on: representative dry laps with history."""
    r = training_laps(laps)
    r = r[r["drv_recent_n"].fillna(0).astype(int) > 0]
    return r[r["drv_recent_compound"].isin(DRY)]


def predict_rolling(rows: pd.DataFrame) -> np.ndarray:
    """Baseline A0: the driver's recent median."""
    return rows["drv_recent_s"].astype(float).to_numpy()


def predict_rolling_corrected(rows: pd.DataFrame, shape: TyreShape) -> np.ndarray:
    """Baseline A: recent median + what the shape says changed since (age, fuel, compound)."""
    now = shape.value(rows["compound"], rows["age"], rows["fuel"])
    then = shape.value(
        rows["drv_recent_compound"],
        rows["drv_recent_age"].astype(float),
        rows["drv_recent_fuel"].astype(float),
    )
    out: np.ndarray = rows["drv_recent_s"].astype(float).to_numpy() + now - then
    return out
