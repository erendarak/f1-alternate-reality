"""Fitting the engine's overtaking model on real 2024 laps (M4 slice 5, D-035, EXP-006).

The fit uses **lap-level encounters**, not race pass counts, so the simulated pass counts remain
an independent check. An encounter is a green lap (the whole lap under status 1, not lap 1)
where car A starts the lap within ``MAX_GAP_S`` behind the car directly ahead of it in the race
order (D, same lap), neither car pits on the lap, and both are on dry tyres. Outcome: A ahead
of D at the end of the lap.

The engine decides a pass from the attacker's margin (``simulation/traffic``):

    margin = (D's lap) - (A's lap) - gap + min_gap      (A's lap including dirty air)
    P(pass | margin) = 1 - exp(-(margin - threshold) / scale)   for margin > threshold

The real laps' *expected* times come from the season tyre shape with each race's adjustment
and a pace term per driver per race (in-sample, as in the engine). The engine adds its own lap
noise to those expectations, so the fitted curve is the one that, **after the engine's noise**,
gives the real pass rate: P(pass) = E_z[h(margin + sd z)], sd from both drivers' persona noise
(Gauss-Hermite). Passes the model can't explain at all (a defender's big mistake, damage) go
into a floor ``eps`` that is reported, not used by the engine (the engine draws mistakes).

Per track: ``threshold`` = season value + a track term, ridge-shrunk to the season value
(penalty ``RIDGE/2 x term^2`` on the log-likelihood, strength chosen by cross-validation over odd
vs even laps within each race, EXP-006). ``scale`` is shared. ``min_gap`` = median end-of-lap gap of
attackers that were clearly faster but stayed behind.
"""

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from f1ar.calibration.pace_models import DRY, TyreShape, fit_race_adjustment, training_laps
from f1ar.domain.provenance import SourceType
from f1ar.domain.race_state import TrafficParams

MAX_GAP_S = 2.5
"""Only cars starting the lap at most this far behind the car ahead are encounters."""
RIDGE = 50.0
"""Shrinkage of the track terms (per s^2). Odd/even-lap cross-validation is flat between 20 and
50 (EXP-006); 50, the more conservative end, is used."""
HELD_FASTER_S = 0.5
"""An attacker this much faster (expected) that didn't pass counts as held up (min_gap)."""
TRAFFIC_FILE = Path("calibration") / "traffic_2024_v1.json"
_NODES, _WEIGHTS = np.polynomial.hermite_e.hermegauss(24)
_WEIGHTS = _WEIGHTS / _WEIGHTS.sum()


def noise_sd(consistency: float, floor_s: float, inconsistency_s: float) -> float:
    """The engine's lap noise sd for a driver (``lap_model.noise_sd_s``)."""
    return floor_s + inconsistency_s * (1.0 - consistency)


def expected_laps(race: pd.DataFrame, shape: TyreShape) -> pd.Series:
    """Expected free-air lap time of every dry-tyre lap of one race: season shape x the race's
    adjustment + the driver's pace term (mean residual of their representative laps)."""
    adjusted = fit_race_adjustment(race, shape).apply(shape)
    r = training_laps(race)
    resid = r["y"] - adjusted.value(r["compound"], r["age"], r["fuel"], r["dirty"])
    pace = resid.groupby(r["driver_code"]).mean()
    dry = race[race["compound"].isin(DRY) & race["tyre_age_start"].notna()]
    value = adjusted.value(
        dry["compound"], dry["tyre_age_start"].astype(float), dry["laps_remaining"].astype(float)
    )
    out = pd.Series(value, index=dry.index) + dry["driver_code"].map(pace).astype(float)
    return out.dropna()


def encounters(
    laps: pd.DataFrame,
    shape: TyreShape,
    sd_of: dict[str, float],
    *,
    dirty_air_window_s: float,
    dirty_air_loss_s: float,
) -> pd.DataFrame:
    """One row per encounter: event, lap, gap, expected advantage, noise sd, outcome."""
    rows: list[dict[str, Any]] = []
    for event_id, race in laps.groupby("event_id", sort=True):
        expected = expected_laps(race, shape)
        r = race.assign(expected=expected)
        ok = (
            (r["lap_number"] >= 2)
            & (r["lap_status"].astype(str) == "1")
            & ~r["pit_in_lap"].astype(bool)
            & ~r["pit_out_lap"].astype(bool)
            & r["expected"].notna()
            & r["position_end"].notna()
            & r["lap_start_time_s"].notna()
            & r["lap_end_time_s"].notna()
        )
        by_lap: dict[int, dict[int, dict[Any, Any]]] = {}
        for rec in r[ok].to_dict("records"):
            by_lap.setdefault(int(rec["lap_number"]), {})[int(rec["position_start"])] = rec
        for lap, at in by_lap.items():
            for pos, a in at.items():
                d = at.get(pos - 1)
                if d is None:
                    continue
                gap = float(a["lap_start_time_s"]) - float(d["lap_start_time_s"])
                if not 0.0 < gap <= MAX_GAP_S:
                    continue
                advantage = float(d["expected"]) - float(a["expected"]) - gap
                dirty = dirty_air_loss_s if gap < dirty_air_window_s else 0.0
                a_code, d_code = str(a["driver_code"]), str(d["driver_code"])
                rows.append(
                    {
                        "event_id": str(event_id),
                        "lap": int(lap),
                        "attacker": a_code,
                        "defender": d_code,
                        "gap_s": gap,
                        "advantage_s": advantage - dirty,
                        "sd_s": math.hypot(sd_of.get(a_code, 0.3), sd_of.get(d_code, 0.3)),
                        "passed": bool(int(a["position_end"]) < int(d["position_end"])),
                        "end_gap_s": float(a["lap_end_time_s"]) - float(d["lap_end_time_s"]),
                    }
                )
    return pd.DataFrame(rows)


def pass_probability(margin: Any, threshold: Any, scale: float) -> Any:
    """The engine's rule: no attempt unless the cars meet (margin > 0); then
    ``1 - exp(-(margin - threshold) / scale)`` above the threshold."""
    margin = np.asarray(margin, dtype=float)
    m = margin - np.asarray(threshold, dtype=float)
    return np.where((m > 0.0) & (margin > 0.0), -np.expm1(-np.maximum(m, 0.0) / scale), 0.0)


def pass_probability_with_noise(margin: Any, sd: Any, threshold: Any, scale: float) -> np.ndarray:
    """E_z[P(pass | margin + sd z)], z standard normal: what the engine produces on average."""
    m = np.asarray(margin, dtype=float)[:, None] + np.asarray(sd, dtype=float)[:, None] * _NODES
    t = np.broadcast_to(np.asarray(threshold, dtype=float), m.shape[:1])[:, None]
    p: np.ndarray = pass_probability(m, t, scale) @ _WEIGHTS
    return p


@dataclass(frozen=True)
class OvertakeFit:
    min_gap_s: float
    threshold_s: float
    """Season threshold (track-agnostic)."""
    scale_s: float
    eps: float
    """Unexplained pass rate per encounter (mistakes, damage): reported, not used."""
    track_threshold_s: dict[str, float]
    track_encounters: dict[str, int]
    track_passes: dict[str, int]
    encounters: int
    passes: int
    log_likelihood: float
    fitted_on: tuple[str, ...] = field(default=())
    note: str = ""
    start_spread_s: float = 0.0
    """Lap-1 launch spread, matched by simulation to the real lap-1 passes (EXP-006)."""

    def threshold_for(self, event_id: str) -> float:
        return self.track_threshold_s.get(event_id, self.threshold_s)

    def to_json(self) -> str:
        d = asdict(self)
        d["fitted_on"] = list(self.fitted_on)
        return json.dumps(d, indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "OvertakeFit":
        d = json.loads(text)
        d["fitted_on"] = tuple(d.get("fitted_on", ()))
        return cls(**d)


def held_gap(enc: pd.DataFrame) -> float:
    """Median end-of-lap gap of attackers clearly faster that stayed behind (and close)."""
    held = enc[~enc["passed"] & (enc["advantage_s"] > HELD_FASTER_S)]
    held = held[(held["end_gap_s"] > 0.0) & (held["end_gap_s"] < MAX_GAP_S)]
    return float(held["end_gap_s"].median()) if len(held) else 0.4


def fit_overtaking(enc: pd.DataFrame, *, ridge: float = RIDGE, note: str = "") -> OvertakeFit:
    """Maximum likelihood: season threshold, shared scale, ridge-shrunk track terms, eps."""
    min_gap = held_gap(enc)
    margin = enc["advantage_s"].to_numpy() + min_gap
    sd = enc["sd_s"].to_numpy()
    y = enc["passed"].to_numpy().astype(float)
    tracks = sorted(enc["event_id"].unique())
    index = enc["event_id"].map({t: i for i, t in enumerate(tracks)}).to_numpy()

    def unpack(theta: np.ndarray) -> tuple[float, float, float, np.ndarray]:
        thr = float(theta[0])
        scale = float(np.exp(theta[1]))
        eps = float(1.0 / (1.0 + np.exp(-theta[2])))
        return thr, scale, eps, theta[3:]

    def nll(theta: np.ndarray) -> float:
        thr, scale, eps, u = unpack(theta)
        p = pass_probability_with_noise(margin, sd, thr + u[index], scale)
        p = eps + (1.0 - eps) * p
        p = np.clip(p, 1e-9, 1 - 1e-9)
        ll = float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))
        return -ll + 0.5 * ridge * float(np.sum(u * u))

    theta0 = np.concatenate([[0.5, np.log(1.0), -4.0], np.zeros(len(tracks))])
    res = minimize(nll, theta0, method="L-BFGS-B")
    thr, scale, eps, u = unpack(res.x)
    counts = enc.groupby("event_id")["passed"].agg(["size", "sum"])
    return OvertakeFit(
        min_gap_s=round(min_gap, 4),
        threshold_s=round(thr, 4),
        scale_s=round(scale, 4),
        eps=round(eps, 5),
        track_threshold_s={t: round(thr + float(u[i]), 4) for i, t in enumerate(tracks)},
        track_encounters={t: int(counts.loc[t, "size"]) for t in tracks},
        track_passes={t: int(counts.loc[t, "sum"]) for t in tracks},
        encounters=len(enc),
        passes=int(y.sum()),
        log_likelihood=round(-float(res.fun) + 0.5 * ridge * float(np.sum(u * u)), 2),
        fitted_on=tuple(tracks),
        note=note,
    )


def load_traffic(configs_dir: Path) -> tuple[OvertakeFit, str]:
    """The fitted overtaking model and its file hash."""
    text = (configs_dir / TRAFFIC_FILE).read_text(encoding="utf-8")
    return OvertakeFit.from_json(text), "sha256:" + hashlib.sha256(text.encode()).hexdigest()


def traffic_params(fit: OvertakeFit, sha: str, event_id: str, base: TrafficParams) -> TrafficParams:
    """The engine's traffic parameters for one race: fitted gap, threshold (this track's, or the
    season's for a track the fit never saw), scale and launch spread; time losses for passing,
    dirty air and blue flags stay editorial (``base``)."""
    own = event_id in fit.track_threshold_s
    where = "this track's" if own else "season (track not in the fit)"
    # validated: a fitted value outside the schema must fail here, not when a save is loaded
    return TrafficParams.model_validate(
        base.model_dump()
        | {
            "source_type": SourceType.ESTIMATED_FROM_DATA,
            "note": (
                f"estimated_from_data: min gap {fit.min_gap_s:.2f} s, {where} overtake "
                f"threshold {fit.threshold_for(event_id):+.2f} s, scale {fit.scale_s:.2f} s from "
                f"{fit.encounters} real encounters (EXP-006, {sha[:15]}); launch spread "
                f"{fit.start_spread_s:.2f} s matched to real lap-1 passes; passing/dirty-air/"
                "blue-flag time losses editorial"
            ),
            "min_gap_s": fit.min_gap_s,
            "overtake_threshold_s": fit.threshold_for(event_id),
            "overtake_scale_s": fit.scale_s,
            "start_spread_s": fit.start_spread_s,
        }
    )
