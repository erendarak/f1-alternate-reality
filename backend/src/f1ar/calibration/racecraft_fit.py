"""Racecraft from 2024 race laps: overtaking with driver attack/defend skills, launch skill,
lap-1 risk by grid position (W3, engine v0.13.0, D-044).

- **Overtaking refit (W2 noise) with driver skills.** The lap-level encounters of EXP-006
  (``traffic_fit.encounters``) are fitted again by maximum likelihood, now with each driver's
  fitted lap noise (W2) convolved instead of the persona noise, and with two terms per driver:
  ``attack`` lowers the pass threshold when he is the attacker, ``defend`` raises it when he is
  the defender. Track terms and driver terms are ridge-shrunk (a driver is average until the
  encounters say otherwise); ``scale`` is shared. This replaces W2's global threshold shift.
- **Launch skill:** positions gained on lap 1 (grid - position after lap 1) minus what the grid
  slot (pooled mean by slot) and the car's race pace vs its grid slot explain; per driver the sum
  over races divided by races + ``LAUNCH_PRIOR_RACES``; converted to seconds with the median
  lap-1 gap between consecutive cars.
- **Lap-1 risk by grid position:** lap-1 incidents are rare at the front, most common in the
  midfield (Dartmouth Sports Analytics, 2022-23: by grid row 3, 16, 18, ... 18, 7, 2 car
  involvements; rows 4-7 not published and interpolated); a profile with mean 1 that scales the
  fitted lap-1 crash rate. The shape is editorial (informed by that study), not fitted here.
"""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from f1ar.calibration.season_files import REFERENCE_SEASON, own_season
from f1ar.calibration.traffic_fit import held_gap, pass_probability_with_noise

RACECRAFT_FILE = Path("calibration") / "racecraft_2024_v1.json"
RIDGE_TRACK = 50.0
RIDGE_DRIVER = 50.0
LAUNCH_PRIOR_RACES = 6.0
DARTMOUTH_ROWS = (3.0, 16.0, 18.0, 17.5, 17.5, 17.5, 17.5, 18.0, 7.0, 2.0)
"""Lap-1 incident involvements by grid row (rows 4-7 interpolated: not published)."""


def lap1_grid_profile(cars: int = 20) -> tuple[float, ...]:
    """Lap-1 risk multiplier per grid slot (two cars per row), mean 1 (editorial shape)."""
    per_slot = [DARTMOUTH_ROWS[min(i // 2, len(DARTMOUTH_ROWS) - 1)] for i in range(cars)]
    mean = sum(per_slot) / len(per_slot)
    return tuple(round(v / mean, 4) for v in per_slot)


def fit_overtaking_skills(
    enc: pd.DataFrame, *, ridge_track: float = RIDGE_TRACK, ridge_driver: float = RIDGE_DRIVER
) -> dict[str, Any]:
    """Season threshold, shared scale, unexplained rate, track terms, attack/defend per driver."""
    min_gap = held_gap(enc)
    margin = enc["advantage_s"].to_numpy() + min_gap
    sd = enc["sd_s"].to_numpy()
    y = enc["passed"].to_numpy().astype(float)
    tracks = sorted(enc["event_id"].unique())
    drivers = sorted(set(enc["attacker"]) | set(enc["defender"]))
    ti = enc["event_id"].map({t: i for i, t in enumerate(tracks)}).to_numpy()
    di = {d: i for i, d in enumerate(drivers)}
    ai = enc["attacker"].map(di).to_numpy()
    fi = enc["defender"].map(di).to_numpy()
    nt, nd = len(tracks), len(drivers)

    def unpack(theta: np.ndarray) -> tuple[float, float, float, np.ndarray, np.ndarray, np.ndarray]:
        return (
            float(theta[0]),
            float(np.exp(theta[1])),
            float(1.0 / (1.0 + np.exp(-theta[2]))),
            theta[3 : 3 + nt],
            theta[3 + nt : 3 + nt + nd],
            theta[3 + nt + nd :],
        )

    def nll(theta: np.ndarray) -> float:
        thr, scale, eps, u, att, dfd = unpack(theta)
        t = thr + u[ti] - att[ai] + dfd[fi]
        p = eps + (1.0 - eps) * pass_probability_with_noise(margin, sd, t, scale)
        p = np.clip(p, 1e-9, 1 - 1e-9)
        ll = float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))
        penalty = ridge_track * float(u @ u) + ridge_driver * float(att @ att + dfd @ dfd)
        return -ll + 0.5 * penalty

    theta0 = np.concatenate([[0.5, 0.0, -4.0], np.zeros(nt + 2 * nd)])
    res = minimize(nll, theta0, method="L-BFGS-B")
    thr, scale, eps, u, att, dfd = unpack(res.x)
    counts = enc.groupby("event_id")["passed"].agg(["size", "sum"])
    made = enc[enc["passed"]].groupby("attacker").size()
    suffered = enc[enc["passed"]].groupby("defender").size()
    return {
        "min_gap_s": round(min_gap, 4),
        "threshold_s": round(thr, 4),
        "scale_s": round(scale, 4),
        "eps": round(eps, 5),
        "track_threshold_s": {t: round(thr + float(u[i]), 4) for i, t in enumerate(tracks)},
        "track_encounters": {t: int(counts.loc[t, "size"]) for t in tracks},
        "track_passes": {t: int(counts.loc[t, "sum"]) for t in tracks},
        "drivers": {
            d: {
                "attack_s": round(float(att[i]), 4),
                "defend_s": round(float(dfd[i]), 4),
                "encounters_attacking": int((enc["attacker"] == d).sum()),
                "encounters_defending": int((enc["defender"] == d).sum()),
                "passes_made": int(made.get(d, 0)),
                "passes_suffered": int(suffered.get(d, 0)),
            }
            for i, d in enumerate(drivers)
        },
        "encounters": len(enc),
        "passes": int(y.sum()),
        "ridge_track": ridge_track,
        "ridge_driver": ridge_driver,
        "log_likelihood": round(-float(res.fun), 2),
    }


def launch_skills(laps: pd.DataFrame, events: Sequence[str]) -> dict[str, Any]:
    """Lap-1 positions gained beyond grid slot and car pace, per driver, shrunk; and the
    median lap-1 gap between consecutive cars (to turn positions into seconds)."""
    rows = []
    gaps = []
    for ev in events:
        r = laps[laps["event_id"] == ev]
        lap1 = r[(r["lap_number"] == 1) & r["position_end"].notna() & (r["grid_position"] > 0)]
        if len(lap1) < 10:
            continue
        pace = r[r["lap_number"] > 1].groupby("team_id")["lap_time_s"].median()
        team_rank = pace.rank()
        rank_of_driver = lap1["team_id"].map(team_rank) * 2 - 0.5  # ~grid slot of that car
        ends = lap1.sort_values("position_end")["lap_end_time_s"].to_numpy(dtype=float)
        gaps += list(np.diff(ends[np.isfinite(ends)]))
        for rec, pace_slot in zip(lap1.to_dict("records"), rank_of_driver, strict=True):
            rows.append(
                {
                    "driver_code": str(rec["driver_code"]),
                    "grid": int(rec["grid_position"]),
                    "gain": int(rec["grid_position"]) - int(rec["position_end"]),
                    "pace_vs_grid": float(rec["grid_position"]) - float(pace_slot),
                }
            )
    df = pd.DataFrame(rows)
    df["slot_mean"] = df.groupby("grid")["gain"].transform("mean")
    x = df["pace_vs_grid"].to_numpy()
    resid0 = (df["gain"] - df["slot_mean"]).to_numpy()
    beta = float(np.dot(x, resid0) / max(1e-9, float(np.dot(x, x))))
    df["resid"] = resid0 - beta * x
    per_pos = float(np.median([g for g in gaps if 0.0 < g < 5.0]))
    drivers = {}
    for d, g in df.groupby("driver_code"):
        n = len(g)
        skill = float(g["resid"].sum()) / (n + LAUNCH_PRIOR_RACES)
        drivers[str(d)] = {
            "positions_per_start": round(skill, 3),
            "launch_offset_s": round(-skill * per_pos, 4),
            "raw_mean_positions": round(float(g["resid"].mean()), 3),
            "starts": n,
        }
    return {
        "seconds_per_position": round(per_pos, 4),
        "pace_vs_grid_beta": round(beta, 4),
        "mean_abs_gain": round(float(df["gain"].abs().mean()), 3),
        "drivers": drivers,
    }


def load_racecraft(configs: Path, season: int = REFERENCE_SEASON) -> dict[str, Any] | None:
    """This season's fit, or None: keyed by driver/team, never borrowed (H2a, D-054)."""
    path = own_season(configs, RACECRAFT_FILE, season)
    if path is None:
        return None
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data
