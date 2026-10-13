"""Fitting the weather model on the 2024 wet races (M4 slice 4, D-034, EXP-005).

Fitted jointly by non-linear least squares on the wet races, from two kinds of evidence:

1. **Lap times** (representative laps), with one pace term per driver per race (absorbed):

       lap = pace(driver, race) + fuel + [slick: dry shape offset + wear]
             + penalty(compound, w(lap start))

2. **The teams' tyre choices** (every timed lap): P(wet-weather tyre) = logistic of how much
   cheaper the model says it is than a slick (scale ``CHOICE_SCALE_S``). Lap times alone can't
   pin the crossover down: in a race that is wet throughout, the pace term absorbs the overall
   wet slowness (found in the first fit: the rates went to their bounds, crossover 0.96).

``w`` follows the rain timeline through the wetting/drying rates (``simulation/weather``), with
plausible bounds (editorial): wet within 2-30 min of rain, dry within 5-90 min.
Fitted: wetting rate, drying rate, slick slope, intermediate base and slope. The full-wet tyre
line is editorial (14 usable full-wet laps in 2024): it is placed so that full wets beat
intermediates only above ``WET_CROSSOVER`` wetness, ``WET_EXTRA_BASE_S`` slower when dry.
"""

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from f1ar.calibration.pace_models import TyreShape
from f1ar.domain.provenance import SourceType
from f1ar.domain.race_state import (
    Compound,
    WeatherMode,
    WeatherParams,
    default_weather_params,
)
from f1ar.simulation.weather import crossovers, penalty_s

WET_CROSSOVER = 0.85
"""Editorial: full wets beat intermediates only above this wetness (heavy rain)."""
WET_EXTRA_BASE_S = 3.0
"""Editorial: full wets are this much slower than intermediates on a dry track."""
CHOICE_SCALE_S = 2.0
"""How sharply teams follow the cheaper tyre (logistic scale, seconds per lap)."""
CHOICE_WEIGHT = 1.0
PRE_RACE_S = 3600.0
"""Rain this long before the start already wets the track (initial wetness)."""


def rain_timeline(weather: pd.DataFrame, race_start_s: float) -> tuple[tuple[float, bool], ...]:
    """(seconds from the race start, raining) samples, including the hour before the start."""
    w = weather[["session_time_s", "rainfall"]].dropna().sort_values("session_time_s")
    rows = [
        (round(float(t) - race_start_s, 1), bool(r))
        for t, r in zip(w["session_time_s"], w["rainfall"], strict=True)
        if float(t) - race_start_s >= -PRE_RACE_S
    ]
    return tuple(rows)


def wetness_series(
    timeline: tuple[tuple[float, bool], ...], times: np.ndarray, wet_rate: float, dry_rate: float
) -> np.ndarray:
    """Wetness at each of ``times`` (seconds from the start), starting dry an hour before."""
    order = np.argsort(times)
    out = np.empty(len(times))
    w, t_prev = 0.0, -PRE_RACE_S
    raining = False
    i = 0
    samples = list(timeline)
    for idx in order:
        t = float(times[idx])
        while i < len(samples) and samples[i][0] <= t:
            ts, flag = samples[i]
            w = _advance(w, raining, ts - t_prev, wet_rate, dry_rate)
            t_prev, raining = ts, flag
            i += 1
        out[idx] = _advance(w, raining, t - t_prev, wet_rate, dry_rate)
    return out


def _advance(w: float, raining: bool, dt: float, wet_rate: float, dry_rate: float) -> float:
    if dt <= 0:
        return w
    if raining:
        return 1.0 - (1.0 - w) * float(np.exp(-wet_rate * dt))
    return w * float(np.exp(-dry_rate * dt))


def initial_wetness(
    timeline: tuple[tuple[float, bool], ...], wet_rate: float, dry_rate: float
) -> float:
    return float(wetness_series(timeline, np.array([0.0]), wet_rate, dry_rate)[0])


@dataclass(frozen=True)
class WetLaps:
    """Representative laps of wet races (lap-time evidence), every timed lap's tyre choice
    (choice evidence), and each race's rain timeline."""

    frame: pd.DataFrame
    timelines: dict[str, tuple[tuple[float, bool], ...]]
    choices: pd.DataFrame | None = None


def wet_laps(laps: pd.DataFrame, weather: dict[str, pd.DataFrame], shape: TyreShape) -> WetLaps:
    r = laps[laps["representative"].astype(bool) & laps["lap_time_s"].notna()].copy()
    starts = laps[laps["lap_number"] == 1].groupby("event_id")["lap_start_time_s"].min()
    timelines = {e: rain_timeline(weather[e], float(starts[e])) for e in r["event_id"].unique()}
    r["t_rel"] = r["lap_start_time_s"].astype(float) - r["event_id"].map(starts).astype(float)
    r["y"] = r["lap_time_s"].astype(float)
    slick = r["compound"].isin(["SOFT", "MEDIUM", "HARD"])
    age = r["tyre_age_start"].astype(float).fillna(0.0)
    known = shape.value(
        r["compound"].where(slick, "MEDIUM"), age, r["laps_remaining"].astype(float)
    )
    # wet tyres: no dry offset / dry wear, only the fuel term
    fuel = shape.fuel_s_per_lap * r["laps_remaining"].astype(float).to_numpy()
    r["known"] = np.where(slick.to_numpy(), known, fuel)
    r["dr"] = r["event_id"].astype(str) + "|" + r["driver_code"].astype(str)
    known_c = laps["compound"].isin([x.value for x in Compound])
    c = laps[known_c & laps["lap_start_time_s"].notna()]
    choices = pd.DataFrame(
        {
            "event_id": c["event_id"].astype(str),
            "t_rel": c["lap_start_time_s"].astype(float) - c["event_id"].map(starts).astype(float),
            "wet_tyre": c["compound"].isin(["INTERMEDIATE", "WET"]).astype(float),
        }
    ).reset_index(drop=True)
    return WetLaps(r[r["compound"].isin([x.value for x in Compound])], timelines, choices)


def _wetness(data: WetLaps, frame: pd.DataFrame, params: WeatherParams) -> np.ndarray:
    w = np.empty(len(frame))
    for e, idx in frame.groupby("event_id").indices.items():
        w[idx] = wetness_series(
            data.timelines[str(e)],
            frame["t_rel"].to_numpy()[idx],
            params.wet_rate_per_s,
            params.dry_rate_per_s,
        )
    return w


def choice_residuals(data: WetLaps, params: WeatherParams) -> np.ndarray:
    """sqrt(2 x negative log-likelihood) of each lap's tyre choice (slick vs wet-weather)."""
    if data.choices is None or data.choices.empty:
        return np.zeros(0)
    f = data.choices
    w = _wetness(data, f, params)
    slick = params.slick_s_per_wetness * w
    wet_type = np.minimum(
        params.inter_base_s + params.inter_s_per_wetness * w,
        params.wet_base_s + params.wet_s_per_wetness * w,
    )
    z = (slick - wet_type) / CHOICE_SCALE_S
    p_wet = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
    chosen = np.where(f["wet_tyre"].to_numpy() > 0.5, p_wet, 1.0 - p_wet)
    out: np.ndarray = np.sqrt(-2.0 * np.log(np.clip(chosen, 1e-9, 1.0)) * CHOICE_WEIGHT)
    return out


def _params(theta: np.ndarray, base: WeatherParams) -> WeatherParams:
    wet_rate, dry_rate, slick, inter_base, inter_slope = (float(v) for v in theta)
    wet_base = inter_base + WET_EXTRA_BASE_S
    wet_slope = max(0.0, inter_slope - WET_EXTRA_BASE_S / WET_CROSSOVER)
    return base.model_copy(
        update={
            "wet_rate_per_s": wet_rate,
            "dry_rate_per_s": dry_rate,
            "slick_s_per_wetness": slick,
            "inter_base_s": inter_base,
            "inter_s_per_wetness": inter_slope,
            "wet_base_s": wet_base,
            "wet_s_per_wetness": wet_slope,
        }
    )


def residuals(data: WetLaps, params: WeatherParams) -> np.ndarray:
    """Lap-time residuals (pace term per driver per race absorbed)."""
    f = data.frame
    w = _wetness(data, f, params)
    pen = np.array(
        [penalty_s(params, Compound(c), x) for c, x in zip(f["compound"], w, strict=True)]
    )
    raw = f["y"].to_numpy() - f["known"].to_numpy() - pen
    s = pd.Series(raw, index=f.index)
    out: np.ndarray = (s - s.groupby(f["dr"]).transform("mean")).to_numpy()
    return out


def fit_weather(data: WetLaps) -> tuple[WeatherParams, dict[str, Any]]:
    base = default_weather_params()
    theta0 = np.array([1 / 600, 1 / 1800, 25.0, 4.0, 10.0])
    lower = np.array([1 / 1800, 1 / 5400, 0.0, 0.0, 0.0])  # wet within 30 min, dry within 90
    upper = np.array([1 / 120, 1 / 300, 120.0, 40.0, 60.0])  # ... not faster than 2 / 5 min

    def total(th: np.ndarray) -> np.ndarray:
        p = _params(th, base)
        return np.concatenate([residuals(data, p), choice_residuals(data, p)])

    fit = least_squares(total, theta0, bounds=(lower, upper), loss="soft_l1", f_scale=1.0)
    params = _params(fit.x, base).model_copy(
        update={
            "source_type": SourceType.ESTIMATED_FROM_DATA,
            "note": (
                "estimated_from_data: wetting/drying rates and slick/intermediate penalty lines "
                f"fitted on {sorted(data.frame['event_id'].unique())} (EXP-005); full-wet line "
                f"editorial (crossover {WET_CROSSOVER}, +{WET_EXTRA_BASE_S} s vs inters when "
                "dry); wet_skill effect, wet noise/incident factors, switch horizon and "
                "forecast noise editorial"
            ),
        }
    )
    info = {
        "laps": int(len(data.frame)),
        "choices": int(0 if data.choices is None else len(data.choices)),
        "cost": float(fit.cost),
        "success": bool(fit.success),
        "residual_mae_s": float(np.mean(np.abs(residuals(data, params)))),
        "at_bound": [
            bool(np.isclose(v, lo) or np.isclose(v, hi))
            for v, lo, hi in zip(fit.x, lower, upper, strict=True)
        ],
    }
    return params, info


def no_weather_residuals(data: WetLaps) -> np.ndarray:
    """The same model with no weather term (reference)."""
    f = data.frame
    raw = pd.Series(f["y"].to_numpy() - f["known"].to_numpy(), index=f.index)
    out: np.ndarray = (raw - raw.groupby(f["dr"]).transform("mean")).to_numpy()
    return out


WEATHER_FILE = Path("calibration") / "weather_2024_v1.json"


def load_weather(configs_dir: Path) -> tuple[WeatherParams, str] | None:
    """The fitted weather model (without a race timeline) and its file hash, if fitted."""
    path = configs_dir / WEATHER_FILE
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8")
    return WeatherParams.model_validate_json(text), "sha256:" + hashlib.sha256(
        text.encode()
    ).hexdigest()


WETNESS_STEP_S = 60.0
SHARE_CLIP = 0.02


def revealed_wetness_timeline(
    laps: pd.DataFrame, race_start_s: float, params: WeatherParams
) -> tuple[tuple[float, float], ...]:
    """Wetness per minute as the real race had it, inferred from the field's tyre choices.

    The share ``s`` of running cars on wet-weather tyres is turned into wetness by inverting
    the fitted choice model, ``s = logistic((slick(w) - inter(w)) / CHOICE_SCALE_S)``. A dry
    race gives an empty timeline. Teams switch a lap or two after conditions change, so this
    lags the real track a little.
    """
    t0 = laps["lap_start_time_s"].astype(float).to_numpy() - race_start_s
    t1 = laps["lap_end_time_s"].astype(float).to_numpy() - race_start_s
    wet = laps["compound"].isin(["INTERMEDIATE", "WET"]).to_numpy()
    full_wet = (laps["compound"] == "WET").to_numpy()
    _, inter_to_wet = crossovers(params)
    known = laps["compound"].notna().to_numpy() & ~np.isnan(t0) & ~np.isnan(t1)
    if not wet[known].any():
        return ()
    slope = params.slick_s_per_wetness - params.inter_s_per_wetness
    out = []
    end = float(np.nanmax(t1[known]))
    t = 0.0
    while t <= end:
        on = known & (t0 <= t) & (t1 > t)
        if on.any():
            share = float(np.clip(wet[on].mean(), SHARE_CLIP, 1 - SHARE_CLIP))
            w = (CHOICE_SCALE_S * np.log(share / (1 - share)) + params.inter_base_s) / slope
            # slick-vs-wet-tyre choices can't tell inters from full wets: unless most wet-tyre
            # runners were on full wets, keep the track in the intermediate band (found in
            # the browser: Britain 2024 came out on full wets, the field ran inters)
            on_wet = on & wet
            mostly_full_wet = on_wet.any() and full_wet[on_wet].mean() >= 0.5
            ceiling = 1.0 if mostly_full_wet else inter_to_wet - 0.05
            floor = inter_to_wet + 0.05 if mostly_full_wet else 0.0
            out.append((t, round(float(np.clip(w, floor, ceiling)), 3)))
        t += WETNESS_STEP_S
    return tuple(out)


def race_weather(
    fitted: WeatherParams,
    sha: str,
    weather: pd.DataFrame,
    race_start_s: float,
    laps: pd.DataFrame | None = None,
) -> WeatherParams:
    """The fitted model with this race's rain timeline (HISTORICAL) and, given the real laps,
    the wetness timeline inferred from the field's tyre choices."""
    timeline = rain_timeline(weather, race_start_s)
    raining = sum(flag for _, flag in timeline if _ >= 0)
    wetness = () if laps is None else revealed_wetness_timeline(laps, race_start_s, fitted)
    wet_note = (
        f"track wetness inferred from the field's real tyre choices ({len(wetness)} minutes; "
        "estimated_from_data)"
        if wetness
        else "dry race: no wet-tyre laps"
    )
    return fitted.model_copy(
        update={
            "mode": WeatherMode.HISTORICAL,
            "rain_timeline": timeline,
            "wetness_timeline": wetness,
            "initial_wetness": round(
                wetness[0][1]
                if wetness
                else initial_wetness(timeline, fitted.wet_rate_per_s, fitted.dry_rate_per_s),
                4,
            ),
            "note": (
                f"{fitted.note}; rain events: this race's weather data ({len(timeline)} samples, "
                f"{raining} raining after the start; historical_verified flags); {wet_note}; "
                f"model file {WEATHER_FILE.as_posix()} ({sha[:19]}...)"
            ),
        }
    )


def lap_weather_penalty(
    params: WeatherParams, laps: pd.DataFrame, race_start_s: float
) -> pd.Series:
    """Neutral driver's weather penalty of each lap (at its start); 0 when dry."""
    from f1ar.simulation.weather import historical_wetness

    t = laps["lap_start_time_s"].astype(float).to_numpy() - race_start_s
    known = ~np.isnan(t)
    w = np.zeros(len(laps))
    if params.wetness_timeline:
        w[known] = [historical_wetness(params, float(x)) for x in t[known]]
    elif params.rain_timeline:
        w[known] = wetness_series(
            params.rain_timeline, t[known], params.wet_rate_per_s, params.dry_rate_per_s
        )
    pen = [
        penalty_s(params, Compound(c), float(x)) if c in Compound.__members__ else 0.0
        for c, x in zip(laps["compound"], w, strict=True)
    ]
    return pd.Series(pen, index=laps.index)
