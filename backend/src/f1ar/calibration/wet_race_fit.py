"""Wet races fitted on the three wet 2024 races (W6, engine v0.14.0, D-046).

**Fresh wet tyres under SC/VSC.** At every 2024 neutralisation with cars on intermediates or
wets, did a car take a new wet-weather set before it ended, by the age of its set
(``refresh_observations``)? A threshold model (``fit_refresh``) gives the age and the share
of cars that did; the baseline strategy copies it with a keyed draw per car and
neutralisation (``lightweight_engine._cheap_stop``).

**Incidents in the wet.**
The engine multiplies every car's crash and mistake risk by ``1 + wet_incident_factor * w``
(``w`` = track wetness, D-034); the factor was editorial (2.0). Here it is fitted by
simulation: for a grid of factors, the wet races (Canada, Britain, São Paulo, historical
weather) are simulated with per-race seeds, and a Poisson likelihood compares the mean counts of
neutralisations (safety cars + VSCs + red flags) and retirements per race with the real ones.
The interval is the profile-likelihood 95% range (log-likelihood within 1.92 of the best). With
three races it is wide; the file records it.

Measured but NOT used (EXP-015): track temperature does not explain tyre wear in 2024, neither
within a race nor across tracks, so the engine has no temperature effect.
"""

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

WET_COMPOUNDS = ("INTERMEDIATE", "WET")
WET_FILE = Path("calibration") / "wet_race_2024_v1.json"
WET_EVENTS = ("2024_canadian_gp", "2024_british_gp", "2024_sao_paulo_gp")
GRID = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0)
CHI2_95_HALF = 1.92


def poisson_loglik(observed: float, mean: float) -> float:
    m = max(mean, 1e-6)
    return observed * math.log(m) - m - math.lgamma(observed + 1.0)


def profile_fit(
    sim_means: Mapping[float, Mapping[str, Mapping[str, float]]],
    real: Mapping[str, Mapping[str, float]],
    metrics: Sequence[str] = ("neutralisations", "retirements"),
) -> dict[str, Any]:
    """Best factor and 95% range; ``sim_means[factor][event][metric]`` vs ``real[event][metric]``.
    The best point is refined by a parabola through the best grid point and its neighbours."""
    ll = {
        f: sum(poisson_loglik(real[e][m], sim_means[f][e][m]) for e in real for m in metrics)
        for f in sim_means
    }
    grid = sorted(ll)
    i = max(range(len(grid)), key=lambda j: ll[grid[j]])
    best = grid[i]
    if 0 < i < len(grid) - 1:
        (x0, x1, x2), (y0, y1, y2) = grid[i - 1 : i + 2], (ll[g] for g in grid[i - 1 : i + 2])
        denom = (x0 - x1) * (x0 - x2) * (x1 - x2)
        a = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / denom
        b = (x2**2 * (y0 - y1) + x1**2 * (y2 - y0) + x0**2 * (y1 - y2)) / denom
        if a < 0:
            best = min(max(-b / (2 * a), x0), x2)
    inside = [g for g in grid if ll[g] >= ll[grid[i]] - CHI2_95_HALF]
    return {
        "best": round(best, 3),
        "range_95": [min(inside), max(inside)],
        "range_open_above": max(inside) == grid[-1],
        "loglik": {str(g): round(ll[g], 3) for g in grid},
    }


def refresh_observations(
    laps: pd.DataFrame, red_flag_starts_s: Sequence[float]
) -> list[dict[str, Any]]:
    """One row per (neutralisation, car on wet-weather tyres): set age when it began and whether
    the car took a new wet-weather set before it ended. A neutralisation = consecutive laps with
    SC/VSC status (4, 6, 7) on any car, plus the lap before (the stop is often on the lap the SC
    comes out); it is cut at a red flag (that tyre change is free and handled by the engine)."""
    laps = laps.sort_values(["driver_code", "lap_number"])
    status = laps["track_status_start"].astype(str)
    neutral = sorted(set(laps.loc[status.str.contains("4|6|7"), "lap_number"].astype(int)))
    red_laps = [
        int(laps.loc[laps["lap_start_time_s"] <= t, "lap_number"].max()) for t in red_flag_starts_s
    ]
    episodes: list[tuple[int, int]] = []
    for lap in neutral:
        if episodes and lap == episodes[-1][1] + 1:
            episodes[-1] = (episodes[-1][0], lap)
        else:
            episodes.append((lap - 1, lap))
    nxt = laps.groupby("driver_code")["compound"].shift(-1).astype(object)
    laps = laps.assign(next_compound=nxt)
    rows = []
    for first, last in episodes:
        last = min([last, *[r - 1 for r in red_laps if r >= first]])
        for driver, g in laps.groupby("driver_code"):
            start = g[g["lap_number"] == first]
            if start.empty or str(start["compound"].iloc[0]) not in WET_COMPOUNDS:
                continue
            window = g[(g["lap_number"] >= first) & (g["lap_number"] <= last)]
            pits = window[window["pit_in_lap"].fillna(False).astype(bool)]
            to = None if pits.empty else pits["next_compound"].iloc[0]
            rows.append(
                {
                    "first_lap": first,
                    "driver": str(driver),
                    "age": float(start["tyre_age_start"].iloc[0]),
                    "refresh": isinstance(to, str) and to in WET_COMPOUNDS,
                }
            )
    return rows


def fit_refresh(obs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Threshold model: P(new set) = ``share`` if age >= ``min_age`` else ``young_share``;
    maximum likelihood over the thresholds 2..30 (young refreshes are mostly type changes,
    which the weather rule makes anyway; the engine uses only ``share``)."""

    def ll(k: float, n: float) -> float:
        p = min(max(k / n, 1e-6), 1 - 1e-6) if n else 0.5
        return k * math.log(p) + (n - k) * math.log(1 - p) if n else 0.0

    best: dict[str, Any] = {}
    for a in range(2, 31):
        old = [o for o in obs if o["age"] >= a]
        young = [o for o in obs if o["age"] < a]
        k_o, k_y = sum(o["refresh"] for o in old), sum(o["refresh"] for o in young)
        score = ll(k_o, len(old)) + ll(k_y, len(young))
        if not best or score > best["loglik"]:
            best = {
                "min_age_laps": a,
                "share": round(k_o / len(old), 3) if old else 0.0,
                "young_share": round(k_y / len(young), 3) if young else 0.0,
                "old_n": len(old),
                "young_n": len(young),
                "loglik": round(score, 3),
            }
    return best


def load_wet_race(configs: Path) -> dict[str, Any] | None:
    path = configs / WET_FILE
    if not path.exists():
        return None
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data
