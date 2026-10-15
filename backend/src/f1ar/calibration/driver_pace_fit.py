"""Driver race pace (vs the teammate) and lap-to-lap noise from 2024 race laps (W2, D-043).

Per race, a same-lap regression on green, clean racing laps (no pit in/out lap, no lap 1, not
within 1 s of the car ahead, no outliers): a term for every lap of the race (fuel burn-off,
track evolution, anything the field shares), a term per driver (that driver's race pace), and
per compound a linear + quadratic curve in tyre age.

- **Driver pace vs teammate:** in each race where both team cars have at least ``MIN_LAPS``
  laps, each driver's term minus the pair's mean. The season value is the sum over races divided
  by ``races + PACE_PRIOR_RACES`` (shrunk toward 0: a driver is as fast as his teammate until
  the races say otherwise). Car pace per team and race stays in the race's own data
  (``race_inputs.estimate_car_offsets``, now net of these offsets).
- **Race form:** how much a driver's pace vs his teammate varies from race to race beyond what
  the laps' own noise explains: within-pair variance of the per-race gap minus its sampling
  variance, halved (both drivers' form moves the gap). The engine draws a new form per driver
  and race (keyed, so forks share it); why form varies isn't in the data.
- **Lap noise:** the residual SD of the driver's laps with clear air ahead (> ``CLEAN_GAP_S``),
  pooled over the season (|residual| < 2 s, so mistakes and incidents stay with the incident
  model), shrunk toward the field median with ``NOISE_PRIOR_LAPS``.

Real data -> ``estimated_from_data``; the persona fields (``race_pace_skill``, ``consistency``)
stay editorial and are no longer used by the lap model of a data-backed race.
"""

import json
from collections.abc import Sequence
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import lsqr

from f1ar.calibration.tyre_life_fit import green_laps

PACE_FILE = Path("calibration") / "driver_pace_2024_v1.json"
MIN_LAPS = 10
PACE_PRIOR_RACES = 3.0
CLEAN_GAP_S = 2.5
NOISE_PRIOR_LAPS = 150.0
MAX_RESIDUAL_S = 2.0


def race_fit(g: pd.DataFrame) -> tuple[dict[str, float], pd.DataFrame]:
    """Driver terms (relative to the race's first driver) and the laps with their residuals."""
    g = g.reset_index(drop=True)
    n = len(g)
    lap_key = pd.factorize(g["lap_number"])[0]
    drivers, drv_key = np.unique(g["driver_code"].astype(str).to_numpy(), return_inverse=True)
    cols: list[np.ndarray] = []
    for c in ("SOFT", "MEDIUM", "HARD"):
        m = (g["compound"] == c).to_numpy(dtype=float)
        if m.sum() > 30:
            age = g["age"].to_numpy(dtype=float)
            cols += [m * age, m * age**2]
    X = sp.hstack(
        [
            sp.csr_matrix((np.ones(n), (np.arange(n), lap_key))),
            sp.csr_matrix((np.ones(n), (np.arange(n), drv_key)))[:, 1:],
            sp.csr_matrix(np.column_stack(cols)) if cols else sp.csr_matrix((n, 0)),
        ]
    ).tocsr()
    y = g["lap_time_s"].to_numpy(dtype=float)
    b = lsqr(X, y, atol=1e-12, btol=1e-12, iter_lim=50_000)[0]
    n_laps = int(lap_key.max()) + 1
    terms = np.concatenate(([0.0], b[n_laps : n_laps + len(drivers) - 1]))
    g = g.assign(residual=y - X @ b)
    return {str(d): float(t) for d, t in zip(drivers, terms, strict=True)}, g


def fit(laps: pd.DataFrame, events: Sequence[str]) -> dict[str, Any]:
    all_green = green_laps(laps[laps["event_id"].isin(events)])
    gaps: dict[str, list[tuple[str, float, int]]] = {}
    pair_gaps: list[tuple[str, float, float]] = []  # (pair, gap, sampling variance)
    residuals: list[pd.DataFrame] = []
    team_of: dict[str, str] = {}
    for ev, g in all_green.groupby("event_id"):
        if len(g) < 200:
            continue
        terms, res = race_fit(g)
        residuals.append(res)
        counts = g.groupby("driver_code").size()
        teams = g.groupby("driver_code")["team_id"].first()
        team_of.update({str(d): str(t) for d, t in teams.items()})
        for _team, members in teams.groupby(teams):
            pair = [str(d) for d in members.index if counts.get(d, 0) >= MIN_LAPS]
            if len(pair) != 2:
                continue
            mean = sum(terms[d] for d in pair) / 2
            sd_r = res.groupby("driver_code")["residual"].std()
            samp = sum(float(sd_r[d]) ** 2 / int(counts[d]) for d in pair)
            pair_gaps.append(("|".join(sorted(pair)), terms[pair[0]] - terms[pair[1]], samp))
            for d in pair:
                gaps.setdefault(d, []).append((str(ev), terms[d] - mean, int(counts[d])))
    pg = pd.DataFrame(pair_gaps, columns=["pair", "gap", "samp"])
    n_pair = pg.groupby("pair")["gap"].transform("count")
    dev = pg["gap"] - pg.groupby("pair")["gap"].transform("mean")
    within = float((dev**2 * n_pair / (n_pair - 1)).where(n_pair > 2).mean())
    form_sd = float(np.sqrt(max(0.0, (within - float(pg["samp"].mean())) / 2.0)))
    res_all = pd.concat(residuals, ignore_index=True)
    clean = res_all[
        (res_all["gap_ahead_s"].fillna(99.0) > CLEAN_GAP_S)
        & (res_all["residual"].abs() < MAX_RESIDUAL_S)
    ]
    sd = clean.groupby("driver_code")["residual"].std()
    n = clean.groupby("driver_code").size()
    field = float(median(sd.to_list()))
    drivers: dict[str, Any] = {}
    for d in sorted(set(gaps) | set(sd.index.astype(str))):
        races = gaps.get(d, [])
        offset = sum(v for _, v, _ in races) / (len(races) + PACE_PRIOR_RACES)
        laps_n = int(n.get(d, 0))
        raw_sd = float(sd.get(d, field))
        noise = (laps_n * raw_sd + NOISE_PRIOR_LAPS * field) / (laps_n + NOISE_PRIOR_LAPS)
        drivers[d] = {
            "team_id": team_of.get(d, "?"),
            "pace_vs_teammate_s": round(offset, 4),
            "raw_mean_gap_s": round(float(np.mean([v for _, v, _ in races])), 4) if races else None,
            "races": len(races),
            "lap_noise_s": round(noise, 4),
            "raw_noise_s": round(raw_sd, 4),
            "clean_laps": laps_n,
        }
    return {
        "label": "2024 driver race pace vs teammate + lap noise (W2, D-043)",
        "events": list(events),
        "field_noise_s": round(field, 4),
        "form_sd_s": round(form_sd, 4),
        "form_note": (
            f"within-pair gap SD {np.sqrt(within):.3f} s, sampling SD "
            f"{np.sqrt(float(pg['samp'].mean())):.3f} s over {len(pg)} team-races"
        ),
        "settings": {
            "min_laps": MIN_LAPS,
            "pace_prior_races": PACE_PRIOR_RACES,
            "clean_gap_s": CLEAN_GAP_S,
            "noise_prior_laps": NOISE_PRIOR_LAPS,
            "max_residual_s": MAX_RESIDUAL_S,
        },
        "drivers": drivers,
    }


def load_pace(configs: Path) -> dict[str, Any] | None:
    path = configs / PACE_FILE
    if not path.exists():
        return None
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data
