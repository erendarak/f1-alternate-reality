"""Driver personas re-fitted from 2024 data where they can be measured (W5, D-045).

The persona (``DriverArchetype``) is editorial. Some of its fields describe things the 2024 laps
can measure; for those, a data-backed race uses a value estimated from data instead, and says
so (``DriverEntry.persona_estimates``):

- ``tire_management``: tyre wear vs the teammate and the team's wear vs the field, from a
  regression per race (a term per lap of the race, an intercept per stint, an age slope per
  driver or per team). Only *differences* of slopes are identified. Odd vs even races agree only
  weakly (all 21 dry races: drivers r = 0.35; the 16 dry_train races: drivers r = 0.07, teams
  r = -0.03), so each is shrunk by its season reliability
  (Spearman-Brown, ``2r / (1 + r)``) and turned into a wear multiplier ``1 + shrunk slope
  difference / the season's medium wear slope``, then into the persona scale of the engine's wear
  factor (``tyre_mgmt_wear_max .. min``), clipped to it.
  **Applied only if the drivers' reliability is at least** ``MIN_RELIABILITY``: on the 16
  dry_train races it is 0.13 (r = 0.07; EXP-014), so ``tire_management`` stays editorial.
- ``consistency``: W2's lap noise per driver (``driver_pace_2024_v1``).
- ``overtake_skill`` / ``defense_skill``: W3's attack / defend terms (``racecraft_2024_v1``).
  These three are **rank-mapped** onto the field's persona values (the data orders the drivers;
  the spread stays the persona's), because their engine effects (incidents, the AI's
  observation) are calibrated on that spread.

Not measurable from 2024 timing, so they stay editorial: mistake proneness (blue flags dominate
the big losses; 24 events left; odd vs even races r = -0.33), composure, wet skill (3 wet
races), risk appetite (a preference), qualifying skill (the grid is real), and the Stage 2/3
fields.
"""

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import lsqr

from f1ar.calibration.season_files import REFERENCE_SEASON, own_season
from f1ar.calibration.tyre_life_fit import green_laps
from f1ar.domain.driver_profiles import ProfileSet
from f1ar.domain.identities import RaceRoster

PERSONA_FILE = Path("calibration") / "personas_2024_v1.json"
MIN_LAPS = 15
WEAR_MULTIPLIER_RANGE = (0.75, 1.25)
MIN_RELIABILITY = 0.3
"""A data estimate replaces a persona field only if its season reliability is at least this."""


def _slopes(g: pd.DataFrame, group: str) -> dict[str, float]:
    """Age slope per ``group`` (driver or team) relative to the race mean: lap terms, stint
    intercepts, one slope per group."""
    g = g.reset_index(drop=True)
    n = len(g)
    lap = pd.factorize(g["lap_number"])[0]
    keys, k = np.unique(g[group].astype(str).to_numpy(), return_inverse=True)
    stint = pd.factorize(g["driver_code"].astype(str) + ":" + g["stint"].astype(str))[0]
    age = g["age"].to_numpy(dtype=float)
    X = sp.hstack(
        [
            sp.csr_matrix((np.ones(n), (np.arange(n), lap))),
            sp.csr_matrix((np.ones(n), (np.arange(n), stint)))[:, 1:],
            sp.csr_matrix((age, (np.arange(n), k))),
        ]
    ).tocsr()
    b = lsqr(X, g["lap_time_s"].to_numpy(dtype=float), atol=1e-12, btol=1e-12, iter_lim=50_000)[0]
    s = b[-len(keys) :]
    return {str(key): float(v - s.mean()) for key, v in zip(keys, s, strict=True)}


def split_half(values: list[tuple[str, int, float]]) -> float:
    """Correlation of per-key means over odd vs even races."""
    df = pd.DataFrame(values, columns=["key", "half", "v"])
    m = df.groupby(["key", "half"])["v"].mean().unstack().dropna()
    if len(m) < 3:
        return 0.0
    return float(np.corrcoef(m[0], m[1])[0, 1])


def fit_wear(laps: pd.DataFrame, events: Sequence[str], reference_slope: float) -> dict[str, Any]:
    drv: list[tuple[str, int, float]] = []
    team: list[tuple[str, int, float]] = []
    team_of: dict[str, str] = {}
    for i, ev in enumerate(events):
        g = green_laps(laps[laps["event_id"] == ev])
        if len(g) < 200:
            continue
        counts = g.groupby("driver_code").size()
        teams = g.groupby("driver_code")["team_id"].first()
        team_of.update({str(d): str(t) for d, t in teams.items()})
        per_driver = _slopes(g, "driver_code")
        for _, members in teams.groupby(teams):
            pair = [str(d) for d in members.index if counts.get(d, 0) >= MIN_LAPS]
            if len(pair) == 2:
                mean = (per_driver[pair[0]] + per_driver[pair[1]]) / 2
                drv += [(d, i % 2, per_driver[d] - mean) for d in pair]
        team += [(t, i % 2, v) for t, v in _slopes(g, "team_id").items()]
    r_d, r_t = split_half(drv), split_half(team)
    rel_d = max(0.0, 2 * r_d / (1 + r_d)) if r_d > 0 else 0.0
    rel_t = max(0.0, 2 * r_t / (1 + r_t)) if r_t > 0 else 0.0
    dmean = (
        pd.DataFrame(drv, columns=["key", "half", "v"]).groupby("key")["v"].agg(["mean", "size"])
    )
    tmean = pd.DataFrame(team, columns=["key", "half", "v"]).groupby("key")["v"].mean()
    lo, hi = WEAR_MULTIPLIER_RANGE
    drivers = {}
    for d, row in dmean.iterrows():
        t = team_of.get(str(d), "?")
        slope = rel_d * float(row["mean"]) + rel_t * float(tmean.get(t, 0.0))
        drivers[str(d)] = {
            "team_id": t,
            "wear_multiplier": round(float(np.clip(1.0 + slope / reference_slope, lo, hi)), 4),
            "vs_teammate_s_per_lap2": round(float(row["mean"]), 5),
            "team_vs_field_s_per_lap2": round(float(tmean.get(t, 0.0)), 5),
            "races": int(row["size"]),
        }
    return {
        "reference_slope_s_per_lap2": reference_slope,
        "split_half_drivers": round(r_d, 3),
        "split_half_teams": round(r_t, 3),
        "reliability_drivers": round(rel_d, 3),
        "reliability_teams": round(rel_t, 3),
        "drivers": drivers,
    }


def rank_map(
    measure: Mapping[str, float], persona: Mapping[str, float], *, higher_is_better: bool
) -> dict[str, float]:
    """Give the drivers the persona values of the field in the order of ``measure``."""
    common = [d for d in persona if d in measure]
    ordered = sorted(common, key=lambda d: measure[d], reverse=higher_is_better)
    values = sorted((persona[d] for d in common), reverse=True)
    return dict(zip(ordered, values, strict=True))


def refit_profiles(
    profiles: ProfileSet,
    roster: RaceRoster,
    personas: Mapping[str, Any] | None,
    pace: Mapping[str, Any] | None,
    racecraft: Mapping[str, Any] | None,
    wear_max: float,
    wear_min: float,
) -> tuple[ProfileSet, dict[str, dict[str, str]]]:
    """The race's copy of the profiles with the measurable fields estimated from data, and per
    driver which fields were replaced (with a short note)."""
    code = {e.driver_id: e.code for e in roster.entries}
    by_code = {e.code: e.driver_id for e in roster.entries}
    new: dict[str, dict[str, float]] = {p.driver_id: {} for p in profiles.profiles}
    notes: dict[str, dict[str, str]] = {p.driver_id: {} for p in profiles.profiles}
    arche = {p.driver_id: p.archetype for p in profiles.profiles}
    wear_ok = personas is not None and personas["wear"]["reliability_drivers"] >= MIN_RELIABILITY
    if personas is not None and wear_ok:
        for d, v in personas["wear"]["drivers"].items():
            if d in by_code and by_code[d] in new:
                m = float(v["wear_multiplier"])
                tm = float(np.clip((wear_max - m) / (wear_max - wear_min), 0.0, 1.0))
                new[by_code[d]]["tire_management"] = round(tm, 4)
                notes[by_code[d]]["tire_management"] = (
                    f"estimated_from_data: wear x{m:.3f} (vs teammate and team, shrunk; W5)"
                )

    def ranked(field: str, measure: dict[str, float], higher_is_better: bool, note: str) -> None:
        persona = {d: float(getattr(arche[d], field)) for d in arche if code.get(d) in measure}
        by_id = {d: measure[code[d]] for d in persona}
        for d, value in rank_map(by_id, persona, higher_is_better=higher_is_better).items():
            new[d][field] = value
            notes[d][field] = note

    if pace is not None:
        noise = {c: float(v["lap_noise_s"]) for c, v in pace["drivers"].items()}
        ranked(
            "consistency",
            noise,
            False,
            "estimated_from_data: rank of W2 lap noise (persona spread)",
        )
    if racecraft is not None:
        att = {c: float(v["attack_s"]) for c, v in racecraft["overtaking"]["drivers"].items()}
        dfd = {c: float(v["defend_s"]) for c, v in racecraft["overtaking"]["drivers"].items()}
        ranked(
            "overtake_skill",
            att,
            True,
            "estimated_from_data: rank of W3 attack term (persona spread)",
        )
        ranked(
            "defense_skill",
            dfd,
            True,
            "estimated_from_data: rank of W3 defend term (persona spread)",
        )
    out = []
    for p in profiles.profiles:
        changes = new[p.driver_id]
        if not changes:
            out.append(p)
            continue
        meta = p.metadata
        fields = tuple(sorted(set(meta.overridden_fields) | set(changes)))
        out.append(
            p.model_copy(
                update={
                    "archetype": p.archetype.model_copy(update=changes),
                    "profile_revision": f"{p.profile_revision}+w5",
                    "metadata": meta.model_copy(
                        update={
                            "overridden_fields": fields,
                            "explanation": f"{meta.explanation} W5 (D-045): "
                            f"{', '.join(sorted(changes))} estimated from {roster.season} data.",
                        }
                    ),
                }
            )
        )
    return profiles.model_copy(update={"profiles": tuple(out)}), notes


def load_personas(configs: Path, season: int = REFERENCE_SEASON) -> dict[str, Any] | None:
    """This season's fit, or None: keyed by driver/team, never borrowed (H2a, D-054)."""
    path = own_season(configs, PERSONA_FILE, season)
    if path is None:
        return None
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data
