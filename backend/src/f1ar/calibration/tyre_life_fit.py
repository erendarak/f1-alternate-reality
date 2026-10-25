"""Tyre life and wear from 2024 race laps (W1, engine v0.10.0, D-041).

Three things are estimated from the lap dataset (``laps_2024_v1``):

1. **Wear on observed race laps, same-lap identification.** A regression per race (and pooled
   over the dry_train races) with a term for every lap of the race (absorbs fuel burn-off,
   track evolution and anything else the whole field shares on that lap), a term per driver,
   and per compound a linear + quadratic curve in tyre age, plus a dummy for the first flying
   lap of a new set after a stop (the fresh-tyre peak). Compound *offsets* come out near zero
   on race laps (drivers drive to a target pace and manage the softer tyre): they are not used
   as the tyres' potential pace, which is a global (``delta``) fitted on behaviour instead.
2. **Raw tyre life per compound and race**: the 75th percentile of how many laps a set had done
   (age at fitting + laps in the stint) when the car pitted under green, over stints of at
   least ``MIN_STINT`` laps. Compounds a race didn't use (or used too little) are filled from
   the race's other compounds with the season's median life ratios.
3. The fresh-tyre peak (pooled dummy) and per-race wear multipliers (shrunk toward 1).

The *globals* ``kappa`` (life = kappa x raw life) and ``delta`` (the potential pace step
between compounds) are fitted afterwards by matching the simulated field's behaviour (stops per
finisher, compound use) on the dry_train races (``scripts/fit_tyre_life.py``). Pace-mode
trade-offs, the cliff and the life used under SC/VSC are editorial (research ranges,
``docs/RESEARCH_REAL_F1.md``).
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from statistics import median
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import lsqr

from f1ar.domain.refuelling import FUEL_S_PER_LAP, LAST_REFUELLING_SEASON

DRY = ("SOFT", "MEDIUM", "HARD")
MIN_STINT = 5
LIFE_QUANTILE = 0.75
MIN_STINTS = 3
SHRINK_STINTS = 5.0
"""Weight (in stints) of the implied value when shrinking a race's measured life."""
SHRINK_LAPS = 400.0
"""Per-race wear multiplier: ``(n k_race + SHRINK_LAPS) / (n + SHRINK_LAPS)``."""
WEAR_MULTIPLIER_FLOOR = 0.2

# editorial (research ranges, docs/RESEARCH_REAL_F1.md section 2.1)
PACE_MODES = {
    "CONSERVE": {"offset_s": 0.35, "wear_multiplier": 0.75},
    "BALANCED": {"offset_s": 0.0, "wear_multiplier": 1.0},
    "PUSH": {"offset_s": -0.30, "wear_multiplier": 1.4},
}
CLIFF_S_PER_LAP = 0.5
NEUTRALISED_LIFE_FACTOR = 0.35
"""TUM's race simulation: degradation x0.25 behind the SC, x0.5 under VSC."""


def green_laps(laps: pd.DataFrame) -> pd.DataFrame:
    """Dry-compound racing laps under green, no pit in/out lap, no lap 1, no obvious outlier,
    not within 1 s of the car ahead (dirty air)."""
    g = laps[
        (laps["track_status_start"].astype(str) == "1")
        & ~laps["pit_in_lap"].astype(bool)
        & ~laps["pit_out_lap"].astype(bool)
        & (laps["lap_number"] > 1)
        & laps["compound"].isin(DRY)
        & (laps["gap_ahead_s"].fillna(99.0) > 1.0)
    ].copy()
    g["age"] = g["tyre_age_start"] + g["stint_lap"] - 1
    # H2a (D-054): FastF1 feed gaps leave a stint's tyre age unknown (2022 Austria: 105 laps)
    g = g[g["age"].notna()]
    # H1b: before 2010 the fuel on board falls within each stint as the tyres age: take its
    # (editorial) effect off, or the wear slope would absorb the fuel burn-off
    if "season" in g:
        refuel = g["season"].astype(int) <= LAST_REFUELLING_SEASON
        if refuel.any():
            fuel = refuel * FUEL_S_PER_LAP * g["laps_remaining"].astype(float)
            g["lap_time_s"] = g["lap_time_s"] - fuel
    med = g.groupby("event_id")["lap_time_s"].transform("median")
    g = g[g["lap_time_s"] < med * 1.07]
    g["fresh"] = ((g["stint_lap"] == 2) & (g["stint"] > 1)).astype(float)
    return g


@dataclass(frozen=True)
class WearFit:
    lin: dict[str, float]
    quad: dict[str, float]
    """Per lap of age squared."""
    fresh_bonus_s: float
    offsets: dict[str, float]
    """Same-lap race offsets vs MEDIUM (diagnostic only)."""
    laps: int


def fit_wear(g: pd.DataFrame) -> WearFit:
    """The same-lap regression on ``green_laps`` of one or more races."""
    g = g.reset_index(drop=True)
    n = len(g)
    lap_key = pd.factorize(g["event_id"].astype(str) + ":" + g["lap_number"].astype(str))[0]
    drv_key = pd.factorize(g["event_id"].astype(str) + ":" + g["driver_code"].astype(str))[0]
    blocks = [
        sp.csr_matrix((np.ones(n), (np.arange(n), lap_key))),
        sp.csr_matrix((np.ones(n), (np.arange(n), drv_key)))[:, 1:],
    ]
    names: list[str] = []
    cols: list[np.ndarray] = []
    present = [c for c in DRY if (g["compound"] == c).sum() > 30]
    for c in present:
        m = (g["compound"] == c).to_numpy(dtype=float)
        if c != "MEDIUM" and "MEDIUM" in present:
            cols.append(m)
            names.append(f"off:{c}")
        age = g["age"].to_numpy(dtype=float)
        cols += [m * age, m * age**2]
        names += [f"lin:{c}", f"quad:{c}"]
    cols.append(g["fresh"].to_numpy(dtype=float))
    names.append("fresh")
    X = sp.hstack([*blocks, sp.csr_matrix(np.column_stack(cols))]).tocsr()
    b = lsqr(X, g["lap_time_s"].to_numpy(dtype=float), atol=1e-12, btol=1e-12, iter_lim=50_000)[0]
    coef = dict(zip(names, b[-len(names) :], strict=True))
    return WearFit(
        lin={c: float(coef[f"lin:{c}"]) for c in present},
        quad={c: float(coef[f"quad:{c}"]) for c in present},
        fresh_bonus_s=float(coef["fresh"]),
        offsets={c: float(coef.get(f"off:{c}", 0.0)) for c in present},
        laps=n,
    )


def stint_lives(laps: pd.DataFrame) -> pd.DataFrame:
    """One row per stint that ended with a green-flag stop: compound and the set's age then."""
    rows = laps.sort_values(["event_id", "driver_code", "lap_number"])
    st = (
        rows.groupby(["event_id", "driver_code", "stint"])
        .agg(
            compound=("compound", "first"),
            laps=("lap_number", "count"),
            age0=("tyre_age_start", "first"),
            pitted=("pit_in_lap", "max"),
            last_status=("track_status_start", "last"),
        )
        .reset_index()
    )
    st = st[
        st["pitted"].astype(bool)
        & (st["last_status"].astype(str) == "1")
        & st["compound"].isin(DRY)
        & (st["laps"] >= MIN_STINT)
    ].copy()
    st["life"] = st["laps"] + st["age0"].fillna(0)
    return st


@dataclass
class EventTyres:
    event_id: str
    total_laps: int
    life_raw: dict[str, float]
    life_source: dict[str, str]
    stints: dict[str, int]
    wear_multiplier: float
    wear_multiplier_raw: float
    same_lap_offsets: dict[str, float] = field(default_factory=dict)


def life_ratios(st: pd.DataFrame) -> dict[str, float]:
    """Median life of each compound relative to MEDIUM in races that used both."""
    q = st.groupby(["event_id", "compound"])["life"].quantile(LIFE_QUANTILE).unstack()
    out = {"MEDIUM": 1.0}
    for c in ("SOFT", "HARD"):
        both = q[[c, "MEDIUM"]].dropna() if c in q and "MEDIUM" in q else pd.DataFrame()
        out[c] = float((both[c] / both["MEDIUM"]).median()) if len(both) else 1.0
    return out


def with_reference(season: WearFit, reference: dict[str, Any]) -> tuple[WearFit, dict[str, float]]:
    """H2b (D-055): a season without compound data (2010-2017: every lap is the MEDIUM
    placeholder) measures one wear curve and one stint life; the soft and hard tyres take the
    reference season's ratios to the medium (borrowed, labelled in the config)."""
    ref = reference["season_wear"]
    lin, quad = dict(season.lin), dict(season.quad)
    for c in DRY:
        if c != "MEDIUM":
            lin[c] = season.lin["MEDIUM"] * float(ref["lin"][c]) / float(ref["lin"]["MEDIUM"])
            quad[c] = season.quad["MEDIUM"] * float(ref["quad"][c]) / float(ref["quad"]["MEDIUM"])
    filled = WearFit(lin, quad, season.fresh_bonus_s, dict.fromkeys(DRY, 0.0), season.laps)
    ratios = {c: float(v) for c, v in reference["life_ratios_vs_medium"].items()}
    return filled, ratios


def fit_events(
    laps: pd.DataFrame,
    events: Sequence[str],
    train: Sequence[str],
    reference: dict[str, Any] | None = None,
) -> tuple[WearFit, dict[str, float], list[EventTyres]]:
    """The season wear shape (pooled over ``train``), life ratios and every event's values;
    ``reference``: another season's tyre-life config for compound ratios (H2b)."""
    g_all = green_laps(laps)
    season = fit_wear(g_all[g_all["event_id"].isin(train)])
    st = stint_lives(laps)
    ratios = life_ratios(st[st["event_id"].isin(train)])
    if reference is not None:
        season, ratios = with_reference(season, reference)
    rate_compounds = ("MEDIUM",) if reference is not None else ("MEDIUM", "HARD")
    season_rate = sum(season.lin[c] for c in rate_compounds) / len(rate_compounds)
    out = []
    for ev in events:
        race = laps[laps["event_id"] == ev]
        total = int(race["scheduled_laps"].iloc[0])
        mine = st[st["event_id"] == ev]
        counts = {str(c): int(n) for c, n in mine.groupby("compound").size().items()}
        quant = {
            str(c): float(v)
            for c, v in mine.groupby("compound")["life"].quantile(LIFE_QUANTILE).items()
        }
        measured = {c: quant[c] for c in quant if counts[c] >= MIN_STINTS}
        # what the race's measured compounds imply for a medium (via the season's ratios)
        if measured:
            med_equiv = float(median([measured[c] / ratios[c] for c in measured]))
        else:
            med_equiv = 0.28 * total  # season median medium stint share (research doc)
        raw: dict[str, float] = {}
        source: dict[str, str] = {}
        for c in DRY:
            implied = med_equiv * ratios[c]
            if c in measured:
                n = counts[c]  # shrink thin estimates toward the implied value
                raw[c] = (n * measured[c] + SHRINK_STINTS * implied) / (n + SHRINK_STINTS)
                source[c] = (
                    f"q{int(LIFE_QUANTILE * 100)} of {n} stints, shrunk with weight "
                    f"{SHRINK_STINTS} toward the race's other compounds x season ratio"
                )
            else:
                raw[c] = implied
                source[c] = "filled from the race's other compounds x season ratio"
        g = g_all[g_all["event_id"] == ev]
        k_raw = 1.0
        offsets: dict[str, float] = {}
        if len(g) > 200:
            wf = fit_wear(g)
            rates = [wf.lin[c] for c in ("MEDIUM", "HARD") if c in wf.lin]
            if rates and season_rate > 0:
                k_raw = float(np.mean(rates)) / season_rate
            offsets = wf.offsets
        n = len(g)
        k = max(WEAR_MULTIPLIER_FLOOR, (n * k_raw + SHRINK_LAPS) / (n + SHRINK_LAPS))
        out.append(
            EventTyres(
                event_id=ev,
                total_laps=total,
                life_raw={c: round(raw[c], 2) for c in DRY},
                life_source=source,
                stints={c: int(counts.get(c, 0)) for c in DRY},
                wear_multiplier=round(k, 3),
                wear_multiplier_raw=round(k_raw, 3),
                same_lap_offsets={c: round(v, 3) for c, v in offsets.items()},
            )
        )
    return season, ratios, out


def to_json(
    season: WearFit,
    ratios: dict[str, float],
    events: list[EventTyres],
    train: Sequence[str],
    *,
    kappa: float,
    delta: float,
    data_sha: str,
    year: int = 2024,
    reference: str | None = None,
) -> dict[str, Any]:
    return {
        "label": f"{year} tyre life + same-lap wear (W1, D-041)",
        "compound_reference": reference,
        "dataset_sha256": data_sha,
        "fitted_on": sorted(train),
        "globals": {
            "kappa": kappa,
            "kappa_note": "life_laps = kappa x life_raw; fitted on dry_train behaviour",
            "delta_s": delta,
            "delta_note": "potential pace step: SOFT -delta, MEDIUM 0, HARD +delta; fitted",
        },
        "season_wear": {
            "lin": season.lin,
            "quad": season.quad,
            "fresh_bonus_s": season.fresh_bonus_s,
            "same_lap_offsets": season.offsets,
            "laps": season.laps,
        },
        "life_ratios_vs_medium": ratios,
        "editorial": {
            "pace_modes": PACE_MODES,
            "cliff_s_per_lap": CLIFF_S_PER_LAP,
            "neutralised_life_factor": NEUTRALISED_LIFE_FACTOR,
            "note": "editorial_assumption: research ranges (docs/RESEARCH_REAL_F1.md)",
        },
        "events": {
            e.event_id: {
                "total_laps": e.total_laps,
                "life_raw": e.life_raw,
                "life_source": e.life_source,
                "stints": e.stints,
                "wear_multiplier": e.wear_multiplier,
                "wear_multiplier_raw": e.wear_multiplier_raw,
                "same_lap_offsets": e.same_lap_offsets,
            }
            for e in events
        },
    }
