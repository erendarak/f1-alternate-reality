"""Hidden race-day truth (H3, D-058): the race runs on one tyre model, the teams plan with another.

``with_hidden_truth`` turns a data-backed setup into a race with a hidden truth:

- ``HISTORICAL``: the truth is the race as it was (its own tyre-life fit, the setup's lap model
  unchanged); the teams know the race-day prior from ``configs/calibration/race_day_prior_v1.json``
  (history strictly before the race, ``calibration/race_day_prior``).
- ``DRAWN``: a new truth drawn around the prior (keyed by ``truth_seed`` and the event), with the
  surprise size measured on history for the race's era: a life factor shared by the compounds,
  a per-compound part (2018+), and the wear with its measured correlation to the life factor.
  For alternate worlds and training: many truths per race, none of them known in advance.

Only the dry compounds' life and wear differ between truth and knowledge: both are the setup's
lap model scaled (life laps are proportional to ``life_raw``, the wear terms to the wear
multiplier, ``calibration/lap_model_fit.life_lap_model``), so everything else stays identical.
"""

import json
import math
from enum import StrEnum
from functools import cache
from pathlib import Path
from typing import Any

from f1ar.calibration.race_day_prior import FIRST_COMPOUND_SEASON, PRIOR_FILE
from f1ar.domain.provenance import SourceType
from f1ar.domain.race_state import (
    DRY_COMPOUNDS,
    LapModelParams,
    RaceSetup,
    TyreKnowledge,
)
from f1ar.simulation.rng_streams import normal

TREND_SD = {"2000-2009": 0.020, "2010-2017": 0.013, "2018-2025": 0.010}
"""Race-to-race spread of the track trend beyond the fitted average, s per lap (EXP-028, D-059;
estimated_from_data; 2000-2009 is an upper bound)."""


class TruthMode(StrEnum):
    HISTORICAL = "historical"
    DRAWN = "drawn"


@cache
def load_prior(configs: Path) -> dict[str, Any] | None:
    path = configs / PRIOR_FILE
    if not path.exists():
        return None
    return dict(json.loads(path.read_text(encoding="utf-8")))


def scaled(lap_model: LapModelParams, life: dict[str, float], wear: float) -> LapModelParams:
    """``lap_model`` with each dry compound's life x ``life[c]`` and its wear terms x ``wear``."""
    compounds = dict(lap_model.compounds)
    for c in DRY_COMPOUNDS:
        p = compounds[c]
        compounds[c] = p.model_copy(
            update={
                "life_laps": None if p.life_laps is None else p.life_laps * life[c.value],
                "deg_s_per_lap": p.deg_s_per_lap * wear,
                "deg_quad_s_per_lap2": p.deg_quad_s_per_lap2 * wear,
            }
        )
    return lap_model.model_copy(update={"compounds": compounds})


def drawn_truth(
    entry: dict[str, Any], surprise: dict[str, Any], event_id: str, truth_seed: int
) -> tuple[dict[str, float], float]:
    """A truth around the prior: ``(life_raw per compound, wear multiplier)``."""
    s = surprise
    z0 = normal(truth_seed, "race_day_truth", event_id, "life")
    z1 = normal(truth_seed, "race_day_truth", event_id, "wear")
    per_compound = entry["season"] >= FIRST_COMPOUND_SEASON
    life = {}
    for c, v in entry["prior"]["life_raw"].items():
        zc = normal(truth_seed, "race_day_truth", event_id, c) if per_compound else 0.0
        life[c] = v * math.exp(s["life_shared_sd"] * z0 + s["life_compound_sd"] * zc)
    rho = float(s["life_wear_correlation"])
    zw = rho * z0 + math.sqrt(max(0.0, 1.0 - rho * rho)) * z1
    wear = entry["prior"]["wear_multiplier"] * math.exp(s["wear_sd"] * zw)
    return life, wear


def with_hidden_truth(
    setup: RaceSetup,
    configs: Path,
    *,
    mode: TruthMode = TruthMode.HISTORICAL,
    truth_seed: int = 0,
    oracle: tuple[str, ...] = (),
    learning: bool = False,
    track_trend: bool = False,
    per_team: bool = False,
) -> RaceSetup:
    """The race with a hidden truth and the teams on the prior (``oracle`` cars know the truth).
    ``learning`` (A1): the teams also learn in the race from the timing screen. ``track_trend``
    (A1): a hidden per-race lap-time trend drawn with the era's spread (EXP-028). ``per_team``
    (A1b): each team gets its own practice measurement and reads its own cars more precisely.
    Raises ``KeyError`` for a race without a prior (no tyre-life fit, e.g. wet, or no history)."""
    prior_cfg = load_prior(configs)
    if prior_cfg is None:
        raise KeyError(f"{PRIOR_FILE.as_posix()} missing: run scripts/fit_race_day_prior.py")
    entry = prior_cfg["events"][setup.event_id]
    hist_life = entry["truth"]["life_raw"]
    hist_wear = float(entry["truth"]["wear_multiplier"])
    prior_life = entry["prior"]["life_raw"]
    prior_wear = float(entry["prior"]["wear_multiplier"])
    era_surprise = prior_cfg["surprise"]["eras"][entry["era"]]
    if mode is TruthMode.HISTORICAL:
        truth_model = setup.lap_model
        true_life, true_wear = hist_life, hist_wear
        how = "the race's own tyre-life fit (as it was)"
    else:
        true_life, true_wear = drawn_truth(entry, era_surprise, setup.event_id, truth_seed)
        truth_model = scaled(
            setup.lap_model,
            {c: true_life[c] / hist_life[c] for c in hist_life},
            true_wear / hist_wear,
        )
        how = (
            f"drawn around the prior (truth seed {truth_seed}; {entry['era']} surprise: life sd "
            f"{era_surprise['life_shared_sd']}, per compound {era_surprise['life_compound_sd']}, "
            f"wear sd {era_surprise['wear_sd']})"
        )
    knowledge_model = scaled(
        setup.lap_model,
        {c: prior_life[c] / hist_life[c] for c in hist_life},
        prior_wear / hist_wear,
    )
    lives = ", ".join(f"{c[0]} {true_life[c]:.1f}/{prior_life[c]:.1f}" for c in prior_life)
    note = (
        f"estimated_from_data: race-day prior {PRIOR_FILE.as_posix()} from "
        f"{entry['track_years']} earlier season(s) at {entry['circuit_id']} and "
        f"{entry['season_races_before']} earlier round(s) of {entry['season']}; truth {how}; "
        f"stint life truth/prior {lives}; wear {true_wear:.2f}/{prior_wear:.2f}"
        + (f"; oracle cars: {', '.join(oracle)}" if oracle else "")
    )
    knowledge = TyreKnowledge(
        source_type=SourceType.ESTIMATED_FROM_DATA,
        note=note,
        lap_model=knowledge_model.model_copy(update={"note": "race-day prior: " + note}),
        wear_scale=prior_wear / true_wear,
        oracle_drivers=tuple(oracle),
        learning=learning,
        life_shared_sd=float(era_surprise["life_shared_sd"]),
        life_compound_sd=float(era_surprise["life_compound_sd"]),
        wear_sd=float(era_surprise["wear_sd"]),
        trend_sd=TREND_SD[entry["era"]],
    )
    if track_trend:
        trend = TREND_SD[entry["era"]] * normal(truth_seed, "track_trend", setup.event_id)
        truth_model = truth_model.model_copy(update={"track_trend_s_per_lap": trend})
        note += f"; hidden track trend {trend:+.4f} s/lap (EXP-028 era sd)"
        knowledge = knowledge.model_copy(update={"note": note})
    out = setup.model_copy(
        update={
            "lap_model": truth_model,
            "knowledge": knowledge,
            "inputs": {**setup.inputs, "race_day_truth": f"{mode.value}: {note}"},
        }
    )
    return with_practice(out, truth_seed) if per_team else out


PRACTICE_LIFE_SD = 0.20
"""A1b: spread of a team's practice reading of log tyre life per compound (EDITORIAL: a few
long runs on light-ish fuel; about the prior's own spread, so practice halves the variance)."""
PRACTICE_WEAR_SD = 0.6
"""A1b: spread of a team's practice reading of log wear (EDITORIAL: wear is hard to read from a
single long run, R-bloggers FP2 analysis in docs/PLAN_HIDDEN_INFO_ERAS.md)."""


def with_practice(setup: RaceSetup, truth_seed: int) -> RaceSetup:
    """A1b (D-062): each team's practice measurement (keyed by truth seed, event and team) and
    its starting estimate (prior + practice)."""
    from f1ar.domain.race_state import TyreEstimate
    from f1ar.policy.race_estimator import belief_model, estimate

    k = setup.knowledge
    assert k is not None
    true_wear = 1.0 / k.wear_scale
    practice = {}
    for team in dict.fromkeys(d.team_id for d in setup.drivers):
        life = {}
        for c in DRY_COMPOUNDS:
            t, p = setup.lap_model.compounds[c].life_laps, k.lap_model.compounds[c].life_laps
            ratio = (t / p) if t is not None and p is not None else 1.0
            z = normal(truth_seed, "practice", setup.event_id, team, c.value)
            life[c] = ratio * math.exp(PRACTICE_LIFE_SD * z)
        zw = normal(truth_seed, "practice", setup.event_id, team, "wear")
        wear = true_wear * math.exp(PRACTICE_WEAR_SD * zw)
        practice[team] = TyreEstimate(
            observations=0,
            wear_factor=round(wear, 4),
            wear_factor_sd=round(PRACTICE_WEAR_SD * wear, 4),
            trend_s_per_lap=0.0,
            life_factor={c: round(v, 4) for c, v in life.items()},
            life_factor_sd=dict.fromkeys(DRY_COMPOUNDS, PRACTICE_LIFE_SD),
        )
    k = k.model_copy(update={"per_team": True, "practice": practice})
    setup = setup.model_copy(update={"knowledge": k})
    estimates = {team: estimate(setup, (), team) for team in practice}
    k = k.model_copy(
        update={
            "team_estimates": estimates,
            "team_belief_models": {t: belief_model(k, e) for t, e in estimates.items()},
            "note": k.note + "; per-team practice (editorial spreads) and private data",
        }
    )
    return setup.model_copy(update={"knowledge": k})


def with_oracle(setup: RaceSetup, oracle: tuple[str, ...]) -> RaceSetup:
    """The same hidden-truth race with ``oracle`` cars planning on the truth."""
    if setup.knowledge is None:
        raise ValueError("no hidden truth in this setup")
    return setup.model_copy(
        update={"knowledge": setup.knowledge.model_copy(update={"oracle_drivers": oracle})}
    )


RIVAL_RULES_FILE = Path("calibration") / "rival_rules_2024_v1.json"


def with_rival_rules(setup: RaceSetup, configs: Path) -> RaceSetup:
    """A1b (D-062): rule teams that undercut and cover, windows fitted on 2024
    (``scripts/fit_rival_rules.py``); the same windows in every season (labelled)."""
    from f1ar.domain.race_state import RivalRules

    path = configs / RIVAL_RULES_FILE
    if not path.exists():
        raise KeyError(f"{RIVAL_RULES_FILE.as_posix()} missing: run scripts/fit_rival_rules.py")
    cfg = json.loads(path.read_text(encoding="utf-8"))
    note = (
        f"estimated_from_data: windows {cfg['fitted']} fitted to the 2024 undercut attempts "
        f"(EXP-020 detector); gaps editorial {cfg['editorial']}"
        + ("" if setup.rules.season == 2024 else f" [2024 windows applied to {setup.rules.season}]")
    )
    rules = RivalRules(
        source_type=SourceType.ESTIMATED_FROM_DATA, note=note, **cfg["fitted"], **cfg["editorial"]
    )
    return setup.model_copy(
        update={"rival_rules": rules, "inputs": {**setup.inputs, "rival_rules": note}}
    )
