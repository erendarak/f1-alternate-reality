"""The engine's lap model for one data-backed race (M4 slice 3, D-033).

    tyre/fuel shape   fitted on the 2024 dry_train races (configs/calibration/tyre_shape_*.json)
    race adjustment   wear multiplier + S/H offset deltas from THIS race's laps (in-sample)
    guards            wear step >= 0 per lap (engine), multiplier >= WEAR_MULTIPLIER_FLOOR
    normalisation     wear rates divided by the field's mean persona wear factor, so the
                      average driver of the race wears at the fitted rate (personas are
                      editorial and mostly above 0.5 in tire_management)

Skill gap, lap noise and pace modes stay editorial (D-010); the base lap is re-anchored on
the race afterwards (``race_inputs.estimate_base_lap``) with this model's own wear.
"""

import hashlib
import json
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from statistics import fmean
from typing import Any

import pandas as pd

from f1ar.calibration.pace_models import RaceAdjustment, TyreShape, fit_race_adjustment
from f1ar.domain.driver_profiles import DriverArchetype
from f1ar.domain.provenance import SourceType
from f1ar.domain.race_state import (
    DRY_COMPOUNDS,
    Compound,
    CompoundParams,
    LapModelParams,
    PaceMode,
    PaceModeParams,
)
from f1ar.simulation.lap_model import tyre_wear_factor

SHAPE_FILE = Path("calibration") / "tyre_shape_2024_v1.json"
WEAR_MULTIPLIER_FLOOR = 0.1
"""No track gets less than 10 % of the season's wear rate (Qatar's fit was negative, -0.24)."""


@cache
def load_shape(configs_dir: Path) -> tuple[TyreShape, str]:
    """The fitted season shape and its file hash."""
    text = (configs_dir / SHAPE_FILE).read_text(encoding="utf-8")
    return TyreShape.from_json(text), "sha256:" + hashlib.sha256(text.encode()).hexdigest()


@dataclass(frozen=True)
class CalibratedLapModel:
    params: LapModelParams
    adjustment: RaceAdjustment
    wear_multiplier_used: float
    field_wear_factor: float
    note: str


def calibrated_lap_model(
    editorial: LapModelParams,
    shape: TyreShape,
    shape_sha: str,
    race_laps: pd.DataFrame | None,
    drivers: list[DriverArchetype],
) -> CalibratedLapModel:
    """``editorial`` supplies the parts that stay editorial (skill, noise, pace modes).
    Without ``race_laps`` the season shape is used as is (no race adjustment)."""
    adjustment = (
        fit_race_adjustment(race_laps, shape) if race_laps is not None else RaceAdjustment(1.0, {})
    )
    k = max(WEAR_MULTIPLIER_FLOOR, adjustment.wear_multiplier)
    field = fmean(tyre_wear_factor(editorial, d) for d in drivers) if drivers else 1.0
    compounds = {c: editorial.compounds[c] for c in Compound if c not in DRY_COMPOUNDS}
    for c in DRY_COMPOUNDS:
        name = c.value
        compounds[c] = CompoundParams(
            offset_s=shape.offset[name] + adjustment.offset_delta.get(name, 0.0),
            deg_s_per_lap=max(0.0, shape.wear_lin[name]) * k / field,
            deg_quad_s_per_lap2=shape.wear_quad[name] * k / field,
        )
    deltas = ", ".join(f"{c} {v:+.3f}" for c, v in adjustment.offset_delta.items())
    note = (
        f"estimated_from_data: tyre/fuel shape {SHAPE_FILE.as_posix()} ({shape_sha[:19]}..., "
        f"fitted on {len(shape.fitted_on)} 2024 dry races) x this race's wear multiplier "
        f"{adjustment.wear_multiplier:.2f} (used {k:.2f}, floor {WEAR_MULTIPLIER_FLOOR}) and "
        f"offset deltas {deltas or 'none'} "
        f"({'in-sample' if race_laps is not None else 'no race tables: season shape only'}), "
        f"wear rates / field persona wear factor "
        f"{field:.3f}; skill gap, noise and pace modes editorial (D-010)"
    )
    params = editorial.model_copy(
        update={
            "source_type": SourceType.ESTIMATED_FROM_DATA,
            "note": note,
            "fuel_s_per_lap_remaining": shape.fuel_s_per_lap,
            "compounds": compounds,
        }
    )
    return CalibratedLapModel(params, adjustment, k, field, note)


# --- W1: tyre life (engine v0.10.0, D-041) ----------------------------------------------------

LIFE_FILE = Path("calibration") / "tyre_life_2024_v1.json"


@cache
def load_life(configs_dir: Path) -> tuple[dict[str, Any], str] | None:
    path = configs_dir / LIFE_FILE
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8")
    return json.loads(text), "sha256:" + hashlib.sha256(text.encode()).hexdigest()


@dataclass(frozen=True)
class LifeLapModel:
    params: LapModelParams
    """What the simulation runs: the tyres' potential pace (``delta`` steps), life, cliff."""
    estimation_params: LapModelParams
    """The same model with the compound offsets seen on race laps (same-lap fit): used to
    read car pace and the base lap off real laps, where drivers drove managed race pace."""
    note: str


def life_lap_model(
    editorial: LapModelParams,
    life_cfg: dict[str, Any],
    life_sha: str,
    event_id: str,
    drivers: list[DriverArchetype],
    fuel_s_per_lap: float,
    *,
    kappa: float | None = None,
    delta: float | None = None,
) -> LifeLapModel:
    """The v0.10.0 lap model of one data-backed race. ``kappa`` / ``delta`` override the
    fitted globals (used while fitting them)."""
    g = life_cfg["globals"]
    k_life = float(g["kappa"] if kappa is None else kappa)
    step = float(g["delta_s"] if delta is None else delta)
    ev = life_cfg["events"].get(event_id)
    season = life_cfg["season_wear"]
    ed = life_cfg["editorial"]
    wear_k = float(ev["wear_multiplier"]) if ev else 1.0
    field = fmean(tyre_wear_factor(editorial, d) for d in drivers) if drivers else 1.0
    potential = {"SOFT": -step, "MEDIUM": 0.0, "HARD": step}
    observed = {c: float(season["same_lap_offsets"].get(c, 0.0)) for c in potential}
    if ev is None:  # a race outside the fit (e.g. wet): the season's median life
        ratios = life_cfg["life_ratios_vs_medium"]
        raw = {c: 0.28 * 60 * float(ratios[c]) for c in potential}
    else:
        raw = {c: float(v) for c, v in ev["life_raw"].items()}

    def build(offsets: dict[str, float]) -> LapModelParams:
        compounds = {c: editorial.compounds[c] for c in Compound if c not in DRY_COMPOUNDS}
        for c in DRY_COMPOUNDS:
            name = c.value
            compounds[c] = CompoundParams(
                offset_s=offsets[name],
                deg_s_per_lap=max(0.0, float(season["lin"][name])) * wear_k / field,
                deg_quad_s_per_lap2=float(season["quad"][name]) * wear_k / field,
                life_laps=k_life * raw[name] * field,
            )
        return editorial.model_copy(
            update={
                "source_type": SourceType.ESTIMATED_FROM_DATA,
                "fuel_s_per_lap_remaining": fuel_s_per_lap,
                "compounds": compounds,
                "pace_modes": {
                    PaceMode(m): PaceModeParams(**v) for m, v in ed["pace_modes"].items()
                },
                "fresh_tyre_bonus_s": float(season["fresh_bonus_s"]),
                "cliff_s_per_lap": float(ed["cliff_s_per_lap"]),
                "neutralised_life_factor": float(ed["neutralised_life_factor"]),
            }
        )

    lives = ", ".join(f"{c[0]} {k_life * raw[c]:.1f}" for c in potential)
    note = (
        f"estimated_from_data: tyre life {LIFE_FILE.as_posix()} ({life_sha[:19]}...): life "
        f"(laps, neutral driver) {lives} = kappa {k_life:.2f} x this race's stint lengths "
        f"({'in-sample' if ev else 'season median'}); wear = same-lap season curve x "
        f"{wear_k:.2f}; potential pace steps +-{step:.2f} s (fitted on dry_train behaviour); "
        f"fresh-tyre peak {float(season['fresh_bonus_s']):+.2f} s; editorial_assumption: "
        f"pace-mode trade-offs, cliff {ed['cliff_s_per_lap']} s/lap, SC/VSC life factor "
        f"{ed['neutralised_life_factor']}; skill gap and noise editorial (D-010)"
    )
    params = build(potential).model_copy(update={"note": note})
    estimation = build(observed).model_copy(update={"note": note + " [estimation variant]"})
    return LifeLapModel(params, estimation, note)
