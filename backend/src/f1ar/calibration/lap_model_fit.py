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
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from statistics import fmean

import pandas as pd

from f1ar.calibration.pace_models import RaceAdjustment, TyreShape, fit_race_adjustment
from f1ar.domain.driver_profiles import DriverArchetype
from f1ar.domain.provenance import SourceType
from f1ar.domain.race_state import DRY_COMPOUNDS, Compound, CompoundParams, LapModelParams
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
