"""Scenario builders: turn roster + profiles into a ``RaceSetup``."""

from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd

from f1ar.domain.driver_profiles import ProfileSet, load_profiles
from f1ar.domain.identities import RaceRoster, load_roster
from f1ar.domain.provenance import SourceType
from f1ar.domain.race_state import (
    Compound,
    CompoundParams,
    DriverEntry,
    LapModelParams,
    PaceMode,
    PaceModeParams,
    PitParams,
    RaceControlParams,
    RaceSetup,
    TrackSpec,
    TrafficParams,
)

DEFAULT_TICK_SECONDS = 0.5
MONZA_2024 = "2024_italian_gp"
_SEED_FILES = {MONZA_2024: "seed_2024_monza_drivers.yaml"}


def load_event(configs_dir: Path, event_id: str = MONZA_2024) -> tuple[RaceRoster, ProfileSet]:
    """Load the verified roster and the persona profiles of one event from ``configs/``."""
    roster = load_roster(configs_dir / "rosters" / f"{event_id}.json", expected_event_id=event_id)
    profiles = load_profiles(
        configs_dir / "drivers" / _SEED_FILES[event_id],
        configs_dir / "drivers" / "archetypes.yaml",
        roster,
    )
    return roster, profiles


def setup_from_codes(
    roster: RaceRoster,
    profiles: ProfileSet,
    codes: Sequence[str] | None,
    *,
    data_dir: Path | None = None,
) -> RaceSetup:
    """Setup from 3-letter driver codes (e.g. ``["VER", "HAM"]``); ``None`` = full grid.

    With ``data_dir`` (a fetched ``data/processed/<event>/``) the setup uses the real start tyres
    and the data-estimated car pace offsets; without it, synthetic defaults.
    """
    by_code = {e.code: e.driver_id for e in roster.entries}
    wanted = [e.code for e in roster.entries] if codes is None else [c.upper() for c in codes]
    unknown = [c for c in wanted if c not in by_code]
    if unknown:
        raise ValueError(f"unknown driver code(s) {unknown}; choose from {sorted(by_code)}")
    driver_ids = [by_code[c] for c in wanted]
    if data_dir is None:
        return build_setup(roster, profiles, driver_ids)

    from f1ar.calibration.race_inputs import (
        estimate_car_offsets,
        estimate_pit_loss,
        start_compounds,
    )

    laps = pd.read_parquet(data_dir / "laps.parquet")
    lap_model = default_lap_model()
    estimate = estimate_car_offsets(laps, roster, profiles, lap_model)
    try:
        pit_loss = estimate_pit_loss(laps)
        pit, pit_note = (
            pit_params_from_total_loss(pit_loss.total_loss_s, pit_loss.note),
            pit_loss.note,
        )
    except ValueError as exc:  # e.g. a race without any complete pit stop
        pit, pit_note = default_pit_params(), f"editorial_assumption: default pit model ({exc})"
    return build_setup(
        roster,
        profiles,
        driver_ids,
        lap_model=lap_model,
        pit=pit,
        start_compounds=start_compounds(laps, roster),
        car_offsets_s=estimate.offsets_s,
        inputs={
            "start_compounds": f"historical_verified: lap-1 compound, {data_dir.name}/laps.parquet",
            "car_pace_offsets": estimate.note,
            "pit_loss": pit_note,
        },
    )


def event_data_dir(root: Path, event_id: str = MONZA_2024) -> Path | None:
    """``data/processed/<event>`` if it has been fetched, else None (synthetic defaults)."""
    path = root / "data" / "processed" / event_id
    return path if (path / "laps.parquet").exists() else None


def monza_2024_track() -> TrackSpec:
    return TrackSpec(
        track_id="monza",
        length_m=5793.0,
        total_laps=53,
        source_note=(
            "Circuit length 5.793 km (published figure); 53 race laps per Jolpica results "
            "for 2024 round 16. Geometry is not used yet."
        ),
    )


def default_lap_model() -> LapModelParams:
    """Placeholder coefficients chosen to look plausible for Monza. NOT calibrated."""
    return LapModelParams(
        source_type=SourceType.EDITORIAL_ASSUMPTION,
        note="Synthetic placeholder coefficients; to be replaced by FastF1 calibration (M1/M4).",
        base_lap_s=82.0,
        skill_gap_s=3.0,
        fuel_s_per_lap_remaining=0.035,
        tyre_mgmt_wear_max=1.3,
        tyre_mgmt_wear_min=0.7,
        noise_floor_s=0.08,
        noise_inconsistency_s=0.6,
        min_lap_fraction=0.9,
        compounds={
            Compound.SOFT: CompoundParams(offset_s=-0.6, deg_s_per_lap=0.09),
            Compound.MEDIUM: CompoundParams(offset_s=0.0, deg_s_per_lap=0.055),
            Compound.HARD: CompoundParams(offset_s=0.4, deg_s_per_lap=0.035),
        },
        pace_modes={
            PaceMode.CONSERVE: PaceModeParams(offset_s=0.6, wear_multiplier=0.7),
            PaceMode.BALANCED: PaceModeParams(offset_s=0.0, wear_multiplier=1.0),
            PaceMode.PUSH: PaceModeParams(offset_s=-0.4, wear_multiplier=1.4),
        },
    )


STANDSTILL_MEDIAN_S = 2.4
"""Typical 2024 standstill; editorial (no stationary times in our data)."""


def default_pit_params() -> PitParams:
    """Synthetic Monza pit model: ~24 s whole stop. NOT calibrated."""
    return PitParams(
        source_type=SourceType.EDITORIAL_ASSUMPTION,
        note="Synthetic placeholder: 21.5 s pit lane + ~2.4 s standstill (log-logistic).",
        lane_loss_s=21.5,
        standstill_median_s=STANDSTILL_MEDIAN_S,
        standstill_shape=9.0,
        standstill_max_s=12.0,
        min_distinct_dry_compounds=2,
    )


def default_traffic_params(*, enabled: bool = True) -> TrafficParams:
    """Synthetic traffic model for a DRS track like Monza. NOT calibrated."""
    return TrafficParams(
        source_type=SourceType.EDITORIAL_ASSUMPTION,
        note="Synthetic placeholders; overtake counts to be compared with real races (EXP-002).",
        enabled=enabled,
        min_gap_s=0.4,
        overtake_threshold_s=0.3,
        overtake_scale_s=0.8,
        attacker_loss_s=0.25,
        defender_loss_s=0.35,
        dirty_air_window_s=1.2,
        dirty_air_loss_s=0.15,
        blue_flag_attacker_loss_s=0.1,
        blue_flag_defender_loss_s=0.6,
    )


def default_race_control_params(*, enabled: bool = True) -> RaceControlParams:
    """Synthetic incident/neutralisation model: about one retirement and 0.6 neutralisations per
    race on average. NOT calibrated."""
    return RaceControlParams(
        source_type=SourceType.EDITORIAL_ASSUMPTION,
        note="Synthetic placeholders; rates to be compared with real 2024 races (EXP-002).",
        enabled=enabled,
        mistake_per_lap=0.012,
        mistake_loss_min_s=0.8,
        mistake_loss_max_s=4.0,
        crash_per_lap=0.0012,
        first_lap_crash_multiplier=6.0,
        failure_per_lap=0.0005,
        push_risk_multiplier=1.6,
        conserve_risk_multiplier=0.6,
        crash_sc_probability=0.65,
        crash_vsc_probability=0.3,
        failure_sc_probability=0.15,
        failure_vsc_probability=0.4,
        vsc_speed_fraction=0.7,
        vsc_min_s=60.0,
        vsc_max_s=160.0,
        sc_min_laps=3,
        sc_max_laps=5,
        sc_lap_factor=1.5,
        sc_catch_up_lap_factor=1.25,
        sc_spacing_m=40.0,
        sc_deploy_ahead_m=150.0,
        neutralised_pit_window_laps=10,
    )


def pit_params_from_total_loss(total_loss_s: float, note: str) -> PitParams:
    """Pit model whose median whole-stop loss equals a data estimate (lane = total - standstill).

    The standstill median and shape stay editorial; only the total is from data.
    """
    return default_pit_params().model_copy(
        update={
            "source_type": SourceType.ESTIMATED_FROM_DATA,
            "note": f"{note}; standstill median {STANDSTILL_MEDIAN_S} s and shape are editorial",
            "lane_loss_s": round(max(total_loss_s - STANDSTILL_MEDIAN_S, 0.0), 2),
        }
    )


def build_setup(
    roster: RaceRoster,
    profiles: ProfileSet,
    driver_ids: Sequence[str],
    *,
    track: TrackSpec | None = None,
    lap_model: LapModelParams | None = None,
    pit: PitParams | None = None,
    traffic: TrafficParams | None = None,
    race_control: RaceControlParams | None = None,
    tick_seconds: float = DEFAULT_TICK_SECONDS,
    start_compound: Compound = Compound.MEDIUM,
    start_compounds: Mapping[str, Compound] | None = None,
    car_offsets_s: Mapping[str, float] | None = None,
    inputs: Mapping[str, str] | None = None,
) -> RaceSetup:
    """Race setup for a subset of the roster, ordered by grid position.

    ``start_compounds`` (per driver) and ``car_offsets_s`` (per team) override the synthetic
    defaults (``start_compound`` for everyone, equal cars); ``inputs`` records where they came from.
    """
    if roster.event_id != profiles.event_id:
        raise ValueError("roster and profiles belong to different events")
    entries = sorted((roster.entry(d) for d in driver_ids), key=lambda e: e.grid_position)
    return RaceSetup(
        event_id=roster.event_id,
        track=track or monza_2024_track(),
        lap_model=lap_model or default_lap_model(),
        pit=pit or default_pit_params(),
        traffic=traffic or default_traffic_params(),
        race_control=race_control or default_race_control_params(),
        tick_seconds=tick_seconds,
        profile_set_manifest_hash=profiles.manifest_hash,
        drivers=tuple(
            DriverEntry(
                driver_id=e.driver_id,
                code=e.code,
                team_id=e.team_id,
                grid_position=e.grid_position,
                start_compound=(start_compounds or {}).get(e.driver_id, start_compound),
                car_pace_offset_s=(car_offsets_s or {}).get(e.team_id, 0.0),
                profile=profiles.get(e.driver_id),
            )
            for e in entries
        ),
        inputs=dict(inputs or {}),
    )
