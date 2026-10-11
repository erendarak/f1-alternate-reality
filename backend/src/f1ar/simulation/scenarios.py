"""Scenario builders: turn roster + profiles into a ``RaceSetup``."""

from collections.abc import Sequence
from pathlib import Path

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
    RaceSetup,
    TrackSpec,
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
    roster: RaceRoster, profiles: ProfileSet, codes: Sequence[str] | None
) -> RaceSetup:
    """Setup from 3-letter driver codes (e.g. ``["VER", "HAM"]``); ``None`` = full grid."""
    by_code = {e.code: e.driver_id for e in roster.entries}
    wanted = [e.code for e in roster.entries] if codes is None else [c.upper() for c in codes]
    unknown = [c for c in wanted if c not in by_code]
    if unknown:
        raise ValueError(f"unknown driver code(s) {unknown}; choose from {sorted(by_code)}")
    return build_setup(roster, profiles, [by_code[c] for c in wanted])


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


def build_setup(
    roster: RaceRoster,
    profiles: ProfileSet,
    driver_ids: Sequence[str],
    *,
    track: TrackSpec | None = None,
    lap_model: LapModelParams | None = None,
    tick_seconds: float = DEFAULT_TICK_SECONDS,
    start_compound: Compound = Compound.MEDIUM,
) -> RaceSetup:
    """Race setup for a subset of the roster, ordered by grid position."""
    if roster.event_id != profiles.event_id:
        raise ValueError("roster and profiles belong to different events")
    entries = sorted((roster.entry(d) for d in driver_ids), key=lambda e: e.grid_position)
    return RaceSetup(
        event_id=roster.event_id,
        track=track or monza_2024_track(),
        lap_model=lap_model or default_lap_model(),
        tick_seconds=tick_seconds,
        profile_set_manifest_hash=profiles.manifest_hash,
        drivers=tuple(
            DriverEntry(
                driver_id=e.driver_id,
                code=e.code,
                team_id=e.team_id,
                grid_position=e.grid_position,
                start_compound=start_compound,
                profile=profiles.get(e.driver_id),
            )
            for e in entries
        ),
    )
