"""Scenario builders: turn roster + profiles into a ``RaceSetup``."""

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd

from f1ar.domain.driver_profiles import (
    DriverArchetype,
    ProfileSet,
    load_profiles,
    load_season_profiles,
)
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
    WeatherMode,
    WeatherParams,
    default_weather_params,
)
from f1ar.policy.baseline_strategy import POLICY_ID as STRATEGY_POLICY_ID

DEFAULT_TICK_SECONDS = 0.5
MONZA_2024 = "2024_italian_gp"
BELGIUM_2024 = "2024_belgian_gp"
_SEED_FILES = {
    MONZA_2024: "seed_2024_monza_drivers.yaml",
    BELGIUM_2024: "seed_2024_belgian_drivers.yaml",
}
SIMULATED_EVENTS = tuple(_SEED_FILES)
"""Events with their own persona seed file (M2). Since M4 every 2024 race also has personas
through the season seed (``SEASON_SEED_FILES``); see ``has_personas``."""
SEASON_SEED_FILES = {2024: "seed_2024_season_drivers.yaml"}
REPO_CONFIGS = Path(__file__).resolve().parents[4] / "configs"
"""This repository's ``configs/`` (the package is installed from the repo, editable)."""


def has_personas(configs_dir: Path, event_id: str) -> bool:
    """A roster exists and a seed covers it (its own seed, or its season's seed)."""
    roster_path = configs_dir / "rosters" / f"{event_id}.json"
    if not roster_path.exists():
        return False
    season = load_roster(roster_path, expected_event_id=event_id).season
    return event_id in _SEED_FILES or season in SEASON_SEED_FILES


def load_event(configs_dir: Path, event_id: str = MONZA_2024) -> tuple[RaceRoster, ProfileSet]:
    """Load the verified roster and the persona profiles of one event from ``configs/``.

    Monza and Spa use their own seed files (M2); every other race of a season with a season
    seed uses that one (same persona values for the same driver)."""
    roster = load_roster(configs_dir / "rosters" / f"{event_id}.json", expected_event_id=event_id)
    templates = configs_dir / "drivers" / "archetypes.yaml"
    if event_id in _SEED_FILES:
        profiles = load_profiles(configs_dir / "drivers" / _SEED_FILES[event_id], templates, roster)
    elif roster.season in SEASON_SEED_FILES:
        profiles = load_season_profiles(
            configs_dir / "drivers" / SEASON_SEED_FILES[roster.season], templates, roster
        )
    else:
        raise ValueError(f"no persona seed for {event_id} (season {roster.season})")
    return roster, profiles


def setup_from_codes(
    roster: RaceRoster,
    profiles: ProfileSet,
    codes: Sequence[str] | None,
    *,
    data_dir: Path | None = None,
    configs_dir: Path | None = None,
    random_rain: bool = False,
    tyre_life_overrides: dict[str, float] | None = None,
) -> RaceSetup:
    """Setup from 3-letter driver codes (e.g. ``["VER", "HAM"]``); ``None`` = full grid.

    With ``data_dir`` (a fetched ``data/processed/<event>/``) the setup uses the real start tyres,
    the calibrated tyre/fuel model (season shape + this race's adjustment, M4/D-033), and the
    data-estimated car offsets, base lap and pit loss; without it, the editorial placeholders
    (Monza and Spa only). ``tyre_life_overrides`` (``kappa``, ``delta``) replace the tyre-life
    globals: only for fitting them (W1).
    """
    tyre_life_overrides = tyre_life_overrides or {}
    by_code = {e.code: e.driver_id for e in roster.entries}
    wanted = [e.code for e in roster.entries] if codes is None else [c.upper() for c in codes]
    unknown = [c for c in wanted if c not in by_code]
    if unknown:
        raise ValueError(f"unknown driver code(s) {unknown}; choose from {sorted(by_code)}")
    driver_ids = [by_code[c] for c in wanted]
    track = track_for(roster.event_id, data_dir)
    if data_dir is None:
        if roster.event_id not in _NOMINAL_BASE_LAP_S:
            raise ValueError(f"{roster.event_id} can only be simulated with its fetched data")
        return build_setup(
            roster,
            profiles,
            driver_ids,
            track=track,
            lap_model=default_lap_model(_NOMINAL_BASE_LAP_S[roster.event_id]),
        )

    from f1ar.calibration.lap_dataset import race_laps
    from f1ar.calibration.lap_model_fit import (
        SHAPE_FILE,
        calibrated_lap_model,
        life_lap_model,
        load_life,
        load_shape,
    )
    from f1ar.calibration.race_inputs import (
        estimate_base_lap,
        estimate_car_offsets,
        estimate_pit_loss,
        start_compounds,
    )
    from f1ar.calibration.weather_fit import lap_weather_penalty, load_weather, race_weather

    laps_file = data_dir / "laps.parquet"
    raw_laps = pd.read_parquet(laps_file)
    configs = configs_dir or REPO_CONFIGS
    weather = default_weather_params()
    weather_note = "editorial_assumption: no weather data; dry"
    laps = raw_laps
    fitted = load_weather(configs)
    if fitted is not None and (data_dir / "weather.parquet").exists():
        race_start = float(raw_laps.loc[raw_laps["lap_number"] == 1, "lap_start_time_s"].min())
        weather = race_weather(
            fitted[0],
            fitted[1],
            pd.read_parquet(data_dir / "weather.parquet"),
            race_start,
            laps=raw_laps,
        )
        field_wet_skill = sum(p.archetype.wet_skill for p in profiles.profiles) / len(
            profiles.profiles
        )
        weather = weather.model_copy(update={"wet_skill_reference": round(field_wet_skill, 4)})
        weather_note = f"{weather.note}; wet_skill reference = field mean {field_wet_skill:.3f}"
        # every pace estimate below is about DRY pace: take the weather penalty off wet laps
        penalty = lap_weather_penalty(weather, raw_laps, race_start)
        laps = raw_laps.assign(lap_time_s=raw_laps["lap_time_s"] - penalty)
    tables = ("results.parquet", "weather.parquet", "track_status.parquet")
    race = None
    if all((data_dir / t).exists() for t in tables):
        race = race_laps(
            laps,
            *(pd.read_parquet(data_dir / t) for t in tables),
            event_id=roster.event_id,
            season=roster.season,
            round_=roster.round,
            circuit_id=roster.circuit_id,
            driver_ids={e.car_number: e.driver_id for e in roster.entries},
            team_ids={e.car_number: e.team_id for e in roster.entries},
        )
    editorial = default_lap_model(_NOMINAL_BASE_LAP_S.get(roster.event_id, 90.0))
    field = [profiles.get(d).archetype for d in roster.driver_ids()]
    life = load_life(configs)
    estimation_params = None
    if life is not None and (configs / SHAPE_FILE).exists():  # W1: tyre life (D-041)
        shape, _ = load_shape(configs)
        model = life_lap_model(
            editorial,
            *life,
            roster.event_id,
            field,
            shape.fuel_s_per_lap,
            kappa=tyre_life_overrides.get("kappa"),
            delta=tyre_life_overrides.get("delta"),
        )
        lap_params, estimation_params, tyre_note = model.params, model.estimation_params, model.note
    elif (configs / SHAPE_FILE).exists():
        calibrated = calibrated_lap_model(editorial, *load_shape(configs), race, field)
        lap_params, tyre_note = calibrated.params, calibrated.note
    else:  # no fitted model: the deterministic editorial model still runs (plan §16 M4)
        lap_params = editorial
        tyre_note = (
            f"editorial_assumption: no fitted tyre shape ({SHAPE_FILE.as_posix()} missing); "
            "placeholder tyre/fuel model (D-010)"
        )
    # W1: car pace and the base lap are read off real (managed) race laps with the offsets seen
    # on race laps; the simulation then runs the tyres' potential pace
    seen = estimation_params or lap_params
    estimate = estimate_car_offsets(laps, roster, profiles, seen)
    base = estimate_base_lap(laps, roster, profiles, seen, estimate, track.total_laps)
    lap_model = lap_params.model_copy(update={"base_lap_s": base.base_lap_s})
    try:
        pit_loss = estimate_pit_loss(laps)  # weather-adjusted: a stop in the rain isn't slower
        pit, pit_note = (
            pit_params_from_total_loss(pit_loss.total_loss_s, pit_loss.note),
            pit_loss.note,
        )
    except ValueError as exc:  # e.g. a race without any complete pit stop
        pit, pit_note = default_pit_params(), f"editorial_assumption: default pit model ({exc})"
    if random_rain:
        weather = with_random_rain(weather)
        weather_note = weather.note
    traffic, traffic_note = _fitted_traffic(configs, roster.event_id)
    race_control, race_control_note = _fitted_race_control(configs, field)
    return build_setup(
        roster,
        profiles,
        driver_ids,
        track=track,
        lap_model=lap_model,
        pit=pit,
        start_compounds=start_compounds(raw_laps, roster),
        # a car without a lap-1 row (didn't start) gets the field's usual start tyre
        start_compound=_common_start(start_compounds(raw_laps, roster)),
        weather=weather,
        traffic=traffic,
        race_control=race_control,
        car_offsets_s=estimate.offsets_s,
        inputs={
            "start_compounds": f"historical_verified: lap-1 compound, {data_dir.name}/laps.parquet",
            "car_pace_offsets": estimate.note,
            "base_lap": base.note,
            "pit_loss": pit_note,
            "tyre_model": tyre_note,
            "weather": weather_note,
            "traffic": traffic_note,
            "race_control": race_control_note,
            "dataset_sha256": "sha256:" + hashlib.sha256(laps_file.read_bytes()).hexdigest(),
        },
    )


def _fitted_traffic(configs: Path, event_id: str) -> tuple[TrafficParams, str]:
    from f1ar.calibration.traffic_fit import TRAFFIC_FILE, load_traffic, traffic_params

    base = default_traffic_params()
    if not (configs / TRAFFIC_FILE).exists():
        return (
            base,
            f"editorial_assumption: no fitted traffic model ({TRAFFIC_FILE.as_posix()} missing)",
        )
    params = traffic_params(*load_traffic(configs), event_id, base)
    return params, params.note


def _fitted_race_control(
    configs: Path, field: Sequence[DriverArchetype]
) -> tuple[RaceControlParams, str]:
    from f1ar.calibration.race_control_fit import (
        RACE_CONTROL_FILE,
        load_race_control,
        race_control_params,
    )

    base = default_race_control_params()
    if not (configs / RACE_CONTROL_FILE).exists():
        note = (
            f"editorial_assumption: no fitted race control ({RACE_CONTROL_FILE.as_posix()} missing)"
        )
        return base, note
    # the engine's skill factor (incidents.skill_factor), averaged over the field
    mean_skill = sum((1.5 - a.consistency) * (1.5 - a.composure) for a in field) / len(field)
    params = race_control_params(*load_race_control(configs), base, mean_skill)
    return params, params.note


def _common_start(starts: Mapping[str, Compound]) -> Compound:
    if not starts:
        return Compound.MEDIUM
    values = list(starts.values())
    return max(set(values), key=lambda c: (values.count(c), c.value))


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


def spa_2024_track() -> TrackSpec:
    return TrackSpec(
        track_id="spa",
        length_m=7004.0,
        total_laps=44,
        source_note=(
            "Circuit length 7.004 km (published figure); 44 race laps per Jolpica results "
            "for 2024 round 14. Geometry is not used yet."
        ),
    )


_TRACKS = {MONZA_2024: monza_2024_track, BELGIUM_2024: spa_2024_track}
_NOMINAL_BASE_LAP_S = {MONZA_2024: 82.0, BELGIUM_2024: 104.0}
"""Editorial base lap per track for runs without data (data runs estimate it, D-026)."""


def track_for(event_id: str, data_dir: Path | None = None) -> TrackSpec:
    """Published figures for Monza/Spa; for other races, from the fetched data (M4): race laps
    from the results, length from the track outline (the winner's typical racing line, 1-2 %
    shorter than the official length; ``estimated_from_data``)."""
    if event_id in _TRACKS:
        return _TRACKS[event_id]()
    if data_dir is None or not (data_dir / "track_outline.parquet").exists():
        raise ValueError(
            f"no track spec for {event_id}: it needs fetched data with the track outline "
            "(scripts/fetch_event.py without --no-telemetry)"
        )
    from f1ar.presentation.track_geometry import TrackGeometry

    outline = pd.read_parquet(data_dir / "track_outline.parquet")
    length = TrackGeometry.from_points(
        list(zip(outline["x_m"].astype(float), outline["y_m"].astype(float), strict=True))
    ).length_m
    laps = int(pd.read_parquet(data_dir / "results.parquet")["laps"].max())
    return TrackSpec(
        track_id=event_id.split("_", 1)[1],
        length_m=round(length, 1),
        total_laps=laps,
        source_note=(
            f"estimated_from_data: length {length / 1000:.3f} km = the fetched track outline "
            "(winner's typical racing line, 1-2 % shorter than the official figure); "
            f"{laps} race laps from results"
        ),
    )


def default_lap_model(base_lap_s: float = 82.0) -> LapModelParams:
    """Placeholder coefficients (the default base suits Monza). NOT calibrated."""
    return LapModelParams(
        source_type=SourceType.EDITORIAL_ASSUMPTION,
        note="Synthetic placeholder coefficients; to be replaced by FastF1 calibration (M1/M4).",
        base_lap_s=base_lap_s,
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
            **wet_tyre_params(),
        },
        pace_modes={
            PaceMode.CONSERVE: PaceModeParams(offset_s=0.6, wear_multiplier=0.7),
            PaceMode.BALANCED: PaceModeParams(offset_s=0.0, wear_multiplier=1.0),
            PaceMode.PUSH: PaceModeParams(offset_s=-0.4, wear_multiplier=1.4),
        },
    )


RANDOM_RAIN_START_PER_MIN = 0.02
RANDOM_RAIN_STOP_PER_MIN = 0.08


def with_random_rain(weather: WeatherParams) -> WeatherParams:
    """A "what if it rains" world: the fitted wetting/drying dynamics driven by a seeded rain
    process (editorial: rain starts about once per 50 min, a shower lasts ~12 min on average).
    Same rain for every fork of the world with the same seed."""
    return weather.model_copy(
        update={
            "mode": WeatherMode.RANDOM,
            "rain_timeline": (),
            "wetness_timeline": (),
            "initial_wetness": 0.0,
            "random_rain_start_per_min": RANDOM_RAIN_START_PER_MIN,
            "random_rain_stop_per_min": RANDOM_RAIN_STOP_PER_MIN,
            "note": (
                f"{weather.note}; RANDOM rain (editorial: start {RANDOM_RAIN_START_PER_MIN}/min, "
                f"stop {RANDOM_RAIN_STOP_PER_MIN}/min, seeded) instead of the real race's"
            ),
        }
    )


def wet_tyre_params() -> dict[Compound, CompoundParams]:
    """Intermediates and full wets (M4): no dry offset (the weather penalty covers how they
    compare with slicks) and modest wear; editorial (wet-tyre wear isn't separable in 2024's
    data)."""
    return {
        Compound.INTERMEDIATE: CompoundParams(offset_s=0.0, deg_s_per_lap=0.03),
        Compound.WET: CompoundParams(offset_s=0.0, deg_s_per_lap=0.02),
    }


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
    weather: WeatherParams | None = None,
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
        weather=weather or default_weather_params(),
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
        policy_manifest={"strategy": STRATEGY_POLICY_ID},
    )
