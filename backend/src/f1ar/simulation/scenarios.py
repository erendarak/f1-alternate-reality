"""Scenario builders: turn roster + profiles into a ``RaceSetup``."""

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd

from f1ar.calibration.season_files import (
    borrowed_note,
    era_rates_note,
    season_path,
    with_fallback,
)
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
    start_slot,
)
from f1ar.domain.season_rules import SEASON_2024, SeasonRules, rules_for
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
through the season seed (``season_seed``); see ``has_personas``."""
REPO_CONFIGS = Path(__file__).resolve().parents[4] / "configs"
"""This repository's ``configs/`` (the package is installed from the repo, editable)."""


def season_seed(configs_dir: Path, season: int) -> Path | None:
    """``configs/drivers/seed_<season>_season_drivers.yaml`` if it exists (2024 since M4,
    2018-2025 since H2a: ``scripts/build_season_seeds.py``)."""
    path = configs_dir / "drivers" / f"seed_{season}_season_drivers.yaml"
    return path if path.exists() else None


def has_personas(configs_dir: Path, event_id: str) -> bool:
    """A roster exists and a seed covers it (its own seed, or its season's seed)."""
    roster_path = configs_dir / "rosters" / f"{event_id}.json"
    if not roster_path.exists():
        return False
    season = load_roster(roster_path, expected_event_id=event_id).season
    return event_id in _SEED_FILES or season_seed(configs_dir, season) is not None


def load_event(configs_dir: Path, event_id: str = MONZA_2024) -> tuple[RaceRoster, ProfileSet]:
    """Load the verified roster and the persona profiles of one event from ``configs/``.

    Monza and Spa use their own seed files (M2); every other race of a season with a season
    seed uses that one (same persona values for the same driver)."""
    roster = load_roster(configs_dir / "rosters" / f"{event_id}.json", expected_event_id=event_id)
    templates = configs_dir / "drivers" / "archetypes.yaml"
    if event_id in _SEED_FILES:
        profiles = load_profiles(configs_dir / "drivers" / _SEED_FILES[event_id], templates, roster)
    elif (seed := season_seed(configs_dir, roster.season)) is not None:
        profiles = load_season_profiles(seed, templates, roster)
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
    race_events_overrides: dict[str, float] | None = None,
    racecraft_overrides: dict[str, float] | None = None,
    wet_overrides: dict[str, float] | None = None,
) -> RaceSetup:
    """Setup from 3-letter driver codes (e.g. ``["VER", "HAM"]``); ``None`` = full grid.

    With ``data_dir`` (a fetched ``data/processed/<event>/``) the setup uses the real start tyres,
    the calibrated tyre/fuel model (season shape + this race's adjustment, M4/D-033), and the
    data-estimated car offsets, base lap and pit loss; without it, the editorial placeholders
    (Monza and Spa only). ``tyre_life_overrides`` (``kappa``, ``delta``) replace the tyre-life
    globals: only for fitting them (W1). ``wet_overrides`` (``wet_incident_factor``,
    ``wet_refresh_min_age_laps``, ``wet_refresh_share``) replace the W6 fit: only for fitting.
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
    # W5 (D-045): the persona fields that 2024 data can measure, estimated for this race's copy
    from f1ar.calibration.driver_pace_fit import load_pace as _load_pace
    from f1ar.calibration.persona_fit import PERSONA_FILE, load_personas, refit_profiles
    from f1ar.calibration.racecraft_fit import load_racecraft as _load_racecraft

    season = roster.season
    personas_cfg = load_personas(configs, season)
    persona_notes: dict[str, dict[str, str]] = {}
    personas_note = (
        f"editorial_assumption: personas as configured (no {season} persona fit: "
        f"{season_path(PERSONA_FILE, season).as_posix()})"
    )
    if personas_cfg is not None:
        lm0 = default_lap_model(90.0)
        profiles, persona_notes = refit_profiles(
            profiles,
            roster,
            personas_cfg,
            _load_pace(configs, season),
            _load_racecraft(configs, season),
            lm0.tyre_mgmt_wear_max,
            lm0.tyre_mgmt_wear_min,
        )
        personas_note = (
            f"estimated_from_data: persona fields re-fitted from {season} where measurable "
            f"({season_path(PERSONA_FILE, season).as_posix()}): consistency (W2 lap noise "
            f"rank), overtake/defense "
            f"skill (W3 terms rank); tire_management only if its reliability >= 0.3 ({season}: "
            f"{personas_cfg['wear']['reliability_drivers']}); other persona fields editorial"
        )
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
    life = load_life(configs, season)
    estimation_params = None
    if life is not None and (configs / SHAPE_FILE).exists():  # W1: tyre life (D-041)
        shape, _ = load_shape(configs, season)
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
        calibrated = calibrated_lap_model(editorial, *load_shape(configs, season), race, field)
        lap_params, tyre_note = calibrated.params, calibrated.note
    else:  # no fitted model: the deterministic editorial model still runs (plan §16 M4)
        lap_params = editorial
        tyre_note = (
            f"editorial_assumption: no fitted tyre shape ({SHAPE_FILE.as_posix()} missing); "
            "placeholder tyre/fuel model (D-010)"
        )
    # W2 (D-043): the driver's pace vs his teammate and his lap noise, fitted on the season;
    # they replace the persona skill term (skill gap 0) and the persona noise
    from f1ar.calibration.driver_pace_fit import PACE_FILE, load_pace

    pace = load_pace(configs, season)
    driver_offsets: dict[str, float] | None = None
    noise: dict[str, float] = {}
    pace_note = (
        f"editorial_assumption: persona race pace and noise (no {season} fit: "
        f"{season_path(PACE_FILE, season).as_posix()})"
    )
    if pace is not None:
        fitted_drivers = pace["drivers"]
        driver_offsets = {
            e.driver_id: float(fitted_drivers.get(e.code, {}).get("pace_vs_teammate_s", 0.0))
            for e in roster.entries
        }
        noise = {
            e.driver_id: float(
                fitted_drivers.get(e.code, {}).get("lap_noise_s", pace["field_noise_s"])
            )
            for e in roster.entries
        }
        lap_params = lap_params.model_copy(
            update={"skill_gap_s": 0.0, "form_sd_s": float(pace.get("form_sd_s", 0.0))}
        )
        if estimation_params is not None:
            estimation_params = estimation_params.model_copy(update={"skill_gap_s": 0.0})
        pace_note = (
            f"estimated_from_data: driver race pace vs teammate and lap noise "
            f"{season_path(PACE_FILE, season).as_posix()} ({season} season, same-lap fit, "
            f"shrunk; field noise "
            f"{pace['field_noise_s']} s); persona race_pace_skill / consistency not used here"
        )
    # W1: car pace and the base lap are read off real (managed) race laps with the offsets seen
    # on race laps; the simulation then runs the tyres' potential pace
    seen = estimation_params or lap_params
    estimate = estimate_car_offsets(laps, roster, profiles, seen, driver_offsets)
    base = estimate_base_lap(
        laps, roster, profiles, seen, estimate, track.total_laps, driver_offsets
    )
    lap_model = lap_params.model_copy(update={"base_lap_s": base.base_lap_s})
    try:
        pit_loss = estimate_pit_loss(laps)  # weather-adjusted: a stop in the rain isn't slower
        pit, pit_note = (
            pit_params_from_total_loss(pit_loss.total_loss_s, pit_loss.note),
            pit_loss.note,
        )
    except ValueError as exc:  # e.g. a race without any complete pit stop
        pit, pit_note = default_pit_params(), f"editorial_assumption: default pit model ({exc})"
    # W6 (D-046): incidents in the wet, fitted on the three wet 2024 races
    from f1ar.calibration.wet_race_fit import WET_FILE, load_wet_race

    wet_cfg = load_wet_race(configs)
    wet_note = (
        f"editorial_assumption: wet incident factor, no fresh wet tyres under SC/VSC "
        f"({WET_FILE.as_posix()} missing)"
    )
    if wet_overrides:
        update: dict[str, float | int] = dict(wet_overrides)
        if "wet_refresh_min_age_laps" in update:
            update["wet_refresh_min_age_laps"] = int(update["wet_refresh_min_age_laps"])
        weather = weather.model_copy(update=update)
        wet_note = f"override: {dict(wet_overrides)} (fitting)"
    elif wet_cfg is not None:
        refresh = wet_cfg["wet_refresh"]
        weather = weather.model_copy(
            update={
                "wet_incident_factor": float(wet_cfg["wet_incident_factor"]["best"]),
                "wet_refresh_min_age_laps": int(refresh["min_age_laps"]),
                "wet_refresh_share": float(refresh["share"]),
            }
        )
        lo, hi = wet_cfg["wet_incident_factor"]["range_95"]
        wet_note = (
            f"estimated_from_data: wet incident factor {weather.wet_incident_factor} (95% range "
            f"{lo}-{hi}); under SC/VSC {refresh['share']:.0%} of cars on wet tyres "
            f">= {refresh['min_age_laps']} laps old take a new set (3 wet 2024 races, "
            f"{WET_FILE.as_posix()})"
        )
    if random_rain:
        weather = with_random_rain(weather)
        weather_note = weather.note
    traffic, traffic_note = _fitted_traffic(configs, roster.event_id)
    from f1ar.calibration.racecraft_fit import load_racecraft

    if (
        pace is not None
        and pace.get("overtake_threshold_shift_s")
        and load_racecraft(configs, season) is None  # W3's refit supersedes the global shift
    ):
        # W2: realistic lap noise makes more passing chances; the pass threshold is shifted
        # (fitted on the simulated dry_train field) so passes stay at 2024's level
        shift = float(pace["overtake_threshold_shift_s"])
        traffic = traffic.model_copy(
            update={"overtake_threshold_s": traffic.overtake_threshold_s + shift}
        )
        traffic_note = f"{traffic_note}; threshold +{shift:.3f} s for W2 lap noise"
    race_control, race_control_note = _fitted_race_control(configs, field)
    setup = build_setup(
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
        rules=rules_for(configs, roster.season),
        car_offsets_s=estimate.offsets_s,
        inputs={
            "start_compounds": f"historical_verified: lap-1 compound, {data_dir.name}/laps.parquet",
            "car_pace_offsets": estimate.note,
            "base_lap": base.note,
            "pit_loss": pit_note,
            "tyre_model": tyre_note,
            "weather": weather_note,
            "wet_incidents": wet_note,
            "traffic": traffic_note,
            "race_control": race_control_note,
            "season_rules": f"historical_verified: {roster.season} sporting rules, "
            "configs/regulations/seasons.json",
            "dataset_sha256": "sha256:" + hashlib.sha256(laps_file.read_bytes()).hexdigest(),
        },
    )
    if driver_offsets is not None:
        setup = setup.model_copy(
            update={
                "drivers": tuple(
                    d.model_copy(
                        update={
                            "driver_pace_offset_s": driver_offsets.get(d.driver_id, 0.0),
                            "lap_noise_s": noise.get(d.driver_id),
                        }
                    )
                    for d in setup.drivers
                )
            }
        )
    setup = setup.model_copy(update={"inputs": {**setup.inputs, "driver_pace": pace_note}})
    setup = setup.model_copy(
        update={
            "drivers": tuple(
                d.model_copy(update={"persona_estimates": persona_notes.get(d.driver_id, {})})
                for d in setup.drivers
            ),
            "inputs": {**setup.inputs, "personas": personas_note},
        }
    )
    setup = with_racecraft(setup, configs, racecraft_overrides or {})
    setup = with_race_events(setup, configs, race_events_overrides or {})
    return _label_other_seasons(setup, configs)


def with_racecraft(
    setup: RaceSetup, configs: Path, overrides: Mapping[str, float] | None = None
) -> RaceSetup:
    """W3 (D-044): overtaking refit with the W2 lap noise, driver attack/defend and launch
    skills, lap-1 risk by grid slot, from ``configs/calibration/racecraft_2024_v1.json``;
    without it the setup is unchanged (and says so). ``overrides`` (``start_spread_s``) only
    while fitting."""
    from f1ar.calibration.racecraft_fit import RACECRAFT_FILE, load_racecraft

    season = setup.rules.season
    cfg = load_racecraft(configs, season)
    if cfg is None:
        note = (
            f"editorial_assumption: no fitted racecraft "
            f"({season_path(RACECRAFT_FILE, season).as_posix()} missing)"
        )
        return setup.model_copy(update={"inputs": {**setup.inputs, "racecraft": note}})
    ov = cfg["overtaking"]
    own = setup.event_id in ov["track_threshold_s"]
    o = dict(overrides or {})
    shift = float(o.get("race_level_shift_s", ov.get("race_level_shift_s", 0.0)))
    threshold = float(ov["track_threshold_s"].get(setup.event_id, ov["threshold_s"])) + shift
    spread = float(o.get("start_spread_s", cfg["launch"]["start_spread_s"]))
    traffic = setup.traffic.model_copy(
        update={
            "source_type": SourceType.ESTIMATED_FROM_DATA,
            "min_gap_s": float(ov["min_gap_s"]),
            "overtake_threshold_s": threshold,
            "overtake_scale_s": float(ov["scale_s"]),
            "start_spread_s": spread,
            "note": (
                f"estimated_from_data: overtaking refit with the W2 lap noise on "
                f"{ov['encounters']} real encounters ({RACECRAFT_FILE.as_posix()}): "
                f"{'this track' if own else 'season'} threshold {threshold:+.2f} s, scale "
                f"{float(ov['scale_s']):.2f} s (incl. race-level shift {shift:+.2f} s fitted by "
                f"simulation), driver attack/defend terms; launch spread "
                f"{spread:.2f} s matched to real lap-1 passes; passing/dirty-air/blue-flag time "
                f"losses editorial"
            ),
        }
    )
    skills, launch = ov["drivers"], cfg["launch"]["drivers"]
    drivers = tuple(
        d.model_copy(
            update={
                "attack_s": float(skills.get(d.code, {}).get("attack_s", 0.0)),
                "defend_s": float(skills.get(d.code, {}).get("defend_s", 0.0)),
                "launch_offset_s": float(launch.get(d.code, {}).get("launch_offset_s", 0.0)),
            }
        )
        for d in setup.drivers
    )
    rc = setup.race_control.model_copy(
        update={"first_lap_grid_profile": tuple(float(v) for v in cfg["lap1_grid_profile"])}
    )
    note = (
        f"estimated_from_data: racecraft {season_path(RACECRAFT_FILE, season).as_posix()} "
        f"({season}): attack/defend per "
        f"driver (ridge-shrunk; driver and car effects not separable), launch skill per driver "
        f"(lap-1 gains beyond grid and car pace, shrunk); lap-1 risk by grid slot: editorial "
        f"shape from Dartmouth's 2022-23 lap-1 incident counts"
    )
    return setup.model_copy(
        update={
            "traffic": traffic,
            "drivers": drivers,
            "race_control": rc,
            "inputs": {**setup.inputs, "traffic": traffic.note, "racecraft": note},
        }
    )


def with_race_events(
    setup: RaceSetup, configs: Path, overrides: Mapping[str, float] | None = None
) -> RaceSetup:
    """W4 (D-042): pit crews, reliability per team, debris, red flags, penalties, the risk scale
    from ``configs/calibration/race_events_2024_v1.json``; without it the setup is unchanged
    (and says so). ``overrides`` replace fitted values (only while fitting them)."""
    from f1ar.calibration.race_events_fit import EVENTS_FILE, load_events

    season = setup.rules.season
    cfg = load_events(configs, season)
    if cfg is None:
        note = (
            f"editorial_assumption: no fitted race events "
            f"({season_path(EVENTS_FILE, season).as_posix()} missing)"
        )
        return setup.model_copy(update={"inputs": {**setup.inputs, "race_events": note}})
    o = dict(overrides or {})
    fitted = cfg["fitted"]
    pens = cfg["penalties"]
    rc = setup.race_control.model_copy(
        update={
            "risk_scale": float(o.get("risk_scale", fitted["risk_scale"])),
            "debris_per_lap": float(cfg["debris"]["per_lap"]),
            "debris_sc_share": float(cfg["debris"]["sc_share"]),
            "red_flag_per_crash": float(o.get("red_flag_per_crash", fitted["red_flag_per_crash"])),
            "penalty_collision_per_pass": float(pens["collision_per_pass"]),
            "penalty_false_start": float(pens["false_start_per_starter"]),
            "penalty_pit_speeding_per_stop": float(pens["pit_speeding_per_stop"]),
            "penalty_per_lap": float(pens["other_per_car_lap"]),
        }
    )
    pit = setup.pit.model_copy(
        update={
            "slow_stop_probability": float(cfg["pit"]["slow_stop_probability"]),
            "slow_stop_scale_s": float(cfg["pit"]["slow_stop_scale_s"]),
        }
    )
    crews, teams = cfg["crews"], cfg["reliability"]["teams"]
    drivers = tuple(
        d.model_copy(
            update={
                "pit_crew_offset_s": float(crews.get(d.team_id, {}).get("pit_crew_offset_s", 0.0)),
                "pit_slow_multiplier": float(
                    crews.get(d.team_id, {}).get("pit_slow_multiplier", 1.0)
                ),
                "failure_multiplier": float(
                    teams.get(d.team_id, {}).get("failure_multiplier", 1.0)
                ),
            }
        )
        for d in setup.drivers
    )
    note = (
        f"estimated_from_data: race events {season_path(EVENTS_FILE, season).as_posix()} "
        f"({season}): pit crews and "
        f"slow stops per team from {cfg['pit']['stops']} green stops, team failure multipliers "
        f"(all-cause retirements as proxy, shrunk), debris SC/VSC {cfg['debris']['per_lap']}/lap, "
        f"penalties by type; risk scale {rc.risk_scale:.2f} and red flags per crash "
        f"{rc.red_flag_per_crash:.3f} fitted on the simulated dry_train field; penalty-proneness "
        f"(1.5 - composure) editorial"
    )
    return setup.model_copy(
        update={
            "race_control": rc,
            "pit": pit,
            "drivers": drivers,
            "inputs": {**setup.inputs, "race_events": note},
        }
    )


def _label_other_seasons(setup: RaceSetup, configs: Path) -> RaceSetup:
    """H2a (D-054): say which inputs of a non-2024 race come from 2024 (tyre model borrowed,
    era-level rates kept until H2c)."""
    from f1ar.calibration.lap_model_fit import LIFE_FILE, SHAPE_FILE

    season = setup.rules.season
    _, life_season = with_fallback(configs, LIFE_FILE, season)
    _, shape_season = with_fallback(configs, SHAPE_FILE, season)
    inputs = dict(setup.inputs)
    if "tyre_model" in inputs:
        inputs["tyre_model"] += borrowed_note(season, life_season or shape_season)
        if life_season == season and shape_season not in (None, season):
            inputs["tyre_model"] += f" [fuel effect from the {shape_season} tyre shape fit]"
    for key in ("traffic", "race_control", "weather", "wet_incidents"):
        if key in inputs:
            inputs[key] += era_rates_note(season)
    return setup.model_copy(update={"inputs": inputs})


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
    outline_dir, borrowed = data_dir, ""
    if data_dir is not None and not (data_dir / "track_outline.parquet").exists():
        sibling = _same_circuit_outline(event_id, data_dir)
        if sibling is not None:  # H2a: no position data in F1's archive (2018 AUS, BHR)
            outline_dir = sibling
            borrowed = f" (outline of {sibling.name}: same circuit, nearest season with one)"
    if (
        outline_dir is None
        or data_dir is None
        or not (outline_dir / "track_outline.parquet").exists()
    ):
        raise ValueError(
            f"no track spec for {event_id}: it needs fetched data with the track outline "
            "(scripts/fetch_event.py without --no-telemetry)"
        )
    from f1ar.presentation.track_geometry import TrackGeometry

    outline = pd.read_parquet(outline_dir / "track_outline.parquet")
    length = TrackGeometry.from_points(
        list(zip(outline["x_m"].astype(float), outline["y_m"].astype(float), strict=True))
    ).length_m
    laps = int(pd.read_parquet(data_dir / "results.parquet")["laps"].max())
    return TrackSpec(
        track_id=event_id.split("_", 1)[1],
        length_m=round(length, 1),
        total_laps=laps,
        source_note=(
            f"estimated_from_data: length {length / 1000:.3f} km = the fetched track outline"
            f"{borrowed} (winner's typical racing line, 1-2 % shorter than the official "
            f"figure); {laps} race laps from results"
        ),
    )


def _same_circuit_outline(event_id: str, data_dir: Path) -> Path | None:
    """The data folder of the nearest-season race on the same circuit (``circuit_id`` in the
    rosters) that has a track outline. A layout change between the seasons isn't detected:
    the nearest season keeps that risk small (labelled in the source note)."""
    root = data_dir.parents[2]
    rosters = root / "configs" / "rosters"
    own = rosters / f"{event_id}.json"
    if not own.exists():
        return None
    me = json.loads(own.read_text(encoding="utf-8"))
    best: tuple[int, Path] | None = None
    for path in rosters.glob("*.json"):
        other = json.loads(path.read_text(encoding="utf-8"))
        if other["circuit_id"] != me["circuit_id"] or path.stem == event_id:
            continue
        folder = root / "data" / "processed" / path.stem
        if (folder / "track_outline.parquet").exists():
            gap = abs(int(other["season"]) - int(me["season"]))
            if best is None or gap < best[0]:
                best = (gap, folder)
    return None if best is None else best[1]


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
    rules: SeasonRules | None = None,
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
    entries = sorted(
        (roster.entry(d) for d in driver_ids), key=lambda e: start_slot(e.grid_position)
    )
    pit = pit or default_pit_params()
    if rules is not None:  # H1: the season's dry-compound rule
        pit = pit.model_copy(
            update={"min_distinct_dry_compounds": rules.min_distinct_dry_compounds}
        )
    return RaceSetup(
        event_id=roster.event_id,
        track=track or monza_2024_track(),
        lap_model=lap_model or default_lap_model(),
        pit=pit,
        traffic=traffic or default_traffic_params(),
        race_control=race_control or default_race_control_params(),
        weather=weather or default_weather_params(),
        rules=rules or SEASON_2024,
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
