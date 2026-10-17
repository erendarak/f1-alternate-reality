"""Authoritative, immutable world state (plan §4.4).

A ``WorldState`` contains everything that influences the future of the world: race setup
(track, model parameters, embedded driver profiles), per-car state, clocks and RNG spec.
Nothing outside it may affect ``step()``. That is what makes save, load, replay and fork
exact. States are frozen; the engine returns a new state per tick and never mutates one.
"""

from enum import StrEnum
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from f1ar.domain.driver_profiles import DriverProfile
from f1ar.domain.provenance import SourceType
from f1ar.domain.season_rules import SEASON_2024, SeasonRules
from f1ar.persistence.snapshot_codec import state_hash
from f1ar.simulation.rng_streams import RngSpec

WORLD_SCHEMA_VERSION = 18


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class WorldStatus(StrEnum):
    CREATED = "CREATED"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    COMPLETED = "COMPLETED"


class CarStatus(StrEnum):
    RUNNING = "RUNNING"
    FINISHED = "FINISHED"
    RETIRED = "RETIRED"
    """Out of the race (crash or mechanical failure); parked where it stopped."""


class RacePhase(StrEnum):
    GREEN = "GREEN"
    VSC = "VSC"
    """Virtual safety car: every car at a fixed fraction of racing speed; the pit lane isn't
    slowed down."""
    SC = "SC"
    """Safety car on track: the field queues behind it, no overtaking."""


class Compound(StrEnum):
    SOFT = "SOFT"
    MEDIUM = "MEDIUM"
    HARD = "HARD"
    INTERMEDIATE = "INTERMEDIATE"
    WET = "WET"


DRY_COMPOUNDS: tuple[Compound, ...] = (Compound.SOFT, Compound.MEDIUM, Compound.HARD)
"""Slicks. Strategy plans and the dry-compound rule are about these (M4: wet tyres, D-034)."""


class PaceMode(StrEnum):
    CONSERVE = "CONSERVE"
    BALANCED = "BALANCED"
    PUSH = "PUSH"


class RacecraftIntent(StrEnum):
    """How hard a driver races the cars around them this lap (M5 slice 4, D-048).

    ATTACK: commit to passes (easier pass, more risk of a crash, a mistake or a collision
    penalty); DEFEND: make yourself hard to pass (costs lap time when attacked, more risk);
    NORMAL: the 2024-calibrated behaviour."""

    DEFEND = "DEFEND"
    NORMAL = "NORMAL"
    ATTACK = "ATTACK"


class StrategyController(StrEnum):
    """Who decides a car's pit stops."""

    BASELINE = "BASELINE"
    """The built-in baseline team strategy (``f1ar.policy.baseline_strategy``)."""
    USER = "USER"
    """A human via pit actions; the baseline stays out of it."""
    AGENT = "AGENT"
    """A learned policy (M5) via the same pit/pace actions; the baseline stays out of it. Set
    only by an agent runner (actions whose ``actor_id`` starts with ``AGENT``); any human
    command on the car takes it over (-> USER): the user can decide for anyone (D-037)."""


# --- setup: fixed for the lifetime of a world ---------------------------------------------


class TrackSpec(_Frozen):
    track_id: str
    length_m: float = Field(gt=0)
    total_laps: int = Field(gt=0)
    source_note: str


class CompoundParams(_Frozen):
    offset_s: float
    """Lap-time offset of a fresh tyre vs. the reference compound."""
    deg_s_per_lap: float = Field(ge=0)
    """Added lap time per lap of tyre age for a neutral driver in BALANCED mode."""
    deg_quad_s_per_lap2: float = 0.0
    """Quadratic wear term (M4, D-033): wear after ``a`` laps = deg a + quad a^2. The lap's
    increment ``deg + quad (2a + 1)`` is clamped at 0, so wear never decreases with age."""
    life_laps: float | None = Field(default=None, gt=0)
    """Tyre life (W1, D-041): laps a neutral driver gets from a new set in BALANCED mode before
    the cliff. ``None`` = no life limit (synthetic setups, wet tyres: the pre-W1 model). With
    a life, managing (CONSERVE) stretches a set and PUSH uses it up; past the life the cliff
    adds lap time."""


class PaceModeParams(_Frozen):
    offset_s: float
    wear_multiplier: float = Field(gt=0)
    """Scales how fast tyre age turns into lap time (push wears tyres faster)."""


class LapModelParams(_Frozen):
    """Coefficients of the minimal lap-time model. Synthetic placeholders until calibration (M4)."""

    source_type: Literal[SourceType.EDITORIAL_ASSUMPTION, SourceType.ESTIMATED_FROM_DATA]
    """``estimated_from_data`` once calibrated (M4): tyre/fuel shape fitted on other races,
    per-race values on the race itself; skill, noise and pace modes stay editorial."""
    note: str
    base_lap_s: float = Field(gt=0)
    skill_gap_s: float = Field(ge=0)
    """Lap time lost by a driver with race_pace_skill=0 vs. 1."""
    fuel_s_per_lap_remaining: float = Field(ge=0)
    tyre_mgmt_wear_max: float = Field(gt=0)
    """Wear factor at tire_management=0; it falls linearly to ``tyre_mgmt_wear_min`` at 1."""
    tyre_mgmt_wear_min: float = Field(gt=0)
    noise_floor_s: float = Field(ge=0)
    noise_inconsistency_s: float = Field(ge=0)
    """Extra lap-time std-dev for a driver with consistency=0 vs. 1."""
    min_lap_fraction: float = Field(gt=0, lt=1)
    """Sampled lap times are clamped to at least this fraction of ``base_lap_s``."""
    compounds: dict[Compound, CompoundParams]
    pace_modes: dict[PaceMode, PaceModeParams]
    fresh_tyre_bonus_s: float = 0.0
    """W1: lap-time change on the first flying lap of a new set after a stop (negative = the
    fresh-tyre peak, measured on 2024 laps)."""
    cliff_s_per_lap: float = Field(default=0.0, ge=0)
    """W1: lap time added per BALANCED-lap of use beyond a set's life, growing with every lap
    past it (editorial shape)."""
    form_sd_s: float = Field(default=0.0, ge=0)
    """W2: SD of a driver's race form (a pace offset drawn once per driver and race)."""
    neutralised_life_factor: float = Field(default=1.0, gt=0)
    """W1: tyre life used on a lap that ends under SC/VSC, relative to a racing lap."""

    @model_validator(mode="after")
    def _complete(self) -> Self:
        if set(self.compounds) != set(Compound) or set(self.pace_modes) != set(PaceMode):
            raise ValueError("lap model must define every compound and pace mode")
        return self


class PitParams(_Frozen):
    """Pit-stop time model (plan §7.5). Standstill shape credited to TUM FTM race-simulation
    (Heilmeier et al.), which models standstill times with a log-logistic (fisk) distribution."""

    source_type: Literal[SourceType.EDITORIAL_ASSUMPTION, SourceType.ESTIMATED_FROM_DATA]
    note: str
    lane_loss_s: float = Field(ge=0)
    """Time lost driving pit entry + pit lane + pit exit instead of the racing line."""
    standstill_median_s: float = Field(gt=0)
    standstill_shape: float = Field(gt=1)
    """Log-logistic shape: higher = fewer slow stops."""
    standstill_max_s: float = Field(gt=0)
    """Draws are capped here (a stuck wheel, not a lost race)."""
    min_distinct_dry_compounds: int = Field(ge=0, le=3)
    """Dry-race rule: compounds a finisher must have used (2 in 2024; 0 disables)."""
    slow_stop_probability: float = Field(default=0.0, ge=0, le=1)
    """W4: share of stops that go wrong (wheel nut, jack, ...): ``slow_stop_min_s`` plus an
    exponential with mean ``slow_stop_scale_s`` on top of the standstill (0 = off)."""
    slow_stop_min_s: float = Field(default=3.0, ge=0)
    slow_stop_scale_s: float = Field(default=7.5, gt=0)


class TrafficParams(_Frozen):
    """Car-to-car interaction (plan §7.4: an overtake is a race event, not two sprites crossing).

    Idea credited to TUM FTM race-simulation (Heilmeier et al.): a car that would reach a car
    ahead is either held behind it at a minimum gap or overtakes, depending on its pace
    advantage, with time lost by both. Values are editorial until calibrated (M4).
    """

    source_type: Literal[SourceType.EDITORIAL_ASSUMPTION, SourceType.ESTIMATED_FROM_DATA]
    note: str
    enabled: bool = True
    """False = cars pass through each other (pre-v0.5.0 behaviour; for isolated experiments)."""
    min_gap_s: float = Field(ge=0)
    """A car held up crosses the line this far behind the car it's stuck behind."""
    overtake_threshold_s: float = Field(ge=-2.0, le=5.0)
    """Margin above which an overtake is possible. Negative (fitted, M4): a car that only just
    reaches the one ahead already has a chance (the attempt itself still needs margin > 0)."""
    overtake_scale_s: float = Field(gt=0)
    """P(success) = 1 - exp(-(advantage - threshold) / scale)."""
    attacker_loss_s: float = Field(ge=0)
    defender_loss_s: float = Field(ge=0)
    """Applied to the defender's next lap (its current lap is already fixed)."""
    dirty_air_window_s: float = Field(ge=0)
    """Following closer than this (time gap) costs ``dirty_air_loss_s`` on that lap."""
    dirty_air_loss_s: float = Field(ge=0)
    blue_flag_attacker_loss_s: float = Field(ge=0)
    blue_flag_defender_loss_s: float = Field(ge=0)
    """A lapped car always yields; it loses this on its next lap."""
    intent_attack_boost_s: float = Field(default=0.2, ge=0)
    """Slice 4 (D-048, editorial size anchored on W3): an ATTACKING car's pass threshold drops by
    this much, about the spread between 2024's best and worst attackers (W3 attack terms)."""
    intent_defend_boost_s: float = Field(default=0.2, ge=0)
    """A DEFENDING car raises its attackers' threshold by this much (W3 defend-term spread)."""
    intent_defend_loss_s: float = Field(default=0.3, ge=0)
    """Extra time a DEFENDING car loses on its next lap for each attempt it fends off or
    suffers (a defensive line is slower; editorial)."""
    start_spread_s: float = Field(default=0.0, ge=0)
    """Launch and first-corner spread: each car's lap 1 gets ``N(0, spread)`` seconds (keyed per
    car), resolved in grid order like any other lap. 0 = everyone launches identically
    (before v0.9.0)."""


class RaceControlParams(_Frozen):
    """Incidents and neutralisations (plan §7.5, §10). Ideas credited to TUM FTM race-simulation
    (per-driver accident probability, higher risk on lap 1, FCY phases); probabilities and
    durations are editorial until calibrated against real seasons (EXP-002, M4)."""

    source_type: Literal[SourceType.EDITORIAL_ASSUMPTION, SourceType.ESTIMATED_FROM_DATA]
    note: str
    enabled: bool = True
    """False = no mistakes, crashes, failures or neutralisations (isolated experiments)."""
    mistake_per_lap: float = Field(ge=0, le=1)
    """Mistake probability per lap for a neutral driver (consistency = composure = 0.5) in
    BALANCED mode."""
    mistake_loss_min_s: float = Field(ge=0)
    mistake_loss_max_s: float = Field(ge=0)
    crash_per_lap: float = Field(ge=0, le=1)
    """Crash (retirement) probability per lap for a neutral driver in BALANCED."""
    first_lap_crash_multiplier: float = Field(ge=1)
    failure_per_lap: float = Field(ge=0, le=1)
    """Mechanical failure (retirement) probability per lap; the same for every car for now."""
    push_risk_multiplier: float = Field(ge=1)
    conserve_risk_multiplier: float = Field(gt=0, le=1)
    crash_sc_probability: float = Field(ge=0, le=1)
    crash_vsc_probability: float = Field(ge=0, le=1)
    """After a crash: SC with the first probability, else VSC with the second, else nothing."""
    failure_sc_probability: float = Field(ge=0, le=1)
    failure_vsc_probability: float = Field(ge=0, le=1)
    vsc_speed_fraction: float = Field(gt=0, le=1)
    """Under VSC every car runs at this fraction of its racing speed."""
    vsc_min_s: float = Field(gt=0)
    vsc_max_s: float = Field(gt=0)
    sc_min_laps: int = Field(ge=1)
    sc_max_laps: int = Field(ge=1)
    sc_lap_factor: float = Field(gt=1)
    """Safety-car lap time = base lap x this."""
    sc_catch_up_lap_factor: float = Field(gt=1)
    """Cars behind the safety car close up at base lap x this until they reach the queue."""
    sc_spacing_m: float = Field(gt=0)
    """Distance kept to the car (or safety car) ahead in the queue."""
    sc_deploy_ahead_m: float = Field(ge=0)
    """The safety car joins this far ahead of the race leader."""
    neutralised_pit_window_laps: int = Field(ge=0)
    """Baseline strategy: under SC/VSC, bring a planned stop forward if it's this close."""
    first_lap_grid_profile: tuple[float, ...] = ()
    """W3: lap-1 crash risk multiplier per grid slot (mean 1; empty = the same for all)."""
    # --- W4 (engine v0.11.0, D-042): all off by default -----------------------------------
    risk_scale: float = Field(default=1.0, gt=0)
    """Multiplies crash and mistake rates (W4: renormalised for the W1 pace-mode mix)."""
    debris_per_lap: float = Field(default=0.0, ge=0, le=1)
    """A neutralisation that doesn't come from a retirement (debris, a car stopped and
    recovered), per race lap, drawn when the leader starts a lap."""
    debris_sc_share: float = Field(default=0.4, ge=0, le=1)
    """Of those: safety car (else VSC)."""
    red_flag_per_crash: float = Field(default=0.0, ge=0, le=1)
    """A crash stops the race (red flag): everyone gets a free tyre change, then a restart
    behind the safety car."""
    red_flag_sc_laps: int = Field(default=3, ge=1)
    penalty_collision_per_pass: float = Field(default=0.0, ge=0, le=1)
    """10 s for the attacker of a pass (causing a collision / forcing a driver off)."""
    intent_attack_risk: float = Field(default=2.0, ge=1)
    """Slice 4 (D-048, editorial): crash and mistake rates x this on a lap raced ATTACKING."""
    intent_defend_risk: float = Field(default=1.5, ge=1)
    """... and x this on a lap raced DEFENDING."""
    intent_attack_collision: float = Field(default=3.0, ge=1)
    """The 2024 collision-penalty rate per pass (W4) x this for a pass made ATTACKING."""
    penalty_false_start: float = Field(default=0.0, ge=0, le=1)
    """5 s per car at the start."""
    penalty_pit_speeding_per_stop: float = Field(default=0.0, ge=0, le=1)
    """5 s per stop."""
    penalty_per_lap: float = Field(default=0.0, ge=0, le=1)
    """5 s per car and lap (track limits, leaving the track, SC infringements, ...)."""


class WeatherMode(StrEnum):
    HISTORICAL = "HISTORICAL"
    """The real race's rain timeline (no rain for a dry race)."""
    RANDOM = "RANDOM"
    """A seeded two-state rain process per race minute (scenario worlds)."""


class WeatherParams(_Frozen):
    """Rain, track wetness and tyre penalties (M4 slice 4, D-034; ``simulation/weather.py``)."""

    source_type: Literal[SourceType.EDITORIAL_ASSUMPTION, SourceType.ESTIMATED_FROM_DATA]
    note: str
    mode: WeatherMode = WeatherMode.HISTORICAL
    rain_timeline: tuple[tuple[float, bool], ...] = ()
    """HISTORICAL: (race time s, raining) samples, piecewise constant, from the race's weather
    data (``historical_verified`` flags; times relative to the race start)."""
    wetness_timeline: tuple[tuple[float, float], ...] = ()
    """HISTORICAL: (race time s, wetness) as the real race had it, inferred from the field's
    real tyre choices (``estimated_from_data``, D-034). When present it sets the track wetness
    (interpolated); the rain flags then only drive the rain events. The on/off rain flag alone
    can't tell how wet a track stays (2024 Canada stayed wet for ~25 min without rain, Britain
    dried in ~10), so the wetting/drying dynamics are used only without it (RANDOM mode)."""
    initial_wetness: float = Field(default=0.0, ge=0, le=1)
    random_rain_start_per_min: float = Field(default=0.0, ge=0, le=1)
    random_rain_stop_per_min: float = Field(default=0.1, ge=0, le=1)
    wet_rate_per_s: float = Field(gt=0)
    dry_rate_per_s: float = Field(gt=0)
    slick_s_per_wetness: float = Field(ge=0)
    inter_base_s: float = Field(ge=0)
    inter_s_per_wetness: float = Field(ge=0)
    wet_base_s: float = Field(ge=0)
    wet_s_per_wetness: float = Field(ge=0)
    wet_skill_effect: float = Field(ge=0, le=1.5)
    """Penalty x (1 + effect (reference - wet_skill)): editorial."""
    wet_skill_reference: float = Field(default=0.5, ge=0, le=1)
    """The field's mean wet_skill for data races, so the average driver gets exactly the
    fitted penalty (personas are editorial and mostly above 0.5)."""
    wet_noise_factor: float = Field(ge=0)
    """Lap-time noise sd x (1 + factor * wetness): editorial."""
    wet_incident_factor: float = Field(ge=0)
    """Mistake/crash rates x (1 + factor * wetness): editorial."""
    switch_horizon_laps: int = Field(ge=1)
    """Baseline wet strategy: change tyre type when the gain over this many laps beats the pit
    loss (it assumes conditions stay as they are: it doesn't see the future)."""
    wet_refresh_min_age_laps: int = Field(default=99, ge=0)
    wet_refresh_share: float = Field(default=0.0, ge=0, le=1)
    """W6: under SC/VSC a baseline car on wet-weather tyres at least ``min_age`` laps old takes
    a new set of the same type with this probability (one keyed draw per car and
    neutralisation). Fitted on 2024 Canada / São Paulo; off (share 0) in editorial setups."""
    forecast_horizon_s: float = Field(gt=0)
    forecast_noise: float = Field(ge=0, le=1)


def default_weather_params() -> "WeatherParams":
    """Dry, with editorial tyre penalties (synthetic setups; data races use the fitted file)."""
    return WeatherParams(
        source_type=SourceType.EDITORIAL_ASSUMPTION,
        note="editorial placeholders; no rain in HISTORICAL mode without a timeline",
        wet_rate_per_s=1 / 600,
        dry_rate_per_s=1 / 1800,
        slick_s_per_wetness=30.0,
        inter_base_s=5.0,
        inter_s_per_wetness=10.0,
        wet_base_s=9.0,
        wet_s_per_wetness=5.0,
        wet_skill_effect=0.4,
        wet_noise_factor=1.0,
        wet_incident_factor=2.0,
        switch_horizon_laps=15,
        forecast_horizon_s=600.0,
        forecast_noise=0.2,
    )


class WeatherState(_Frozen):
    raining: bool = False
    wetness: float = Field(default=0.0, ge=0, le=1)


class DriverEntry(_Frozen):
    driver_id: str
    code: str
    team_id: str
    grid_position: int = Field(ge=0)
    start_compound: Compound
    car_pace_offset_s: float = Field(default=0.0, ge=0.0, le=10.0)
    """Car performance: lap time lost vs. the fastest car in the field (0 = fastest car).
    Separate from driver skill (plan §5.2); see ``RaceSetup.inputs`` for its provenance."""
    pit_crew_offset_s: float = Field(default=0.0, ge=-2.0, le=5.0)
    """W4: the team's stationary time vs. the field's median."""
    pit_slow_multiplier: float = Field(default=1.0, ge=0)
    """W4: the team's slow-stop rate vs. the field's."""
    failure_multiplier: float = Field(default=1.0, ge=0)
    """W4: the team's mechanical failure rate vs. the field's."""
    driver_pace_offset_s: float = Field(default=0.0, ge=-3.0, le=3.0)
    """W2: the driver's race pace vs. his teammate (season fit; negative = faster). When a race
    uses it, the lap model's persona skill term is off (``LapModelParams.skill_gap_s`` = 0)."""
    lap_noise_s: float | None = Field(default=None, gt=0)
    """W2: the driver's lap-to-lap spread (season fit); ``None`` = the persona formula."""
    attack_s: float = Field(default=0.0, ge=-2.0, le=2.0)
    """W3: lowers the pass threshold when this driver attacks (racecraft, 2024 encounters)."""
    defend_s: float = Field(default=0.0, ge=-2.0, le=2.0)
    """W3: raises the pass threshold when this driver defends."""
    launch_offset_s: float = Field(default=0.0, ge=-3.0, le=3.0)
    """W3: lap-1 time vs an average start (negative = good launches; 2024 lap-1 gains)."""
    persona_estimates: dict[str, str] = Field(default_factory=dict)
    """W5: persona fields of this race's profile copy that were estimated from data, with a note
    (the rest of the persona stays editorial)."""
    profile: DriverProfile


class RaceSetup(_Frozen):
    event_id: str
    track: TrackSpec
    lap_model: LapModelParams
    pit: PitParams
    traffic: TrafficParams
    race_control: RaceControlParams
    weather: WeatherParams = Field(default_factory=default_weather_params)
    rules: SeasonRules = SEASON_2024
    """The season's sporting rules (H1, D-053); 2024's unless the race is from another season."""
    tick_seconds: float = Field(gt=0)
    profile_set_manifest_hash: str
    """Hash of the full ProfileSet the embedded profiles were taken from."""
    drivers: tuple[DriverEntry, ...]
    inputs: dict[str, str] = Field(default_factory=dict)
    """Provenance of data-derived inputs, e.g. ``{"start_compounds": "historical_verified: ..."}``.
    Empty = synthetic defaults (all MEDIUM, equal cars). ``dataset_sha256`` pins the data file."""
    policy_manifest: dict[str, str]
    """Which decision policies drive the cars, e.g. ``{"strategy": "baseline_strategy_v1"}``
    (plan §4.4 ``active_policy_manifest``). Pinned: a save made with another policy build is
    refused, like one made with another engine version."""

    @model_validator(mode="after")
    def _unique(self) -> Self:
        ids = [d.driver_id for d in self.drivers]
        if len(set(ids)) != len(ids) or not ids:
            raise ValueError("race setup needs at least one driver and unique driver ids")
        return self

    def driver(self, driver_id: str) -> DriverEntry:
        for d in self.drivers:
            if d.driver_id == driver_id:
                return d
        raise KeyError(driver_id)


# --- dynamic state ------------------------------------------------------------------------


class PlannedStop(_Frozen):
    lap: int = Field(ge=1)
    """Pit at the end of this lap (the car crosses the line in the pit lane)."""
    compound: Compound


class PendingOvertake(_Frozen):
    """A pass decided when the attacker joined the track; announced (OVERTAKE) when the
    attacker actually moves ahead on the road, dropped if a safety car comes out first."""

    rival: str
    lap: int = Field(ge=1)
    margin_s: float
    probability: float = Field(ge=0, le=1)


class CarState(_Frozen):
    driver_id: str
    status: CarStatus
    laps_completed: int = Field(ge=0)
    distance_m: float = Field(ge=0)
    """Distance into the current lap, in [0, track length). Derived from ``lap_elapsed_s``."""
    lap_elapsed_s: float = Field(ge=0)
    lap_target_s: float = Field(gt=0)
    """Pre-sampled duration of the current lap, including ``pit_hold_s``. HIDDEN: never part
    of an observation."""
    pit_hold_s: float = Field(ge=0)
    """Out-lap only: pit-lane loss + standstill, spent at the line before driving the lap."""
    compound: Compound
    tyre_age_laps: int = Field(ge=0)
    penalty_s: float = Field(default=0.0, ge=0)
    """W4: time penalties collected (added to the race time at the finish)."""
    tyre_life_used: float = Field(default=0.0, ge=0)
    """W1: share of the set's life used (1 = at the cliff); 0 without a life model. HIDDEN."""
    tyre_wear_s: float = Field(ge=0)
    """Accumulated degradation of the current tyre set, as lap time added. HIDDEN: a real
    team only infers this from lap times, so it is not part of an observation."""
    pace_mode: PaceMode
    pace_by_user: bool = False
    """W1: a human set this car's pace mode; the baseline's automatic tyre management leaves it
    alone (until the car is handed back to the baseline)."""
    """Driver's intended mode; applies from the next lap start."""
    lap_pace_mode: PaceMode
    """Mode the lap in progress is being driven in (locked when the lap started)."""
    intent: RacecraftIntent = RacecraftIntent.NORMAL
    """Slice 4: the driver's racecraft intent; applies from the next lap start (like pace)."""
    lap_intent: RacecraftIntent = RacecraftIntent.NORMAL
    """The intent the lap in progress is raced with (locked when the lap started)."""
    pit_request: Compound | None
    """Pending stop: the car pits for this compound at its next line crossing."""
    pit_stops: int = Field(ge=0)
    compounds_used: tuple[Compound, ...]
    """Distinct compounds raced so far, in first-use order (for the dry-tyre rule)."""
    controller: StrategyController
    planned_stops: tuple[PlannedStop, ...]
    """Baseline strategy's remaining plan (empty under USER control)."""
    traffic_loss_s: float
    """Time the lap in progress lost to traffic (held up, dirty air, passing, carried losses);
    slightly negative when a pass gained time (DRS/slipstream). HIDDEN: part of the
    pre-sampled lap; ``last_lap_traffic_loss_s`` is the observable version."""
    last_lap_traffic_loss_s: float
    carried_loss_s: float = Field(ge=0)
    """Time lost defending or yielding, added to the car's next lap."""
    pending_overtakes: tuple[PendingOvertake, ...]
    """HIDDEN: passes decided for this lap that haven't happened on the road yet."""
    incident_kind: Literal["MISTAKE", "CRASH", "FAILURE"] | None
    """This lap's incident, drawn at lap start from keyed randomness. HIDDEN, like
    ``lap_target_s``: a MISTAKE's time loss is already in the lap time; it's announced, or the
    car retires (CRASH/FAILURE), when the car reaches ``incident_at_fraction``."""
    incident_at_fraction: float | None
    lap_started_at_s: float = Field(ge=0)
    """Sim time the lap in progress started (line crossing); lap times are measured from it."""
    last_lap_time_s: float | None
    best_lap_time_s: float | None = None
    """Fastest lap so far (H1: the fastest-lap point of 2019-2024)."""
    finish_time_s: float | None
    retired_reason: str | None
    """``"CRASH"`` or ``"FAILURE"`` once RETIRED."""
    retired_at_s: float | None


class RaceControl(_Frozen):
    """Neutralisation state. Under SC the safety car is a real entity on track."""

    phase: RacePhase
    since_s: float = Field(ge=0)
    vsc_until_s: float | None
    """VSC only: sim time the VSC ends."""
    sc_distance_m: float | None
    """SC only: the safety car's distance into the lap."""
    sc_laps_left: int = Field(ge=0)
    """SC only: line crossings of the safety car left before it pulls in (0 = pulls in now)."""
    cause: str | None
    """Driver whose incident caused the neutralisation."""
    sc_count: int = Field(ge=0)
    vsc_count: int = Field(ge=0)


GREEN_FLAG = RaceControl(
    phase=RacePhase.GREEN,
    since_s=0.0,
    vsc_until_s=None,
    sc_distance_m=None,
    sc_laps_left=0,
    cause=None,
    sc_count=0,
    vsc_count=0,
)


EventType = Literal[
    "LAP_COMPLETED",
    "CAR_FINISHED",
    "RACE_COMPLETED",
    "PACE_MODE_CHANGED",
    "PIT_REQUESTED",
    "PIT_CANCELLED",
    "PIT_STOP",
    "STRATEGY_PLANNED",
    "CONTROLLER_CHANGED",
    "CHEQUERED_FLAG",
    "TYRE_RULE_BREACHED",
    "OVERTAKE",
    "BLUE_FLAG",
    "MISTAKE",
    "CAR_RETIRED",
    "VSC_DEPLOYED",
    "VSC_ENDED",
    "SAFETY_CAR_DEPLOYED",
    "SAFETY_CAR_IN",
    "RESTART",
    "RAIN_STARTED",
    "RAIN_STOPPED",
    "RED_FLAG",
    "TIME_PENALTY",
    "INTENT_CHANGED",
]


class RaceEvent(_Frozen):
    seq: int = Field(ge=0)
    tick: int = Field(ge=0)
    type: EventType
    driver_id: str | None
    payload: dict[str, float | int | str]


# --- actions: decisions that change the future ----------------------------------------------


class SetPaceMode(_Frozen):
    """Driver pace intent (plan §3.6, §9.3). Takes effect from the driver's NEXT lap.

    The duration of the lap in progress was sampled when it started, so a mid-lap change
    can't retroactively change it. This also keeps decisions lap-granular like the engine.
    """

    type: Literal["SET_PACE_MODE"] = "SET_PACE_MODE"
    driver_id: str
    pace_mode: PaceMode
    actor_id: str = "USER"


class RequestPitStop(_Frozen):
    """Box at the end of the current lap for ``compound`` (plan §7.5: one command per stop).

    Replaces any pending request. Takes the car's strategy over (controller -> USER).
    """

    type: Literal["REQUEST_PIT_STOP"] = "REQUEST_PIT_STOP"
    driver_id: str
    compound: Compound
    actor_id: str = "USER"


class CancelPitStop(_Frozen):
    """Withdraw a pending request (stay out). Takes the car's strategy over (-> USER)."""

    type: Literal["CANCEL_PIT_STOP"] = "CANCEL_PIT_STOP"
    driver_id: str
    actor_id: str = "USER"


class SetStrategyController(_Frozen):
    """Hand a car's pit decisions to the baseline (it replans from now) or to the user."""

    type: Literal["SET_STRATEGY_CONTROLLER"] = "SET_STRATEGY_CONTROLLER"
    driver_id: str
    controller: StrategyController
    actor_id: str = "USER"


class SetRacecraftIntent(_Frozen):
    """How hard to race the cars around (slice 4, D-048). Takes effect from the NEXT lap."""

    type: Literal["SET_RACECRAFT_INTENT"] = "SET_RACECRAFT_INTENT"
    driver_id: str
    intent: RacecraftIntent
    actor_id: str = "USER"


DriverAction = (
    SetPaceMode | RequestPitStop | CancelPitStop | SetStrategyController | SetRacecraftIntent
)
"""Union of all driver/team actions."""


class WorldState(_Frozen):
    schema_version: Literal[18]
    world_id: str
    timeline_id: str
    parent_timeline_id: str | None
    branch_event_seq: int | None = Field(ge=0)
    """For a branch: the parent's event count at the fork (plan §4.4 ``branch_point_event_id``).
    The branch's history = parent events before this + its own events."""
    world_status: WorldStatus
    mode: Literal["FREE_SIMULATION"]
    engine_version: str
    rng: RngSpec
    setup: RaceSetup
    sim_tick: int = Field(ge=0)
    sim_clock_s: float = Field(ge=0)
    cars: tuple[CarState, ...]
    chequered_flag_s: float | None
    """Sim time the leader took the flag; every later line crossing finishes that car."""
    race_control: RaceControl
    race_clock_s: float = Field(ge=0)
    """Racing clock: runs with sim time under green, slower under VSC (all lap progress is
    measured in it), and like sim time under SC (where the queue model moves cars)."""
    event_seq: int = Field(ge=0)
    """Number of events emitted so far; the next event gets this sequence number."""
    weather: WeatherState = Field(default_factory=WeatherState)

    @model_validator(mode="after")
    def _cars_match_setup(self) -> Self:
        if tuple(c.driver_id for c in self.cars) != tuple(d.driver_id for d in self.setup.drivers):
            raise ValueError("cars must match setup drivers one-to-one, in the same order")
        return self

    def car(self, driver_id: str) -> CarState:
        for c in self.cars:
            if c.driver_id == driver_id:
                return c
        raise KeyError(driver_id)

    def canonical_hash(self) -> str:
        return state_hash(self.model_dump(mode="json"))
