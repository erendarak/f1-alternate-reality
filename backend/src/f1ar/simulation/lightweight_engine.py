"""Lightweight deterministic race engine: pure ``step(state) -> (state, events)``.

Plan option C in miniature (§7.2): no physics. Cars move along the track at a speed implied
by their pre-sampled lap time. Each ``step`` advances exactly one bounded tick of
``setup.tick_seconds``, so pausing between ticks is always possible.

Pit stops (M2 slice 1): a stop happens at a line crossing (the pit box is treated as being on
the line). The car takes the new compound and its out-lap starts with a *hold* of pit-lane
loss + standstill time at the line, then the lap is driven as usual. The pit lane isn't drawn.

Chequered flag: once the leader completes the race distance, every other car finishes at its
next line crossing, so lapped cars finish a lap (or more) down, as in real racing.

Traffic (M2 slice 2, ``simulation/traffic.py``): when a car joins the track (line crossing,
leaving the pit lane, or the start in grid order) its lap is adjusted for dirty air, being
held up, overtaking and blue flags. Within a tick, joins are processed in time order, so every
car reacts to cars ahead whose laps are already fixed.

Race control (M2 slice 3, ``simulation/incidents.py``): mistakes cost time; crashes and
mechanical failures retire cars where they stop; race control may answer with a VSC (all lap
progress runs on a slowed racing clock, the pit lane doesn't slow down) or a safety car (a real
car on track that the field queues behind, no overtaking, then a restart).

Not modelled yet: start performance, weather, penalties, red flags.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import assert_never

from f1ar.domain.race_state import (
    GREEN_FLAG,
    WORLD_SCHEMA_VERSION,
    CancelPitStop,
    CarState,
    CarStatus,
    Compound,
    DriverAction,
    PaceMode,
    PendingOvertake,
    PlannedStop,
    RaceControl,
    RacecraftIntent,
    RaceEvent,
    RacePhase,
    RaceSetup,
    RequestPitStop,
    SetPaceMode,
    SetRacecraftIntent,
    SetStrategyController,
    StrategyController,
    WeatherMode,
    WeatherState,
    WorldState,
    WorldStatus,
    start_slot,
)
from f1ar.policy.baseline_strategy import (
    POLICY_ID,
    baseline_pace_mode,
    dry_compounds_required,
    free_change_compound,
    plan_for_car,
    team_setup,
    team_view,
    weather_stop,
)
from f1ar.simulation.incidents import (
    IncidentKind,
    Neutralisation,
    debris_neutralisation,
    lap_incident,
    mistake_loss_s,
    neutralisation_after,
    penalty_draw,
)
from f1ar.simulation.lap_model import life_use_per_lap, sample_lap_time_s, wear_per_lap_s
from f1ar.simulation.rng_streams import RngSpec, normal, uniform
from f1ar.simulation.traffic import Ahead, resolve
from f1ar.simulation.weather import (
    WET_TYRES,
    best_tyre,
    driver_factor,
    historical_rain,
    historical_wetness,
    is_slick,
    next_wetness,
    raining_at,
)
from f1ar.simulation.weather import penalty_s as weather_penalty_s

ENGINE_VERSION = "lightweight_engine_v0.19.0"
"""v0.19.0: hidden race-day truth (H3, D-058): with ``setup.knowledge`` the baseline plans with
the race-day prior's tyres while the race runs on the truth; without it, identical to v0.18.
v0.15.0: racecraft intent ATTACK / NORMAL / DEFEND per lap (risk for track position, D-048).
v0.14.0: fresh wet-weather tyres under SC/VSC and the wet incident factor, fitted on the 2024
wet races (D-046).
v0.13.0: overtaking refit with W2 noise + driver attack/defend skills, launch skill, lap-1
risk by grid position (D-044).
v0.12.0: driver race pace vs teammate, race form and lap noise per driver, fitted on 2024
(D-043).
v0.11.0: debris SC/VSC, red flags (free tyre change), pit crews and reliability per team,
time penalties, crash/mistake rates renormalised for the W1 pace mix (D-042).
v0.10.0: tyre life (management vs push, cliff), fresh-tyre peak, baseline manages (D-041).
v0.9.0: launch spread on lap 1; overtaking and race-control rates from 2024 data (D-035).
v0.8.0: weather: rain, track wetness, intermediate and wet tyres (D-034).
v0.7.0: calibrated tyre/fuel model, curved wear, planner for it (D-033).
v0.6.0: mistakes, crashes, failures (RETIRED), VSC (racing clock), safety car + restart.
v0.5.0: traffic: dirty air, held up, overtakes, blue flags; lap 1 resolved in grid order.
v0.4.0: pit stops + tyre choice, baseline strategy, chequered-flag rule (lapped finishers).
v0.3.0: per-car performance offset (``DriverEntry.car_pace_offset_s``).
v0.2.0: tyre wear accumulated per lap at the mode driven (v0.1.0 scaled it retroactively)."""


POLICY_MANIFEST: dict[str, str] = {"strategy": POLICY_ID}
"""Decision policies built into this engine. Worlds pin it (``RaceSetup.policy_manifest``)."""


class EngineStateError(RuntimeError):
    """An operation was requested in a world status that does not allow it."""


class ActionRejectedError(EngineStateError):
    """An action is illegal in the current state (unknown driver, finished car, ...)."""


@dataclass(frozen=True)
class StepResult:
    state: WorldState
    events: tuple[RaceEvent, ...]


_Emit = Callable[[str, str | None, dict[str, float | int | str]], None]


def _sample_next_lap(
    setup: RaceSetup,
    seed: int,
    driver_id: str,
    *,
    compound: Compound,
    pace_mode: PaceMode,
    laps_completed: int,
    tyre_wear_s: float,
    wetness: float = 0.0,
    tyre_life_used: float = 0.0,
    fresh_lap: bool = False,
    fuel_laps: float | None = None,
) -> float:
    """``fuel_laps``: laps of fuel on board (refuelling era, H1b); None = the race's fuel."""
    entry = setup.driver(driver_id)
    weather = setup.weather
    extra = weather_penalty_s(weather, compound, wetness) * driver_factor(
        weather, entry.profile.archetype.wet_skill
    )
    if laps_completed == 0 and setup.traffic.start_spread_s > 0.0:  # the launch
        extra += setup.traffic.start_spread_s * normal(seed, "launch", driver_id)
    if laps_completed == 0:
        extra += entry.launch_offset_s  # W3: the driver's launch skill
    return sample_lap_time_s(
        setup.lap_model,
        entry.profile.archetype,
        seed=seed,
        driver_id=driver_id,
        lap_number=laps_completed + 1,
        compound=compound,
        tyre_wear_s=tyre_wear_s,
        pace_mode=pace_mode,
        laps_remaining=setup.track.total_laps - laps_completed
        if fuel_laps is None
        else max(0, round(fuel_laps)),
        car_offset_s=entry.car_pace_offset_s
        + entry.driver_pace_offset_s  # W2: the driver vs his teammate
        + setup.lap_model.form_sd_s * normal(seed, "race_form", driver_id),  # and today's form
        extra_s=extra,
        noise_scale=1.0 + weather.wet_noise_factor * wetness,
        tyre_life_used=tyre_life_used,
        fresh_lap=fresh_lap,
        noise_sd=entry.lap_noise_s,
    )


def standstill_s(setup: RaceSetup, seed: int, driver_id: str, lap: int) -> float:
    """Stationary time of a stop at the end of ``lap``: log-logistic, keyed by (driver, lap).

    Inverse CDF of the fisk distribution: ``median * (u / (1 - u)) ** (1 / shape)``.
    """
    u = uniform(seed, "pit_standstill", driver_id, lap)
    p = setup.pit
    entry = setup.driver(driver_id)
    median = max(0.5, p.standstill_median_s + entry.pit_crew_offset_s)  # W4: the team's crew
    draw: float = min(median * (u / (1.0 - u)) ** (1.0 / p.standstill_shape), p.standstill_max_s)
    slow = p.slow_stop_probability * entry.pit_slow_multiplier
    if slow > 0.0 and uniform(seed, "pit_slow", driver_id, lap) < slow:  # W4: a stop goes wrong
        v = uniform(seed, "pit_slow_size", driver_id, lap)
        draw += p.slow_stop_min_s - p.slow_stop_scale_s * math.log(1.0 - v)
    return draw


def plan_text(plan: tuple[PlannedStop, ...]) -> str:
    """Compact plan description for events and displays, e.g. ``"L24 HARD, L40 MEDIUM"``."""
    return ", ".join(f"L{s.lap} {s.compound.value}" for s in plan) or "no stop"


def pending_stop(car: CarState) -> Compound | None:
    """Compound the car will pit for at its next line crossing, if any."""
    if car.pit_request is not None:
        return car.pit_request
    if (
        car.controller is StrategyController.BASELINE
        and car.planned_stops
        and car.planned_stops[0].lap == car.laps_completed + 1
    ):
        return car.planned_stops[0].compound
    return None


def _start_fuel(setup: RaceSetup, driver_id: str, compound: Compound) -> float | None:
    """Refuelling era (H1b): laps of fuel on the grid: fixed in qualifying (race-fuel rule) or
    the baseline plan's first stint; None without refuelling."""
    if not setup.rules.refuelling:
        return None
    entry = setup.driver(driver_id)
    if entry.start_fuel_laps is not None:
        return float(min(entry.start_fuel_laps, setup.track.total_laps))
    from f1ar.policy.refuel_strategy import best_plan_refuel

    plan = best_plan_refuel(
        team_setup(setup, driver_id),
        entry.profile.archetype,
        laps_completed=0,
        compound=compound,
        compounds_used=(compound,),
        fuel_laps=None,
        free_first=True,
    )
    return float(setup.track.total_laps if plan is None else plan.first_stint_laps)


def create_world(
    setup: RaceSetup, *, seed: int, world_id: str, timeline_id: str | None = None
) -> WorldState:
    """New world in CREATED status: all cars on the line at lap 0, first lap pre-sampled,
    every car under the baseline strategy with an initial plan."""
    if setup.policy_manifest != POLICY_MANIFEST:
        raise EngineStateError(
            f"setup pins policies {setup.policy_manifest}, this engine runs {POLICY_MANIFEST}"
        )
    historical = setup.weather.mode is WeatherMode.HISTORICAL
    weather = WeatherState(
        raining=historical and historical_rain(setup.weather, 0.0),
        wetness=historical_wetness(setup.weather, 0.0)
        if historical and setup.weather.wetness_timeline
        else setup.weather.initial_wetness,
    )
    cars = []
    for entry in setup.drivers:
        pace_mode = PaceMode.BALANCED
        start_fuel = _start_fuel(setup, entry.driver_id, entry.start_compound)
        target = _sample_next_lap(
            setup,
            seed,
            entry.driver_id,
            compound=entry.start_compound,
            pace_mode=pace_mode,
            laps_completed=0,
            tyre_wear_s=0.0,
            wetness=weather.wetness,
            fuel_laps=start_fuel,
        )
        car = CarState(
            driver_id=entry.driver_id,
            status=CarStatus.RUNNING,
            laps_completed=0,
            distance_m=0.0,
            lap_elapsed_s=0.0,
            lap_target_s=target,
            pit_hold_s=0.0,
            compound=entry.start_compound,
            tyre_age_laps=0,
            tyre_wear_s=0.0,
            pace_mode=pace_mode,
            lap_pace_mode=pace_mode,
            pit_request=None,
            pit_stops=0,
            compounds_used=(entry.start_compound,),
            controller=StrategyController.BASELINE,
            planned_stops=(),
            traffic_loss_s=0.0,
            last_lap_traffic_loss_s=0.0,
            carried_loss_s=0.0,
            pending_overtakes=(),
            incident_kind=None,
            incident_at_fraction=None,
            lap_started_at_s=0.0,
            last_lap_time_s=None,
            fuel_laps=start_fuel,
            finish_time_s=None,
            retired_reason=None,
            retired_at_s=None,
        )
        cars.append(car.model_copy(update={"planned_stops": plan_for_car(setup, car)}))
    return WorldState(
        schema_version=WORLD_SCHEMA_VERSION,
        world_id=world_id,
        timeline_id=timeline_id or f"{world_id}.main",
        parent_timeline_id=None,
        branch_event_seq=None,
        world_status=WorldStatus.CREATED,
        mode="FREE_SIMULATION",
        engine_version=ENGINE_VERSION,
        rng=RngSpec(seed=seed),
        setup=setup,
        sim_tick=0,
        sim_clock_s=0.0,
        cars=tuple(cars),
        chequered_flag_s=None,
        race_control=GREEN_FLAG,
        race_clock_s=0.0,
        event_seq=0,
        weather=weather,
    )


def start(state: WorldState) -> WorldState:
    if state.world_status is not WorldStatus.CREATED:
        raise EngineStateError(f"cannot start a world in status {state.world_status}")
    return state.model_copy(update={"world_status": WorldStatus.RUNNING})


def pause(state: WorldState) -> WorldState:
    """RUNNING -> PAUSED. Only the status changes; tick, clocks, cars and events are frozen."""
    if state.world_status is not WorldStatus.RUNNING:
        raise EngineStateError(f"cannot pause a world in status {state.world_status}")
    return state.model_copy(update={"world_status": WorldStatus.PAUSED})


def resume(state: WorldState) -> WorldState:
    if state.world_status is not WorldStatus.PAUSED:
        raise EngineStateError(f"cannot resume a world in status {state.world_status}")
    return state.model_copy(update={"world_status": WorldStatus.RUNNING})


# --- actions ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class ValidActions:
    """What ``apply_action`` will accept for one driver right now (plan §7.3, §9.3 action masks).

    Re-requesting the current pace mode or compound is legal (an accepted no-op); a cancel is
    only listed when there's a stop to cancel.
    """

    driver_id: str
    pace_modes: tuple[PaceMode, ...]
    pit_compounds: tuple[Compound, ...]
    cancel_pit: bool
    controllers: tuple[StrategyController, ...]
    intents: tuple[RacecraftIntent, ...] = ()

    @property
    def any(self) -> bool:
        return bool(self.pace_modes or self.pit_compounds or self.cancel_pit or self.controllers)


def valid_actions(state: WorldState, driver_id: str) -> ValidActions:
    """Legal actions for ``driver_id`` (same rules as ``apply_action``, without trying them)."""
    car = state.car(driver_id)
    if state.world_status is WorldStatus.COMPLETED or car.status is not CarStatus.RUNNING:
        return ValidActions(driver_id, (), (), False, (), ())
    can_pit = (
        state.chequered_flag_s is None and car.laps_completed + 1 < state.setup.track.total_laps
    )
    return ValidActions(
        driver_id=driver_id,
        pace_modes=tuple(PaceMode),
        pit_compounds=tuple(Compound) if can_pit else (),
        cancel_pit=pending_stop(car) is not None,
        controllers=tuple(StrategyController),
        intents=tuple(RacecraftIntent),
    )


def apply_action(state: WorldState, action: DriverAction) -> StepResult:
    """Apply a driver/team action between ticks. Pure; returns the new state and its events.

    Requests that change nothing (same pace mode, same pending compound, nothing to cancel)
    are accepted no-ops without events.
    """
    if state.world_status is WorldStatus.COMPLETED:
        raise ActionRejectedError("the race is over")
    if action.driver_id not in {c.driver_id for c in state.cars}:
        raise ActionRejectedError(f"unknown driver {action.driver_id!r}")
    car = state.car(action.driver_id)
    if car.status is not CarStatus.RUNNING:
        raise ActionRejectedError(f"{action.driver_id} is {car.status}, not RUNNING")

    events: list[tuple[str, dict[str, float | int | str]]] = []

    def emit(type_: str, payload: dict[str, float | int | str]) -> None:
        events.append((type_, {**payload, "actor_id": action.actor_id}))

    match action:
        case SetPaceMode():
            new = _set_pace_mode(state, car, action, emit)
        case RequestPitStop():
            new = _request_pit(state, car, action, emit)
        case CancelPitStop():
            new = _cancel_pit(car, action.actor_id, emit)
        case SetStrategyController():
            new = _set_controller(state, car, action.controller, emit)
        case SetRacecraftIntent():
            new = _set_intent(car, action, emit)
        case _:
            assert_never(action)

    if not events:
        return StepResult(state, ())
    race_events = tuple(
        RaceEvent.model_validate(
            {
                "seq": state.event_seq + i,
                "tick": state.sim_tick,
                "type": type_,
                "driver_id": car.driver_id,
                "payload": payload,
            }
        )
        for i, (type_, payload) in enumerate(events)
    )
    cars = tuple(new if c.driver_id == car.driver_id else c for c in state.cars)
    return StepResult(
        state.model_copy(update={"cars": cars, "event_seq": state.event_seq + len(events)}),
        race_events,
    )


_ActionEmit = Callable[[str, dict[str, float | int | str]], None]


def _replan(state: WorldState, car: CarState, emit: _ActionEmit) -> CarState:
    plan = plan_for_car(state.setup, car)
    if plan != car.planned_stops:
        emit("STRATEGY_PLANNED", {"plan": plan_text(plan), "policy_id": POLICY_ID})
    return car.model_copy(update={"planned_stops": plan})


def _set_pace_mode(
    state: WorldState, car: CarState, action: SetPaceMode, emit: _ActionEmit
) -> CarState:
    if car.controller is StrategyController.AGENT and not action.actor_id.startswith("AGENT"):
        car = _take_over(car, emit)  # a human decision on an AI car: the human has it now
    if not action.actor_id.startswith("AGENT") and not car.pace_by_user:
        car = car.model_copy(update={"pace_by_user": True})  # W1: no automatic management now
    if car.pace_mode is action.pace_mode:
        return car
    emit(
        "PACE_MODE_CHANGED",
        {
            "from": car.pace_mode.value,
            "to": action.pace_mode.value,
            "effective_from_lap": car.laps_completed + 2,
        },
    )
    new = car.model_copy(update={"pace_mode": action.pace_mode})
    if new.controller is StrategyController.BASELINE:
        new = _replan(state, new, emit)  # a different wear rate moves the best stop lap
    return new


def _set_intent(car: CarState, action: SetRacecraftIntent, emit: _ActionEmit) -> CarState:
    if car.controller is StrategyController.AGENT and not action.actor_id.startswith("AGENT"):
        car = _take_over(car, emit)  # a human decision on an AI car: the human has it now
    if car.intent is action.intent:
        return car
    emit(
        "INTENT_CHANGED",
        {
            "from": car.intent.value,
            "to": action.intent.value,
            "effective_from_lap": car.laps_completed + 2,
        },
    )
    return car.model_copy(update={"intent": action.intent})


def _controller_of(actor_id: str) -> StrategyController:
    """Who takes a car over by acting on it: an agent runner, or else a human."""
    return StrategyController.AGENT if actor_id.startswith("AGENT") else StrategyController.USER


def _take_over(
    car: CarState, emit: _ActionEmit, to: StrategyController = StrategyController.USER
) -> CarState:
    if car.controller is to:
        return car
    emit("CONTROLLER_CHANGED", {"from": car.controller.value, "to": to.value})
    # A stop the baseline planned for this lap becomes an explicit request, so taking over
    # never silently changes what the car was about to do.
    request = pending_stop(car)
    return car.model_copy(
        update={
            "controller": to,
            "planned_stops": (),
            "pit_request": request,
        }
    )


def _request_pit(
    state: WorldState, car: CarState, action: RequestPitStop, emit: _ActionEmit
) -> CarState:
    if state.chequered_flag_s is not None:
        raise ActionRejectedError("the chequered flag is out")
    if car.laps_completed + 1 >= state.setup.track.total_laps:
        raise ActionRejectedError(f"{car.driver_id} is on the final lap")
    new = _take_over(car, emit, _controller_of(action.actor_id))
    if new.pit_request is action.compound:
        return new
    emit(
        "PIT_REQUESTED",
        {"compound": action.compound.value, "in_lap": new.laps_completed + 1},
    )
    return new.model_copy(update={"pit_request": action.compound})


def _cancel_pit(car: CarState, actor_id: str, emit: _ActionEmit) -> CarState:
    if pending_stop(car) is None:
        return car  # nothing to cancel; a BASELINE car keeps its plan
    new = _take_over(car, emit, _controller_of(actor_id))
    assert new.pit_request is not None
    emit("PIT_CANCELLED", {"compound": new.pit_request.value, "in_lap": new.laps_completed + 1})
    return new.model_copy(update={"pit_request": None})


def _set_controller(
    state: WorldState, car: CarState, controller: StrategyController, emit: _ActionEmit
) -> CarState:
    if car.controller is controller:
        if controller is StrategyController.BASELINE and (
            car.pace_by_user or car.intent is not RacecraftIntent.NORMAL
        ):
            # W1: the pace (tyre management) is handed back to the baseline as well; slice 4:
            # and the racecraft (the baseline races NORMAL)
            emit("CONTROLLER_CHANGED", {"from": "BASELINE", "to": "BASELINE", "pace": "automatic"})
            return car.model_copy(update={"pace_by_user": False, "intent": RacecraftIntent.NORMAL})
        return car
    if controller is not StrategyController.BASELINE:
        return _take_over(car, emit, controller)
    emit("CONTROLLER_CHANGED", {"from": car.controller.value, "to": controller.value})
    # The baseline decides from here on; a pending user request is replaced by its plan.
    new = car.model_copy(
        update={
            "controller": controller,
            "pit_request": None,
            "pace_by_user": False,
            "intent": RacecraftIntent.NORMAL,  # the baseline races NORMAL (slice 4)
        }
    )
    return _replan(state, new, emit)


# --- stepping ---------------------------------------------------------------------------------


def step(state: WorldState, *, allow_paused: bool = False) -> StepResult:
    """Advance the world by exactly one tick. Pure: ``state`` is not modified.

    ``allow_paused`` supports manual single-stepping of a paused world ("Step Tick"); the
    world stays PAUSED afterwards (or becomes COMPLETED if that tick ends the race).
    """
    allowed = {WorldStatus.RUNNING, WorldStatus.PAUSED} if allow_paused else {WorldStatus.RUNNING}
    if state.world_status not in allowed:
        raise EngineStateError(f"cannot step a world in status {state.world_status}")

    setup = state.setup
    tick = state.sim_tick + 1
    events: list[RaceEvent] = []

    def emit(type_: str, driver_id: str | None, payload: dict[str, float | int | str]) -> None:
        events.append(
            RaceEvent.model_validate(
                {
                    "seq": state.event_seq + len(events),
                    "tick": tick,
                    "type": type_,
                    "driver_id": driver_id,
                    "payload": payload,
                }
            )
        )

    if state.race_control.phase is RacePhase.SC:
        moved = _advance_safety_car(state, emit)
    else:
        moved = _advance_racing(state, emit)
    tick_end = tick * setup.tick_seconds
    weather = _advance_weather(state, tick_end, emit)
    cars, race_control = _incidents_and_race_control(state, moved, tick_end, emit)
    cars = _announce_overtakes(state, cars, race_control, emit)

    if moved.new_flag_s is not None:
        winner = min(
            (c for c in cars if c.finish_time_s is not None), key=lambda c: c.finish_time_s or 0.0
        )
        emit("CHEQUERED_FLAG", winner.driver_id, {"flag_time_s": moved.new_flag_s})

    status: WorldStatus = state.world_status
    if all(c.status is not CarStatus.RUNNING for c in cars):
        status = WorldStatus.COMPLETED
        emit("RACE_COMPLETED", None, {"tick": tick})

    new_state = state.model_copy(
        update={
            "sim_tick": tick,
            # Derived from the tick count, not accumulated, so the clock never drifts.
            "sim_clock_s": tick_end,
            "race_clock_s": moved.race_clock_s,
            "cars": cars,
            "chequered_flag_s": moved.flag_s,
            "race_control": race_control,
            "world_status": status,
            "event_seq": state.event_seq + len(events),
            "weather": weather,
        }
    )
    return StepResult(new_state, tuple(events))


def _advance_weather(state: WorldState, tick_end: float, emit: _Emit) -> WeatherState:
    """Rain during this tick (timeline or keyed process) and the wetness at its end. Weather is
    exogenous: decisions never change it, so every fork with the same seed shares it."""
    params, dt = state.setup.weather, state.setup.tick_seconds
    before = state.weather
    raining = raining_at(params, state.rng.seed, tick_end, before.raining, dt)
    if raining != before.raining:
        emit(
            "RAIN_STARTED" if raining else "RAIN_STOPPED",
            None,
            {"race_time_s": tick_end, "wetness": round(before.wetness, 3)},
        )
    if params.mode is WeatherMode.HISTORICAL and params.wetness_timeline:
        wetness = historical_wetness(params, tick_end)  # the real race's conditions
    else:
        wetness = next_wetness(params, before.wetness, raining, dt)
    return WeatherState(raining=raining, wetness=min(1.0, max(0.0, wetness)))


@dataclass
class _Moved:
    cars: list[CarState]
    race_clock_s: float
    flag_s: float | None
    new_flag_s: float | None
    race_control: RaceControl
    restart: bool = False
    """The safety car pulled in during this tick: green from the end of the tick."""


def _distance(length: float, elapsed: float, target: float, hold: float) -> float:
    """Position in the lap: parked at the line during the pit hold, then constant speed."""
    if elapsed <= hold:
        return 0.0
    return min(length * (elapsed - hold) / (target - hold), math.nextafter(length, 0.0))


def _in_pit_lane(car: CarState) -> bool:
    return car.lap_elapsed_s < car.pit_hold_s


# --- green flag and VSC: lap-sampled motion on the racing clock -------------------------------


def _advance_racing(state: WorldState, emit: _Emit) -> _Moved:
    """Green or VSC tick. Lap progress runs on the racing clock (``k`` x sim time under VSC);
    pit-lane holds run in sim time, which is why stops under VSC are cheap. Line crossings and
    pit exits inside the tick are processed in time order (traffic, M2 slice 2)."""
    setup = state.setup
    dt = setup.tick_seconds
    vsc = state.race_control.phase is RacePhase.VSC
    k = setup.race_control.vsc_speed_fraction if vsc else 1.0
    t0, tau0 = state.sim_clock_s, state.race_clock_s
    tau1 = tau0 + k * dt
    cars = list(state.cars)

    def to_sim(tau: float) -> float:
        return t0 + (tau - tau0) / k

    # Racing-clock time each car's current lap started. A car in the pit lane leaves it after
    # (hold - elapsed) sim seconds, i.e. at tau0 + k (hold - elapsed) on the racing clock.
    lap_start = [
        tau0 + k * (c.pit_hold_s - c.lap_elapsed_s) - c.pit_hold_s
        if _in_pit_lane(c)
        else tau0 - c.lap_elapsed_s
        for c in cars
    ]
    if state.sim_tick == 0:
        _resolve_start(state, cars, emit)

    flag_s = state.chequered_flag_s
    new_flag = None
    if flag_s is None:
        final_lap = setup.track.total_laps - 1
        ends = [
            lap_start[i] + c.lap_target_s
            for i, c in enumerate(cars)
            if c.status is CarStatus.RUNNING
            and not _in_pit_lane(c)
            and c.laps_completed == final_lap
            and lap_start[i] + c.lap_target_s <= tau1
        ]
        if ends:
            flag_s = new_flag = to_sim(min(ends))

    due: list[tuple[float, int, str]] = []
    for i, c in enumerate(cars):
        if c.status is not CarStatus.RUNNING:
            continue
        if _in_pit_lane(c):
            if lap_start[i] + c.pit_hold_s <= tau1:
                due.append((lap_start[i] + c.pit_hold_s, i, "rejoin"))
        elif lap_start[i] + c.lap_target_s <= tau1:
            due.append((lap_start[i] + c.lap_target_s, i, "line"))

    crossed_at: dict[int, float] = {}  # car index -> sim time it crossed the line this tick
    for tau, i, kind in sorted(due):
        if kind == "line":
            crossed = to_sim(tau)
            cars[i] = _complete_lap(state, cars[i], crossed, flag_s, emit)
            if cars[i].status is CarStatus.FINISHED:
                continue
            crossed_at[i] = crossed
            hold = cars[i].pit_hold_s
            lap_start[i] = tau + k * hold - hold
            if hold > 0.0 and lap_start[i] + hold > tau1:
                continue  # in the pit lane; it joins the track at the end of its hold
            tau = lap_start[i] + hold
        _join_track(state, i, tau, cars, lap_start, emit)

    for i, c in enumerate(cars):
        if c.status is not CarStatus.RUNNING:
            continue
        if lap_start[i] + c.pit_hold_s > tau1:  # still in the pit lane: sim-time progress
            elapsed = t0 + dt - crossed_at[i] if i in crossed_at else c.lap_elapsed_s + dt
            elapsed = min(elapsed, math.nextafter(c.pit_hold_s, 0.0))
        else:
            elapsed = max(tau1 - lap_start[i], c.pit_hold_s)
        cars[i] = c.model_copy(
            update={
                "lap_elapsed_s": elapsed,
                "distance_m": _distance(
                    setup.track.length_m, elapsed, c.lap_target_s, c.pit_hold_s
                ),
            }
        )
    return _Moved(cars, tau1, flag_s, new_flag, state.race_control)


def _resolve_start(state: WorldState, cars: list[CarState], emit: _Emit) -> None:
    """Lap 1: everyone leaves the line at t=0; the road order is the grid order."""
    grid = sorted(
        range(len(cars)),
        key=lambda i: start_slot(state.setup.driver(cars[i].driver_id).grid_position),
    )
    for k, i in enumerate(grid):
        cars[i] = _with_incident(state, cars[i])
        ahead = [Ahead(cars[j], 0.0, 0.0) for j in reversed(grid[:k])]
        _apply_traffic(state, i, 0.0, ahead, cars, emit)


def _join_track(
    state: WorldState,
    i: int,
    tau: float,
    cars: list[CarState],
    lap_start: list[float],
    emit: _Emit,
    *,
    context: str = "",
) -> None:
    """Car ``i`` joins the track at racing-clock time ``tau``: resolve the cars ahead of it.

    Usually it's at the line (crossing, pit exit). At a restart it's anywhere in the lap.
    """
    me = cars[i]
    my_fraction = _lap_fraction(me, tau - lap_start[i])
    ahead: list[tuple[float, int, float]] = []
    for j, other in enumerate(cars):
        if j == i or other.status is not CarStatus.RUNNING:
            continue
        driving = tau - lap_start[j] - other.pit_hold_s
        if driving <= 0.0:
            continue  # parked in the pit lane: not on the road ahead
        fraction = driving / (other.lap_target_s - other.pit_hold_s)
        if fraction <= my_fraction:
            continue
        gap = (fraction - my_fraction) * (other.lap_target_s - other.pit_hold_s)
        ahead.append((fraction, j, gap))
    ahead.sort()
    _apply_traffic(
        state,
        i,
        lap_start[i],
        [Ahead(cars[j], lap_start[j], gap) for _, j, gap in ahead],
        cars,
        emit,
        context=context,
    )


def _lap_fraction(car: CarState, elapsed: float) -> float:
    if elapsed <= car.pit_hold_s:
        return 0.0
    return (elapsed - car.pit_hold_s) / (car.lap_target_s - car.pit_hold_s)


def _apply_traffic(
    state: WorldState,
    i: int,
    my_lap_start: float,
    ahead: list[Ahead],
    cars: list[CarState],
    emit: _Emit,
    *,
    context: str = "",
) -> None:
    me = cars[i]
    out = resolve(state.setup, state.rng.seed, me, my_lap_start, ahead, context=context)
    pending = list(me.pending_overtakes)
    for type_, payload in out.events:
        if type_ == "OVERTAKE":  # announced when it happens on the road
            pending = [p for p in pending if p.rival != payload["passed"]]
            pending.append(
                PendingOvertake(
                    rival=str(payload["passed"]),
                    lap=int(payload["lap"]),
                    margin_s=float(payload["margin_s"]),
                    probability=float(payload["probability"]),
                )
            )
        else:
            emit(type_, me.driver_id, payload)
    cars[i] = me.model_copy(
        update={
            "lap_target_s": out.lap_target_s,
            "traffic_loss_s": me.traffic_loss_s + out.lap_target_s - me.lap_target_s,
            "pending_overtakes": tuple(pending),
        }
    )
    for j, other in enumerate(cars):
        loss = out.carried_losses.get(other.driver_id)
        if loss:
            cars[j] = other.model_copy(update={"carried_loss_s": other.carried_loss_s + loss})


# --- safety car: the field queues behind a real safety car --------------------------------------


def _advance_safety_car(state: WorldState, emit: _Emit) -> _Moved:
    """SC tick. The safety car laps at ``sc_lap_factor`` x base lap; cars close up at
    ``sc_catch_up_lap_factor`` x base lap but never closer than ``sc_spacing_m`` to the car (or
    safety car) ahead, so nobody overtakes. Cars are processed from the safety car backwards
    around the lap. Pit-lane holds run in sim time, so a stop costs little track position."""
    setup = state.setup
    p = setup.race_control
    length = setup.track.length_m
    dt = setup.tick_seconds
    t0 = state.sim_clock_s
    rc = state.race_control
    assert rc.sc_distance_m is not None
    cars = list(state.cars)
    flag_s = state.chequered_flag_s
    new_flag = None

    sc_from = rc.sc_distance_m
    sc_travel = length / (setup.lap_model.base_lap_s * p.sc_lap_factor) * dt
    catch_up = length / (setup.lap_model.base_lap_s * p.sc_catch_up_lap_factor) * dt
    laps_left = rc.sc_laps_left
    restart = False
    if sc_from + sc_travel >= length:
        laps_left -= 1
        restart = laps_left <= 0
    sc_to = (sc_from + sc_travel) % length

    on_road = [
        i for i, c in enumerate(cars) if c.status is CarStatus.RUNNING and not _in_pit_lane(c)
    ]
    on_road.sort(key=lambda i: ((sc_from - cars[i].distance_m) % length, i))
    behind_ahead = 0.0  # distance behind the safety car of the car ahead, after its move
    for i in on_road:
        c = cars[i]
        behind = (sc_from - c.distance_m) % length
        room = behind + sc_travel - (behind_ahead + p.sc_spacing_m)
        travel = max(0.0, min(catch_up, room))
        behind_ahead = behind + sc_travel - travel
        new_distance = c.distance_m + travel
        if new_distance < length:
            cars[i] = c.model_copy(
                update={"distance_m": new_distance, "lap_elapsed_s": c.lap_elapsed_s + dt}
            )
            continue
        crossed = t0 + dt * (length - c.distance_m) / travel
        if flag_s is None and c.laps_completed == setup.track.total_laps - 1:
            flag_s = new_flag = crossed
        done = _complete_lap(state, c, crossed, flag_s, emit)
        if done.status is not CarStatus.RUNNING:
            cars[i] = done
            continue
        if done.pit_hold_s > 0.0:  # into the pit lane: hold runs in sim time
            cars[i] = done.model_copy(update={"lap_elapsed_s": t0 + dt - crossed})
        else:
            cars[i] = done.model_copy(
                update={
                    "distance_m": new_distance - length,
                    "lap_elapsed_s": max(t0 + dt - crossed, done.pit_hold_s),
                }
            )

    for i, c in enumerate(cars):  # cars that were already in the pit lane
        if (
            c.status is CarStatus.RUNNING
            and i not in on_road
            and _in_pit_lane(state.cars[i])
            and state.cars[i].laps_completed == c.laps_completed  # didn't just cross
        ):
            cars[i] = c.model_copy(
                update={"lap_elapsed_s": min(c.lap_elapsed_s + dt, c.pit_hold_s)}
            )

    race_control = rc.model_copy(update={"sc_distance_m": sc_to, "sc_laps_left": max(laps_left, 0)})
    return _Moved(cars, state.race_clock_s + dt, flag_s, new_flag, race_control, restart=restart)


def _restart(state: WorldState, cars: list[CarState], tau: float, emit: _Emit) -> list[CarState]:
    """Green again after the safety car: each car's lap is re-expressed from its position (the
    rest of its lap at racing pace), then traffic is resolved front to back, as at a join."""
    length = state.setup.track.length_m
    lap_start = [tau - c.lap_elapsed_s for c in cars]
    for i, c in enumerate(cars):
        if c.status is not CarStatus.RUNNING or _in_pit_lane(c):
            continue
        hold = c.pit_hold_s
        elapsed = hold + (c.lap_target_s - hold) * (c.distance_m / length)
        cars[i] = c.model_copy(update={"lap_elapsed_s": elapsed})
        lap_start[i] = tau - elapsed
    road = [i for i, c in enumerate(cars) if c.status is CarStatus.RUNNING and not _in_pit_lane(c)]
    road.sort(key=lambda i: (-(cars[i].laps_completed + cars[i].distance_m / length), i))
    context = f"restart{state.race_control.sc_count}"
    for i in road:
        _join_track(state, i, tau, cars, lap_start, emit, context=context)
        # Traffic fixed this car's arrival at the line (lap start + new target). It's mid-lap,
        # so keep its position and stretch the rest of the lap to arrive then; cars resolved
        # after it see the same position and the same arrival time.
        c = cars[i]
        fraction = c.distance_m / length
        remaining = lap_start[i] + c.lap_target_s - tau
        drive = remaining / (1.0 - fraction)
        elapsed = c.pit_hold_s + fraction * drive
        cars[i] = c.model_copy(
            update={"lap_target_s": c.pit_hold_s + drive, "lap_elapsed_s": elapsed}
        )
        lap_start[i] = tau - elapsed
    return cars


# --- incidents and race control ---------------------------------------------------------------


def _incidents_and_race_control(
    state: WorldState, moved: _Moved, now: float, emit: _Emit
) -> tuple[tuple[CarState, ...], RaceControl]:
    """Mistakes and retirements reached during this tick, then race-control transitions.

    A neutralisation decided in this tick starts with the next tick (the phase is constant
    within a tick).
    """
    setup = state.setup
    length = setup.track.length_m
    cars = moved.cars
    rc = moved.race_control
    seed = state.rng.seed

    for i, c in enumerate(cars):
        at, kind_name = c.incident_at_fraction, c.incident_kind
        if c.status is not CarStatus.RUNNING or at is None or kind_name is None:
            continue
        if _in_pit_lane(c):
            continue
        before = state.cars[i]
        start = before.distance_m / length if before.laps_completed == c.laps_completed else 0.0
        if not start < at <= c.distance_m / length:
            continue
        lap = c.laps_completed + 1
        kind = IncidentKind(kind_name)
        if kind is IncidentKind.MISTAKE:
            loss = mistake_loss_s(setup, seed, c.driver_id, lap)
            emit("MISTAKE", c.driver_id, {"lap": lap, "time_loss_s": loss})
            cars[i] = c.model_copy(update={"incident_kind": None, "incident_at_fraction": None})
            continue
        cars[i] = c.model_copy(
            update={
                "status": CarStatus.RETIRED,
                "distance_m": at * length,
                "retired_reason": kind.value,
                "retired_at_s": now,
                "pit_request": None,
                "planned_stops": (),
                "pending_overtakes": (),
                "incident_kind": None,
                "incident_at_fraction": None,
            }
        )
        emit(
            "CAR_RETIRED",
            c.driver_id,
            {"lap": lap, "laps": c.laps_completed, "reason": kind.value, "lap_fraction": at},
        )
        if moved.flag_s is None:
            rc = _neutralise(state, cars, rc, c, kind, now, emit)

    # W4: a neutralisation without a retirement, drawn when the leader starts a new lap
    lead_before = max((c.laps_completed for c in state.cars), default=0)
    lead_now = max((c.laps_completed for c in cars), default=0)
    if (
        lead_now > lead_before
        and lead_now >= 1
        and moved.flag_s is None
        and rc.phase is RacePhase.GREEN
        and lead_now < setup.track.total_laps - 1
    ):
        debris = debris_neutralisation(setup, seed, lead_now + 1)
        if debris is not None:
            rc = _deploy(state, cars, rc, debris, now, emit, cause=None)

    if moved.restart:
        emit("SAFETY_CAR_IN", None, {"sc_number": rc.sc_count})
        cars = _restart(state, cars, moved.race_clock_s, emit)
        rc = rc.model_copy(
            update={
                "phase": RacePhase.GREEN,
                "since_s": now,
                "sc_distance_m": None,
                "sc_laps_left": 0,
                "cause": None,
            }
        )
        emit("RESTART", None, {"after": "SC"})
    elif rc.phase is RacePhase.VSC and rc.vsc_until_s is not None and now >= rc.vsc_until_s:
        emit("VSC_ENDED", None, {"vsc_number": rc.vsc_count})
        rc = rc.model_copy(
            update={"phase": RacePhase.GREEN, "since_s": now, "vsc_until_s": None, "cause": None}
        )
    return tuple(cars), rc


def _neutralise(
    state: WorldState,
    cars: list[CarState],
    rc: RaceControl,
    car: CarState,
    kind: IncidentKind,
    now: float,
    emit: _Emit,
) -> RaceControl:
    if rc.phase is RacePhase.SC:
        return rc
    decision = neutralisation_after(
        state.setup, state.rng.seed, car.driver_id, car.laps_completed + 1, kind
    )
    if decision is None:
        return rc
    if decision.red_flag:
        emit("RED_FLAG", car.driver_id, {"lap": car.laps_completed + 1})
        _red_flag_tyres(state, cars, emit)
    return _deploy(state, cars, rc, decision, now, emit, cause=car.driver_id)


def _red_flag_tyres(state: WorldState, cars: list[CarState], emit: _Emit) -> None:
    """Red flag (W4): every running car gets a set of new tyres for free (2024 rules), chosen by
    the baseline planner for BASELINE cars; a USER/AGENT car takes its requested compound, or a
    new set of the same one. It counts for the two-compound rule, not as a pit stop."""
    setup = state.setup
    for i, c in enumerate(cars):
        if c.status is not CarStatus.RUNNING:
            continue
        if c.controller is StrategyController.BASELINE:
            compound = free_change_compound(setup, c, state.weather.wetness)
        else:
            compound = c.pit_request or c.compound
        used = c.compounds_used if compound in c.compounds_used else (*c.compounds_used, compound)
        new = c.model_copy(
            update={
                "compound": compound,
                "tyre_age_laps": 0,
                "tyre_wear_s": 0.0,
                "tyre_life_used": 0.0,
                "compounds_used": used,
                "pit_request": None,
            }
        )
        if new.controller is StrategyController.BASELINE:
            new = new.model_copy(update={"planned_stops": plan_for_car(setup, new)})
        emit(
            "PIT_STOP",
            c.driver_id,
            {
                "lap": c.laps_completed,
                "from_compound": c.compound.value,
                "to_compound": compound.value,
                "tyre_age_laps": c.tyre_age_laps,
                "standstill_s": 0.0,
                "pit_loss_s": 0.0,
                "stop_number": c.pit_stops,
                "under": "RED_FLAG",
            },
        )
        cars[i] = new


def _deploy(
    state: WorldState,
    cars: list[CarState] | tuple[CarState, ...],
    rc: RaceControl,
    decision: Neutralisation,
    now: float,
    emit: _Emit,
    *,
    cause: str | None,
    actor_id: str = "RACE_CONTROL",
) -> RaceControl:
    """Start a safety car (ahead of the leader) or a VSC; a VSC during a VSC extends it."""
    if decision.phase is RacePhase.SC:
        length = state.setup.track.length_m
        road = [c for c in cars if c.status is CarStatus.RUNNING and not _in_pit_lane(c)]
        if not road:
            return rc
        leader = max(road, key=lambda c: (c.laps_completed, c.distance_m))
        sc_at = (leader.distance_m + state.setup.race_control.sc_deploy_ahead_m) % length
        emit(
            "SAFETY_CAR_DEPLOYED",
            cause,
            {"laps": decision.sc_laps, "sc_number": rc.sc_count + 1, "actor_id": actor_id},
        )
        return rc.model_copy(
            update={
                "phase": RacePhase.SC,
                "since_s": now,
                "vsc_until_s": None,
                "sc_distance_m": sc_at,
                "sc_laps_left": decision.sc_laps,
                "cause": cause,
                "sc_count": rc.sc_count + 1,
            }
        )
    until = now + decision.vsc_s
    if rc.phase is RacePhase.VSC:  # already under VSC: extend it
        return rc.model_copy(update={"vsc_until_s": max(rc.vsc_until_s or now, until)})
    emit(
        "VSC_DEPLOYED",
        cause,
        {"duration_s": decision.vsc_s, "vsc_number": rc.vsc_count + 1, "actor_id": actor_id},
    )
    return rc.model_copy(
        update={
            "phase": RacePhase.VSC,
            "since_s": now,
            "vsc_until_s": until,
            "cause": cause,
            "vsc_count": rc.vsc_count + 1,
        }
    )


def deploy_neutralisation(
    state: WorldState,
    phase: RacePhase,
    *,
    sc_laps: int = 4,
    vsc_s: float = 90.0,
    actor_id: str = "USER",
) -> StepResult:
    """Race-director action between ticks: send out the safety car or start a VSC now.

    For exploring scenarios ("what if a safety car came out on lap 18?"). It's a command like
    any other, so it is logged and a save/fork/replay reproduces it. A VSC can be upgraded to
    a safety car; a VSC during a VSC extends it.
    """
    if state.world_status is WorldStatus.COMPLETED or state.chequered_flag_s is not None:
        raise ActionRejectedError("the race is over")
    if phase is RacePhase.GREEN:
        raise ActionRejectedError("deploy SC or VSC, not GREEN")
    if state.race_control.phase is RacePhase.SC:
        raise ActionRejectedError("the safety car is already out")
    if phase is RacePhase.VSC and not state.setup.rules.vsc:
        season = state.setup.rules.season
        raise ActionRejectedError(f"no virtual safety car in {season} (introduced 2015)")
    if sc_laps < 1 or vsc_s <= 0:
        raise ActionRejectedError("the safety car needs >= 1 lap and a VSC a positive duration")
    events: list[RaceEvent] = []

    def emit(type_: str, driver_id: str | None, payload: dict[str, float | int | str]) -> None:
        events.append(
            RaceEvent.model_validate(
                {
                    "seq": state.event_seq + len(events),
                    "tick": state.sim_tick,
                    "type": type_,
                    "driver_id": driver_id,
                    "payload": payload,
                }
            )
        )

    decision = Neutralisation(phase, vsc_s=vsc_s, sc_laps=sc_laps)
    rc = _deploy(
        state,
        state.cars,
        state.race_control,
        decision,
        state.sim_clock_s,
        emit,
        cause=None,
        actor_id=actor_id,
    )
    if not events:
        raise ActionRejectedError("no car on track to start a safety car behind")
    return StepResult(
        state.model_copy(update={"race_control": rc, "event_seq": state.event_seq + len(events)}),
        tuple(events),
    )


def _announce_overtakes(
    state: WorldState, cars: tuple[CarState, ...], race_control: RaceControl, emit: _Emit
) -> tuple[CarState, ...]:
    """Emit OVERTAKE for every pending pass that has now happened on the road (the attacker
    is further into the lap than the rival). Under a safety car all pending passes are void:
    the field queues in its current road order."""
    by_id = {c.driver_id: c for c in cars}
    out = list(cars)
    for i, c in enumerate(cars):
        if not c.pending_overtakes:
            continue
        if race_control.phase is RacePhase.SC or c.status is not CarStatus.RUNNING:
            out[i] = c.model_copy(update={"pending_overtakes": ()})
            continue
        keep = []
        for p in c.pending_overtakes:
            rival = by_id[p.rival]
            if rival.status is not CarStatus.RUNNING or _in_pit_lane(rival):
                continue  # retired, finished or pitted before the pass: it never happened
            if c.distance_m > rival.distance_m and not _in_pit_lane(c):
                emit(
                    "OVERTAKE",
                    c.driver_id,
                    {
                        "passed": p.rival,
                        "lap": p.lap,
                        "margin_s": p.margin_s,
                        "probability": p.probability,
                    },
                )
                setup = state.setup
                if penalty_draw(  # W4: the move caused a collision / forced the rival off
                    setup,
                    state.rng.seed,
                    "collision",
                    c.driver_id,
                    p.lap,
                    p.rival,
                    rate=setup.race_control.penalty_collision_per_pass
                    * (
                        setup.race_control.intent_attack_collision
                        if c.lap_intent is RacecraftIntent.ATTACK
                        else 1.0
                    ),
                ):
                    c = _penalise(c, 10.0, "collision", p.lap, emit)
                    out[i] = c
            else:
                keep.append(p)
        if len(keep) != len(c.pending_overtakes):
            out[i] = c.model_copy(update={"pending_overtakes": tuple(keep)})
    return tuple(out)


def _penalise(car: CarState, seconds: float, reason: str, lap: int, emit: _Emit) -> CarState:
    """A time penalty (W4): added to the car's race time at the finish."""
    emit("TIME_PENALTY", car.driver_id, {"seconds": seconds, "reason": reason, "lap": lap})
    return car.model_copy(update={"penalty_s": car.penalty_s + seconds})


def _with_incident(state: WorldState, car: CarState) -> CarState:
    """Draw this lap's incident at lap start. A mistake's time loss goes into the lap time
    before traffic is resolved; behind the safety car there are no racing mistakes."""
    incident = lap_incident(
        state.setup, state.rng.seed, car, car.laps_completed + 1, state.weather.wetness
    )
    if incident is None or (
        incident.kind is IncidentKind.MISTAKE and state.race_control.phase is RacePhase.SC
    ):
        return car
    extra = incident.time_loss_s if incident.kind is IncidentKind.MISTAKE else 0.0
    return car.model_copy(
        update={
            "lap_target_s": car.lap_target_s + extra,
            "incident_kind": incident.kind.value,
            "incident_at_fraction": incident.lap_fraction,
        }
    )


# --- line crossing ----------------------------------------------------------------------------


def _complete_lap(
    state: WorldState, car: CarState, crossing_time: float, flag_s: float | None, emit: _Emit
) -> CarState:
    """The car crosses the line: lap bookkeeping, then finish, or pit and/or start a new lap."""
    setup = state.setup
    driver = setup.driver(car.driver_id).profile.archetype
    neutralised = state.race_control.phase is not RacePhase.GREEN
    stop = pending_stop(car)  # decided for the lap just driven, before the counter moves
    if stop is None and neutralised:
        stop = _cheap_stop(state, car)
    if car.controller is StrategyController.BASELINE:
        # the weather rule overrides a planned stop that would fit the wrong tyre type
        wet_stop = weather_stop(
            setup, car, state.weather.wetness, setup.track.total_laps - car.laps_completed - 1
        )
        if wet_stop is not None:
            stop = wet_stop
        elif stop is not None and not _fits_conditions(setup, car, stop, state.weather.wetness):
            stop = None
    laps = car.laps_completed + 1
    lap_time = crossing_time - car.lap_started_at_s
    # The lap just driven wears the tyres at the rate of the mode it was driven in.
    wear = car.tyre_wear_s + wear_per_lap_s(
        setup.lap_model,
        driver,
        compound=car.compound,
        pace_mode=car.lap_pace_mode,
        tyre_age_laps=car.tyre_age_laps,
    )
    # W1: the tyre life it used (less when the lap ends behind the SC / under VSC)
    life_used = car.tyre_life_used + life_use_per_lap(
        setup.lap_model, driver, compound=car.compound, pace_mode=car.lap_pace_mode
    ) * (setup.lap_model.neutralised_life_factor if neutralised else 1.0)
    emit(
        "LAP_COMPLETED",
        car.driver_id,
        {"lap": laps, "lap_time_s": lap_time, "crossing_time_s": crossing_time},
    )
    if car.fuel_laps is not None:  # H1b: one lap of fuel burnt; an empty tank must stop
        fuel_left = car.fuel_laps - 1.0
        if stop is None and fuel_left < 1.0 and laps < setup.track.total_laps:
            stop = car.compound
    common = {
        "fuel_laps": None if car.fuel_laps is None else car.fuel_laps - 1.0,
        "laps_completed": laps,
        "tyre_age_laps": car.tyre_age_laps + 1,
        "tyre_wear_s": wear,
        "tyre_life_used": life_used,
        "last_lap_time_s": lap_time,
        "best_lap_time_s": (
            lap_time if car.best_lap_time_s is None else min(car.best_lap_time_s, lap_time)
        ),
        "last_lap_traffic_loss_s": car.traffic_loss_s,
        "lap_started_at_s": crossing_time,
        "distance_m": 0.0,
        "lap_elapsed_s": 0.0,
        "pit_hold_s": 0.0,
        "incident_kind": None,
        "incident_at_fraction": None,
        "pending_overtakes": (),
    }

    rc_params = setup.race_control
    if laps == 1 and penalty_draw(
        setup, state.rng.seed, "false_start", car.driver_id, rate=rc_params.penalty_false_start
    ):
        car = _penalise(car, 5.0, "false start", laps, emit)
    if not neutralised and penalty_draw(
        setup, state.rng.seed, "lap", car.driver_id, laps, rate=rc_params.penalty_per_lap
    ):
        car = _penalise(car, 5.0, "track limits / race conduct", laps, emit)
    if laps == setup.track.total_laps or (flag_s is not None and crossing_time >= flag_s):
        finish = crossing_time + car.penalty_s  # W4: time penalties count in the result
        emit(
            "CAR_FINISHED",
            car.driver_id,
            {"finish_time_s": finish, "laps": laps, "penalty_s": car.penalty_s},
        )
        need = dry_compounds_required(setup, car.compounds_used)
        if len(set(car.compounds_used)) < need:
            emit(
                "TYRE_RULE_BREACHED",
                car.driver_id,
                {"compounds_used": len(set(car.compounds_used)), "required": need},
            )
        return car.model_copy(
            update={
                **common,
                "status": CarStatus.FINISHED,
                "finish_time_s": finish,
                "pit_request": None,
                "planned_stops": (),
                "traffic_loss_s": 0.0,
                "carried_loss_s": 0.0,
            }
        )

    car = car.model_copy(update=common)
    hold = 0.0
    if stop is not None:
        car, hold = _pit(state, car, stop, emit, neutralised=neutralised)

    if car.controller is StrategyController.BASELINE and not car.pace_by_user:
        mode = baseline_pace_mode(setup, car)  # W1: manage or push to make the plan work
        if mode is not car.pace_mode:
            car = car.model_copy(update={"pace_mode": mode})
    lap_mode = car.pace_mode  # the driver's current intent is locked in for the new lap
    # Time lost defending or yielding on the previous lap is paid on this one.
    target = (
        hold
        + car.carried_loss_s
        + _sample_next_lap(
            setup,
            state.rng.seed,
            car.driver_id,
            compound=car.compound,
            pace_mode=lap_mode,
            laps_completed=laps,
            tyre_wear_s=car.tyre_wear_s,
            wetness=state.weather.wetness,
            tyre_life_used=car.tyre_life_used,
            fresh_lap=car.tyre_age_laps == 1 and car.pit_stops > 0,
            fuel_laps=car.fuel_laps,
        )
    )
    car = car.model_copy(
        update={
            "lap_target_s": target,
            "pit_hold_s": hold,
            "lap_pace_mode": lap_mode,
            "lap_intent": car.intent,
            "traffic_loss_s": car.carried_loss_s,
            "carried_loss_s": 0.0,
        }
    )
    return _with_incident(state, car)


def _fits_conditions(setup: RaceSetup, car: CarState, stop: Compound, wetness: float) -> bool:
    """A planned dry stop still makes sense (the track isn't so wet that slicks are wrong)."""
    if car.controller is not StrategyController.BASELINE:
        return True
    best = best_tyre(setup.weather, wetness)
    return is_slick(stop) == is_slick(best)


def _cheap_stop(state: WorldState, car: CarState) -> Compound | None:
    """Baseline strategy under SC/VSC: bring the next planned stop forward if it's close; on
    old wet-weather tyres, maybe take a new set of the same type (W6, D-046)."""
    setup = state.setup
    if car.controller is not StrategyController.BASELINE:
        return None
    weather = setup.weather
    if (
        car.compound in WET_TYRES
        and car.tyre_age_laps >= weather.wet_refresh_min_age_laps
        and setup.track.total_laps - car.laps_completed > 3
    ):
        episode = state.race_control.sc_count + state.race_control.vsc_count
        draw = uniform(state.rng.seed, "wet_refresh", car.driver_id, episode)
        if draw < weather.wet_refresh_share:
            return car.compound
    if not car.planned_stops:
        return None
    nxt = car.planned_stops[0]
    window = setup.race_control.neutralised_pit_window_laps
    if nxt.lap - car.laps_completed <= window:
        return nxt.compound
    return None


def _pit(
    state: WorldState, car: CarState, compound: Compound, emit: _Emit, *, neutralised: bool = False
) -> tuple[CarState, float]:
    """Execute a stop at the end of lap ``car.laps_completed``; returns the car and its hold."""
    setup = state.setup
    if setup.rules.refuelling and car.fuel_laps is not None:
        return _refuel_stop(state, car, compound, emit, neutralised=neutralised)
    standstill = standstill_s(setup, state.rng.seed, car.driver_id, car.laps_completed)
    hold = setup.pit.lane_loss_s + standstill
    emit(
        "PIT_STOP",
        car.driver_id,
        {
            "lap": car.laps_completed,
            "from_compound": car.compound.value,
            "to_compound": compound.value,
            "tyre_age_laps": car.tyre_age_laps,
            "standstill_s": standstill,
            "pit_loss_s": hold,
            "stop_number": car.pit_stops + 1,
            "under": state.race_control.phase.value if neutralised else "GREEN",
        },
    )
    used = car.compounds_used if compound in car.compounds_used else (*car.compounds_used, compound)
    planned = car.planned_stops
    if planned and planned[0].lap <= car.laps_completed:
        planned = planned[1:]
    car = car.model_copy(
        update={
            "compound": compound,
            "tyre_age_laps": 0,
            "tyre_wear_s": 0.0,
            "tyre_life_used": 0.0,
            "pit_stops": car.pit_stops + 1,
            "compounds_used": used,
            "pit_request": None,
            "planned_stops": planned,
        }
    )
    if car.controller is StrategyController.BASELINE:
        plan = plan_for_car(setup, car)
        if plan != car.planned_stops:
            emit(
                "STRATEGY_PLANNED",
                car.driver_id,
                {"plan": plan_text(plan), "policy_id": POLICY_ID, "actor_id": POLICY_ID},
            )
        car = car.model_copy(update={"planned_stops": plan})
    return car, hold


def _refuel_stop(
    state: WorldState, car: CarState, compound: Compound, emit: _Emit, *, neutralised: bool
) -> tuple[CarState, float]:
    """A stop in the refuelling era (H1b): tyres (unless 2005's rule keeps the set: a dry stop
    is then fuel only) and the fuel for the next stint, chosen now by the plan (a USER/AGENT
    car takes the baseline's fuel too). Stationary time = the longer of the tyre change and the
    refuelling (``refuel_s_per_lap`` per lap of fuel put in)."""
    from f1ar.policy.refuel_strategy import best_plan_refuel

    setup = state.setup
    keep_set = (
        not setup.rules.tyre_changes_allowed and compound is car.compound and is_slick(compound)
    )
    if not keep_set:
        used = (
            car.compounds_used
            if compound in car.compounds_used
            else (*car.compounds_used, compound)
        )
        car = car.model_copy(
            update={
                "compound": compound,
                "tyre_age_laps": 0,
                "tyre_wear_s": 0.0,
                "tyre_life_used": 0.0,
                "compounds_used": used,
            }
        )
    left = max(0.0, car.fuel_laps or 0.0)
    seen_setup, seen = team_view(setup, car)  # H3: the team plans with what it knows
    plan = best_plan_refuel(
        seen_setup,
        setup.driver(car.driver_id).profile.archetype,
        laps_completed=car.laps_completed,
        compound=car.compound,
        compounds_used=car.compounds_used,
        fuel_laps=None,
        free_first=True,
        at_stop=True,
        tyre_age_laps=car.tyre_age_laps,
        tyre_wear_s=seen.tyre_wear_s,
        life_used=seen.tyre_life_used,
    )
    remaining = setup.track.total_laps - car.laps_completed
    stint = remaining if plan is None else plan.first_stint_laps
    added = max(0.0, stint - left)
    refuel = added * setup.pit.refuel_s_per_lap
    tyres = (
        0.0 if keep_set else standstill_s(setup, state.rng.seed, car.driver_id, car.laps_completed)
    )
    standstill = max(tyres, refuel)
    hold = setup.pit.lane_loss_s + standstill
    emit(
        "PIT_STOP",
        car.driver_id,
        {
            "lap": car.laps_completed,
            "from_compound": car.compound.value if keep_set else compound.value,
            "to_compound": compound.value,
            "tyre_age_laps": car.tyre_age_laps,
            "standstill_s": standstill,
            "pit_loss_s": hold,
            "stop_number": car.pit_stops + 1,
            "under": state.race_control.phase.value if neutralised else "GREEN",
            "fuel_laps_added": round(added, 1),
            "refuel_s": round(refuel, 2),
        },
    )
    stops = () if plan is None else plan.stops
    if car.controller is StrategyController.BASELINE and stops != car.planned_stops:
        emit(
            "STRATEGY_PLANNED",
            car.driver_id,
            {"plan": plan_text(stops), "policy_id": POLICY_ID, "actor_id": POLICY_ID},
        )
    car = car.model_copy(
        update={
            "fuel_laps": left + added,
            "pit_stops": car.pit_stops + 1,
            "pit_request": None,
            "planned_stops": stops if car.controller is StrategyController.BASELINE else (),
        }
    )
    return car, hold


def standings(state: WorldState) -> tuple[str, ...]:
    """Driver ids in race order.

    More completed laps first, then further into the lap. Cars parked at the line during a
    pit hold are ordered by time since they crossed it. Among equal laps, a car that finished
    (it's on the line) is behind a car already out on its next lap, which must have crossed
    before the flag. Finishers with equal laps are ordered by finish time. Ties (e.g.
    everyone on the line at the start) are broken by grid position. Retired cars come last,
    more laps first, then later retirements first.
    """

    def key(car: CarState) -> tuple[int, int, float, float, float, int]:
        grid = start_slot(state.setup.driver(car.driver_id).grid_position)
        if car.status is CarStatus.RETIRED:  # classified by distance completed, then by time
            return (1, -car.laps_completed, -(car.retired_at_s or 0.0), 0.0, 0.0, grid)
        return (
            0,
            -car.laps_completed,
            -car.distance_m,
            -car.lap_elapsed_s,
            car.finish_time_s or 0.0,
            grid,
        )

    return tuple(c.driver_id for c in sorted(state.cars, key=key))
