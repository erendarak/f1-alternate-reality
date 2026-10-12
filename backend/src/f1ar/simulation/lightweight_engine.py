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

Not modelled yet (later M2 slices): safety cars, DNFs, start performance.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import assert_never

from f1ar.domain.race_state import (
    WORLD_SCHEMA_VERSION,
    CancelPitStop,
    CarState,
    CarStatus,
    Compound,
    DriverAction,
    PaceMode,
    PlannedStop,
    RaceEvent,
    RaceSetup,
    RequestPitStop,
    SetPaceMode,
    SetStrategyController,
    StrategyController,
    WorldState,
    WorldStatus,
)
from f1ar.policy.baseline_strategy import POLICY_ID, plan_for_car
from f1ar.simulation.lap_model import sample_lap_time_s, wear_per_lap_s
from f1ar.simulation.rng_streams import RngSpec, uniform
from f1ar.simulation.traffic import Ahead, resolve

ENGINE_VERSION = "lightweight_engine_v0.5.0"
"""v0.5.0: traffic: dirty air, held up, overtakes, blue flags; lap 1 resolved in grid order.
v0.4.0: pit stops + tyre choice, baseline strategy, chequered-flag rule (lapped finishers).
v0.3.0: per-car performance offset (``DriverEntry.car_pace_offset_s``).
v0.2.0: tyre wear accumulated per lap at the mode driven (v0.1.0 scaled it retroactively)."""


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
) -> float:
    entry = setup.driver(driver_id)
    return sample_lap_time_s(
        setup.lap_model,
        entry.profile.archetype,
        seed=seed,
        driver_id=driver_id,
        lap_number=laps_completed + 1,
        compound=compound,
        tyre_wear_s=tyre_wear_s,
        pace_mode=pace_mode,
        laps_remaining=setup.track.total_laps - laps_completed,
        car_offset_s=entry.car_pace_offset_s,
    )


def standstill_s(setup: RaceSetup, seed: int, driver_id: str, lap: int) -> float:
    """Stationary time of a stop at the end of ``lap``: log-logistic, keyed by (driver, lap).

    Inverse CDF of the fisk distribution: ``median * (u / (1 - u)) ** (1 / shape)``.
    """
    u = uniform(seed, "pit_standstill", driver_id, lap)
    p = setup.pit
    draw: float = p.standstill_median_s * (u / (1.0 - u)) ** (1.0 / p.standstill_shape)
    return min(draw, p.standstill_max_s)


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


def create_world(
    setup: RaceSetup, *, seed: int, world_id: str, timeline_id: str = "timeline_main"
) -> WorldState:
    """New world in CREATED status: all cars on the line at lap 0, first lap pre-sampled,
    every car under the baseline strategy with an initial plan."""
    cars = []
    for entry in setup.drivers:
        pace_mode = PaceMode.BALANCED
        target = _sample_next_lap(
            setup,
            seed,
            entry.driver_id,
            compound=entry.start_compound,
            pace_mode=pace_mode,
            laps_completed=0,
            tyre_wear_s=0.0,
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
            last_lap_time_s=None,
            finish_time_s=None,
        )
        cars.append(car.model_copy(update={"planned_stops": plan_for_car(setup, car)}))
    return WorldState(
        schema_version=WORLD_SCHEMA_VERSION,
        world_id=world_id,
        timeline_id=timeline_id,
        parent_timeline_id=None,
        world_status=WorldStatus.CREATED,
        mode="FREE_SIMULATION",
        engine_version=ENGINE_VERSION,
        rng=RngSpec(seed=seed),
        setup=setup,
        sim_tick=0,
        sim_clock_s=0.0,
        cars=tuple(cars),
        chequered_flag_s=None,
        event_seq=0,
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
            new = _cancel_pit(car, emit)
        case SetStrategyController():
            new = _set_controller(state, car, action.controller, emit)
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


def _take_over(car: CarState, emit: _ActionEmit) -> CarState:
    if car.controller is StrategyController.USER:
        return car
    emit("CONTROLLER_CHANGED", {"from": car.controller.value, "to": StrategyController.USER.value})
    # A stop the baseline planned for this lap becomes an explicit request, so taking over
    # never silently changes what the car was about to do.
    request = pending_stop(car)
    return car.model_copy(
        update={
            "controller": StrategyController.USER,
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
    new = _take_over(car, emit)
    if new.pit_request is action.compound:
        return new
    emit(
        "PIT_REQUESTED",
        {"compound": action.compound.value, "in_lap": new.laps_completed + 1},
    )
    return new.model_copy(update={"pit_request": action.compound})


def _cancel_pit(car: CarState, emit: _ActionEmit) -> CarState:
    if pending_stop(car) is None:
        return car  # nothing to cancel; a BASELINE car keeps its plan
    new = _take_over(car, emit)
    assert new.pit_request is not None
    emit("PIT_CANCELLED", {"compound": new.pit_request.value, "in_lap": new.laps_completed + 1})
    return new.model_copy(update={"pit_request": None})


def _set_controller(
    state: WorldState, car: CarState, controller: StrategyController, emit: _ActionEmit
) -> CarState:
    if car.controller is controller:
        return car
    if controller is StrategyController.USER:
        return _take_over(car, emit)
    emit("CONTROLLER_CHANGED", {"from": car.controller.value, "to": controller.value})
    # The baseline decides from here on; a pending user request is replaced by its plan.
    new = car.model_copy(update={"controller": controller, "pit_request": None})
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

    flag_s = state.chequered_flag_s
    new_flag = None if flag_s is not None else _flag_time_in_tick(state)
    if new_flag is not None:
        flag_s = new_flag

    cars = _advance_all(state, flag_s, emit)
    if new_flag is not None:
        winner = min(
            (c for c in cars if c.finish_time_s is not None), key=lambda c: c.finish_time_s or 0.0
        )
        emit("CHEQUERED_FLAG", winner.driver_id, {"flag_time_s": new_flag})

    status: WorldStatus = state.world_status
    if all(c.status is CarStatus.FINISHED for c in cars):
        status = WorldStatus.COMPLETED
        emit("RACE_COMPLETED", None, {"tick": tick})

    new_state = state.model_copy(
        update={
            "sim_tick": tick,
            # Derived from the tick count, not accumulated, so the clock never drifts.
            "sim_clock_s": tick * setup.tick_seconds,
            "cars": cars,
            "chequered_flag_s": flag_s,
            "world_status": status,
            "event_seq": state.event_seq + len(events),
        }
    )
    return StepResult(new_state, tuple(events))


def _flag_time_in_tick(state: WorldState) -> float | None:
    """Earliest time in this tick at which a car completes the full race distance, if any.

    Known before any car advances, so a lapped car that crosses the line *after* the leader
    in the same tick is flagged, and one that crosses before is not.
    """
    final_lap = state.setup.track.total_laps - 1
    times = [
        state.sim_clock_s + (c.lap_target_s - c.lap_elapsed_s)
        for c in state.cars
        if c.status is CarStatus.RUNNING
        and c.laps_completed == final_lap
        and c.lap_target_s - c.lap_elapsed_s <= state.setup.tick_seconds
    ]
    return min(times, default=None)


def _distance(length: float, elapsed: float, target: float, hold: float) -> float:
    """Position in the lap: parked at the line during the pit hold, then constant speed."""
    if elapsed <= hold:
        return 0.0
    return min(length * (elapsed - hold) / (target - hold), math.nextafter(length, 0.0))


def _advance_all(state: WorldState, flag_s: float | None, emit: _Emit) -> tuple[CarState, ...]:
    """Move every car one tick, processing line crossings and pit exits in time order."""
    setup = state.setup
    t0 = state.sim_clock_s
    t1 = t0 + setup.tick_seconds
    cars = list(state.cars)
    lap_start = [t0 - c.lap_elapsed_s for c in cars]
    if state.sim_tick == 0:
        _resolve_start(state, cars, emit)

    # (time, car index, kind); each car has at most one such moment per tick.
    due: list[tuple[float, int, str]] = []
    for i, c in enumerate(cars):
        if c.status is not CarStatus.RUNNING:
            continue
        if c.lap_elapsed_s < c.pit_hold_s:
            t = lap_start[i] + c.pit_hold_s
            if t <= t1:
                due.append((t, i, "rejoin"))
        elif lap_start[i] + c.lap_target_s <= t1:
            due.append((lap_start[i] + c.lap_target_s, i, "line"))

    joined: set[int] = set()
    for t, i, kind in sorted(due):
        if kind == "line":
            cars[i] = _complete_lap(state, cars[i], t, flag_s, emit)
            if cars[i].status is CarStatus.FINISHED:
                continue
            lap_start[i] = t
            joined.add(i)
            if cars[i].pit_hold_s > 0.0 and t + cars[i].pit_hold_s > t1:
                continue  # in the pit lane; it joins the track at the end of its hold
            t += cars[i].pit_hold_s
        _join_track(state, i, t, cars, lap_start, emit)

    for i, c in enumerate(cars):
        if c.status is not CarStatus.RUNNING:
            continue
        if i in joined:
            elapsed = t1 - lap_start[i]
        else:
            elapsed = c.lap_elapsed_s + setup.tick_seconds
            if c.lap_elapsed_s < c.pit_hold_s <= elapsed + 1e-9:
                elapsed = max(elapsed, c.pit_hold_s)  # the pit exit is handled exactly once
        cars[i] = c.model_copy(
            update={
                "lap_elapsed_s": elapsed,
                "distance_m": _distance(
                    setup.track.length_m, elapsed, c.lap_target_s, c.pit_hold_s
                ),
            }
        )
    return tuple(cars)


def _resolve_start(state: WorldState, cars: list[CarState], emit: _Emit) -> None:
    """Lap 1: everyone leaves the line at t=0; the road order is the grid order."""
    grid = sorted(
        range(len(cars)), key=lambda i: state.setup.driver(cars[i].driver_id).grid_position
    )
    for k, i in enumerate(grid):
        ahead = [Ahead(cars[j], 0.0) for j in reversed(grid[:k])]
        _apply_traffic(state, i, 0.0, 0.0, ahead, cars, emit)


def _join_track(
    state: WorldState,
    i: int,
    t: float,
    cars: list[CarState],
    lap_start: list[float],
    emit: _Emit,
) -> None:
    """Car ``i`` (at the line) joins the track at sim time ``t``: resolve the cars ahead."""
    ahead: list[tuple[float, int]] = []
    for j, other in enumerate(cars):
        if j == i or other.status is not CarStatus.RUNNING:
            continue
        driving = t - lap_start[j] - other.pit_hold_s
        if driving <= 0.0:
            continue  # parked in the pit lane: not on the road ahead
        ahead.append((driving / (other.lap_target_s - other.pit_hold_s), j))
    ahead.sort()
    _apply_traffic(
        state, i, lap_start[i], t, [Ahead(cars[j], lap_start[j]) for _, j in ahead], cars, emit
    )


def _apply_traffic(
    state: WorldState,
    i: int,
    my_lap_start: float,
    t: float,
    ahead: list[Ahead],
    cars: list[CarState],
    emit: _Emit,
) -> None:
    me = cars[i]
    out = resolve(state.setup, state.rng.seed, me, my_lap_start, t, ahead)
    cars[i] = me.model_copy(
        update={
            "lap_target_s": out.lap_target_s,
            "traffic_loss_s": me.traffic_loss_s + out.lap_target_s - me.lap_target_s,
        }
    )
    for type_, payload in out.events:
        emit(type_, me.driver_id, payload)
    for j, other in enumerate(cars):
        loss = out.carried_losses.get(other.driver_id)
        if loss:
            cars[j] = other.model_copy(update={"carried_loss_s": other.carried_loss_s + loss})


def _complete_lap(
    state: WorldState, car: CarState, crossing_time: float, flag_s: float | None, emit: _Emit
) -> CarState:
    """The car crosses the line: lap bookkeeping, then finish, or pit and/or start a new lap."""
    setup = state.setup
    driver = setup.driver(car.driver_id).profile.archetype
    stop = pending_stop(car)  # decided for the lap just driven, before the counter moves
    laps = car.laps_completed + 1
    lap_time = car.lap_target_s
    # The lap just driven wears the tyres at the rate of the mode it was driven in.
    wear = car.tyre_wear_s + wear_per_lap_s(
        setup.lap_model, driver, compound=car.compound, pace_mode=car.lap_pace_mode
    )
    emit(
        "LAP_COMPLETED",
        car.driver_id,
        {"lap": laps, "lap_time_s": lap_time, "crossing_time_s": crossing_time},
    )
    common = {
        "laps_completed": laps,
        "tyre_age_laps": car.tyre_age_laps + 1,
        "tyre_wear_s": wear,
        "last_lap_time_s": lap_time,
        "last_lap_traffic_loss_s": car.traffic_loss_s,
        "distance_m": 0.0,
        "lap_elapsed_s": 0.0,
        "pit_hold_s": 0.0,
    }

    if laps == setup.track.total_laps or (flag_s is not None and crossing_time >= flag_s):
        emit("CAR_FINISHED", car.driver_id, {"finish_time_s": crossing_time, "laps": laps})
        need = setup.pit.min_distinct_dry_compounds
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
                "finish_time_s": crossing_time,
                "pit_request": None,
                "planned_stops": (),
                "traffic_loss_s": 0.0,
                "carried_loss_s": 0.0,
            }
        )

    car = car.model_copy(update=common)
    hold = 0.0
    if stop is not None:
        car, hold = _pit(state, car, stop, emit)

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
        )
    )
    return car.model_copy(
        update={
            "lap_target_s": target,
            "pit_hold_s": hold,
            "lap_pace_mode": lap_mode,
            "traffic_loss_s": car.carried_loss_s,
            "carried_loss_s": 0.0,
        }
    )


def _pit(
    state: WorldState, car: CarState, compound: Compound, emit: _Emit
) -> tuple[CarState, float]:
    """Execute a stop at the end of lap ``car.laps_completed``; returns the car and its hold."""
    setup = state.setup
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


def standings(state: WorldState) -> tuple[str, ...]:
    """Driver ids in race order.

    More completed laps first, then further into the lap. Cars parked at the line during a
    pit hold are ordered by time since they crossed it. Among equal laps, a car that finished
    (it's on the line) is behind a car already out on its next lap, which must have crossed
    before the flag. Finishers with equal laps are ordered by finish time. Ties (e.g.
    everyone on the line at the start) are broken by grid position.
    """

    def key(car: CarState) -> tuple[int, float, float, float, int]:
        grid = state.setup.driver(car.driver_id).grid_position
        return (
            -car.laps_completed,
            -car.distance_m,
            -car.lap_elapsed_s,
            car.finish_time_s or 0.0,
            grid,
        )

    return tuple(c.driver_id for c in sorted(state.cars, key=key))
