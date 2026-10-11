"""Lightweight deterministic race engine: pure ``step(state) -> (state, events)``.

Plan option C in miniature (§7.2): no physics. Cars move along the track at a speed implied
by their pre-sampled lap time. Each ``step`` advances exactly one bounded tick of
``setup.tick_seconds``, so pausing between ticks is always possible.

Not modelled yet (later steps): pit stops, strategy decisions, traffic/dirty air,
overtaking cost, safety cars, weather, DNFs. Cars pass through each other freely.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass

from f1ar.domain.race_state import (
    WORLD_SCHEMA_VERSION,
    CarState,
    CarStatus,
    Compound,
    PaceMode,
    RaceEvent,
    RaceSetup,
    WorldState,
    WorldStatus,
)
from f1ar.simulation.lap_model import sample_lap_time_s
from f1ar.simulation.rng_streams import RngSpec

ENGINE_VERSION = "lightweight_engine_v0.1.0"


class EngineStateError(RuntimeError):
    """An operation was requested in a world status that does not allow it."""


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
    tyre_age_laps: int,
) -> float:
    return sample_lap_time_s(
        setup.lap_model,
        setup.driver(driver_id).profile.archetype,
        seed=seed,
        driver_id=driver_id,
        lap_number=laps_completed + 1,
        compound=compound,
        tyre_age_laps=tyre_age_laps,
        pace_mode=pace_mode,
        laps_remaining=setup.track.total_laps - laps_completed,
    )


def create_world(
    setup: RaceSetup, *, seed: int, world_id: str, timeline_id: str = "timeline_main"
) -> WorldState:
    """New world in CREATED status: all cars on the line at lap 0, first lap pre-sampled."""
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
            tyre_age_laps=0,
        )
        cars.append(
            CarState(
                driver_id=entry.driver_id,
                status=CarStatus.RUNNING,
                laps_completed=0,
                distance_m=0.0,
                lap_elapsed_s=0.0,
                lap_target_s=target,
                compound=entry.start_compound,
                tyre_age_laps=0,
                pace_mode=pace_mode,
                last_lap_time_s=None,
                finish_time_s=None,
            )
        )
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

    cars = tuple(_advance_car(state, car, emit) for car in state.cars)

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
            "world_status": status,
            "event_seq": state.event_seq + len(events),
        }
    )
    return StepResult(new_state, tuple(events))


def _advance_car(state: WorldState, car: CarState, emit: _Emit) -> CarState:
    if car.status is CarStatus.FINISHED:
        return car

    setup = state.setup
    length = setup.track.length_m
    dt = setup.tick_seconds
    tick_start_clock = state.sim_clock_s

    distance = car.distance_m
    elapsed = car.lap_elapsed_s
    target = car.lap_target_s
    laps = car.laps_completed
    tyre_age = car.tyre_age_laps
    last_lap = car.last_lap_time_s
    remaining = dt

    while remaining > 0.0:
        speed = length / target
        time_to_line = (length - distance) / speed
        if time_to_line > remaining:
            distance = min(distance + speed * remaining, math.nextafter(length, 0.0))
            elapsed += remaining
            break

        # Crossed the line inside this tick: interpolate the exact crossing time.
        remaining -= time_to_line
        crossing_time = tick_start_clock + (dt - remaining)
        last_lap = elapsed + time_to_line
        laps += 1
        tyre_age += 1
        emit(
            "LAP_COMPLETED",
            car.driver_id,
            {"lap": laps, "lap_time_s": last_lap, "crossing_time_s": crossing_time},
        )
        if laps == setup.track.total_laps:
            emit("CAR_FINISHED", car.driver_id, {"finish_time_s": crossing_time, "laps": laps})
            return car.model_copy(
                update={
                    "status": CarStatus.FINISHED,
                    "laps_completed": laps,
                    "distance_m": 0.0,
                    "lap_elapsed_s": 0.0,
                    "tyre_age_laps": tyre_age,
                    "last_lap_time_s": last_lap,
                    "finish_time_s": crossing_time,
                }
            )
        distance = 0.0
        elapsed = 0.0
        target = _sample_next_lap(
            setup,
            state.rng.seed,
            car.driver_id,
            compound=car.compound,
            pace_mode=car.pace_mode,
            laps_completed=laps,
            tyre_age_laps=tyre_age,
        )

    return car.model_copy(
        update={
            "laps_completed": laps,
            "distance_m": distance,
            "lap_elapsed_s": elapsed,
            "lap_target_s": target,
            "tyre_age_laps": tyre_age,
            "last_lap_time_s": last_lap,
        }
    )


def standings(state: WorldState) -> tuple[str, ...]:
    """Driver ids in race order: finishers by finish time, then by race progress.

    Ties (e.g. everyone on the line at the start) are broken by grid position.
    """
    length = state.setup.track.length_m

    def key(car: CarState) -> tuple[int, float, int]:
        grid = state.setup.driver(car.driver_id).grid_position
        if car.status is CarStatus.FINISHED:
            assert car.finish_time_s is not None
            return (0, car.finish_time_s, grid)
        return (1, -(car.laps_completed * length + car.distance_m), grid)

    return tuple(c.driver_id for c in sorted(state.cars, key=key))
