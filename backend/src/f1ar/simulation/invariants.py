"""World-state invariants (plan §18.2), checkable on every tick.

``state_violations`` checks one state on its own; ``step_violations`` checks one transition
(previous state -> state after one ``engine.step``, with the tick's events). Both return a list
of human-readable violations (empty = fine) instead of raising, so tests can show all problems
at once and the runner can decide what to do in debug mode.
"""

import math

from f1ar.domain.race_state import (
    CarStatus,
    RaceEvent,
    RacePhase,
    StrategyController,
    WorldState,
    WorldStatus,
)


def state_violations(state: WorldState) -> list[str]:
    setup = state.setup
    length = setup.track.length_m
    total = setup.track.total_laps
    out: list[str] = []

    def bad(msg: str) -> None:
        out.append(f"tick {state.sim_tick}: {msg}")

    if not math.isclose(state.sim_clock_s, state.sim_tick * setup.tick_seconds, abs_tol=1e-9):
        bad(f"sim clock {state.sim_clock_s} != tick x tick_seconds")
    if state.race_clock_s < 0 or state.race_clock_s > state.sim_clock_s + 1e-6:
        bad(f"race clock {state.race_clock_s} outside [0, sim clock]")
    ids = [c.driver_id for c in state.cars]
    if len(set(ids)) != len(ids):
        bad("duplicate driver ids")

    for c in state.cars:
        who = c.driver_id
        if not 0.0 <= c.distance_m < length:
            bad(f"{who} distance {c.distance_m} outside [0, {length})")
        if not 0 <= c.laps_completed <= total:
            bad(f"{who} laps {c.laps_completed} outside [0, {total}]")
        if c.tyre_age_laps > c.laps_completed or c.tyre_wear_s < 0:
            bad(f"{who} tyre age {c.tyre_age_laps} / wear {c.tyre_wear_s} impossible")
        if c.compound not in c.compounds_used:
            bad(f"{who} on {c.compound} but compounds used {c.compounds_used}")
        if c.pit_stops < len(set(c.compounds_used)) - 1:
            bad(f"{who} used {len(set(c.compounds_used))} compounds with {c.pit_stops} stops")
        if not c.lap_target_s > c.pit_hold_s >= 0.0:
            bad(f"{who} lap target {c.lap_target_s} not above pit hold {c.pit_hold_s}")
        if c.status is CarStatus.FINISHED:
            if c.finish_time_s is None:
                bad(f"{who} finished without a finish time")
            if c.laps_completed != total and state.chequered_flag_s is None:
                bad(f"{who} finished {c.laps_completed} laps before any chequered flag")
        elif c.status is CarStatus.RETIRED:
            if c.retired_reason is None or c.retired_at_s is None:
                bad(f"{who} retired without reason/time")
        else:
            if c.finish_time_s is not None or c.retired_reason is not None:
                bad(f"{who} running but has a finish time or retirement")
            if c.controller is StrategyController.BASELINE and any(
                s.lap <= c.laps_completed for s in c.planned_stops
            ):
                bad(f"{who} has a planned stop in the past: {c.planned_stops}")

    rc = state.race_control
    if rc.phase is RacePhase.SC:
        if rc.sc_distance_m is None or not 0.0 <= rc.sc_distance_m < length:
            bad(f"safety car out but at {rc.sc_distance_m}")
    elif rc.sc_distance_m is not None:
        bad("safety car position set outside an SC phase")
    if (rc.phase is RacePhase.VSC) != (rc.vsc_until_s is not None):
        bad(f"VSC end time {rc.vsc_until_s} inconsistent with phase {rc.phase}")

    if state.chequered_flag_s is not None and not any(
        c.status is CarStatus.FINISHED and c.laps_completed == total for c in state.cars
    ):
        bad("chequered flag out but nobody completed the race distance")
    running = any(c.status is CarStatus.RUNNING for c in state.cars)
    if (state.world_status is WorldStatus.COMPLETED) == running:
        bad(f"world {state.world_status} with running cars = {running}")
    return out


def step_violations(
    prev: WorldState, state: WorldState, events: tuple[RaceEvent, ...]
) -> list[str]:
    """Violations of one ``engine.step`` transition, plus ``state_violations(state)``."""
    out = state_violations(state)

    def bad(msg: str) -> None:
        out.append(f"tick {state.sim_tick}: {msg}")

    if state.sim_tick != prev.sim_tick + 1:
        bad(f"tick went {prev.sim_tick} -> {state.sim_tick}")
    if state.race_clock_s < prev.race_clock_s:
        bad("race clock went backwards")
    if [e.seq for e in events] != list(range(prev.event_seq, state.event_seq)):
        bad(f"event seqs {[e.seq for e in events]} not contiguous from {prev.event_seq}")
    if any(e.tick != state.sim_tick for e in events):
        bad("event stamped with another tick")
    if prev.chequered_flag_s is not None and state.chequered_flag_s != prev.chequered_flag_s:
        bad("chequered flag time changed")
    for before, after in zip(prev.cars, state.cars, strict=True):
        who = after.driver_id
        if after.laps_completed < before.laps_completed:
            bad(f"{who} lost a lap")
        if before.status is not CarStatus.RUNNING and after != before:
            bad(f"{who} changed after {before.status}")
        if after.pit_stops < before.pit_stops:
            bad(f"{who} lost a pit stop")
    return out
