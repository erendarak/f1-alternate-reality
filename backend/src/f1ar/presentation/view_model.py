"""JSON view models for the debug viewer: canonical state in, drawable data out (pure).

This is a spectator ("god") view, not a policy observation: it may show things a driver
can't know. Gaps are approximate display values derived from track position.
"""

import math
from typing import Any

from f1ar.domain.driver_profiles import STAGE1_FIELDS
from f1ar.domain.provenance import SourceType
from f1ar.domain.race_state import CarStatus, RaceEvent, RacePhase, WorldState
from f1ar.history.reference import HistoricalRace
from f1ar.presentation.track_geometry import TrackGeometry
from f1ar.simulation.lightweight_engine import pending_stop, plan_text, standings

DEFAULT_COLOR = "9AA0A6"


def track_view(
    geometry: TrackGeometry,
    *,
    engine_length_m: float | None,
    corners: list[dict[str, Any]],
    rotation_deg: float,
    source: str,
) -> dict[str, Any]:
    return {
        "outline": [
            [round(x, 1), round(y, 1)] for x, y in zip(geometry.xs, geometry.ys, strict=True)
        ],
        "start_finish": [round(geometry.xs[0], 1), round(geometry.ys[0], 1)],
        "corners": corners,
        "rotation_deg": rotation_deg,
        "outline_length_m": round(geometry.length_m, 1),
        "engine_length_m": engine_length_m,
        "bounds": [round(v, 1) for v in geometry.bounds()],
        "source": source,
    }


def _progress_m(state: WorldState, driver_id: str) -> float:
    car = state.car(driver_id)
    return car.laps_completed * state.setup.track.length_m + car.distance_m


def state_view(
    state: WorldState,
    geometry: TrackGeometry,
    *,
    team_colors: dict[str, str],
    speed: int,
    recent_events: tuple[RaceEvent, ...] = (),
) -> dict[str, Any]:
    track = state.setup.track
    order = standings(state)
    # Display-only gap estimate: metres behind the leader at a nominal race pace.
    nominal_speed = track.length_m / state.setup.lap_model.base_lap_s
    leader = order[0]
    leader_car = state.car(leader)
    cars = []
    for pos, driver_id in enumerate(order, start=1):
        car = state.car(driver_id)
        entry = state.setup.driver(driver_id)
        fraction = car.distance_m / track.length_m
        x, y = geometry.point_at(fraction)
        gap: float | None
        laps_down: int | None
        if car.status is CarStatus.RETIRED:
            gap, laps_down = None, None
        elif car.status is CarStatus.FINISHED and leader_car.finish_time_s is not None:
            gap = (car.finish_time_s or 0.0) - leader_car.finish_time_s
            laps_down = leader_car.laps_completed - car.laps_completed
        else:
            behind_m = _progress_m(state, leader) - _progress_m(state, driver_id)
            gap = behind_m / nominal_speed
            laps_down = math.floor(behind_m / track.length_m)
        box = pending_stop(car)
        cars.append(
            {
                "position": pos,
                "driver_id": driver_id,
                "code": entry.code,
                "team_color": team_colors.get(driver_id, DEFAULT_COLOR),
                "x": round(x, 1),
                "y": round(y, 1),
                "heading_deg": round(geometry.heading_at(fraction), 1),
                "lap_fraction": round(fraction, 6),
                "lap": min(car.laps_completed + 1, track.total_laps),
                "laps_completed": car.laps_completed,
                "status": car.status.value,
                "compound": car.compound.value,
                "tyre_age_laps": car.tyre_age_laps,
                "pace_mode": car.pace_mode.value,
                "lap_pace_mode": car.lap_pace_mode.value,
                "last_lap_time_s": car.last_lap_time_s,
                "gap_to_leader_s": None if gap is None else round(gap, 3),
                "laps_down": laps_down,
                "pit_stops": car.pit_stops,
                "box": None if box is None else box.value,
                "in_pit": car.lap_elapsed_s < car.pit_hold_s,
                "controller": car.controller.value,
                "plan": plan_text(car.planned_stops),
                "last_lap_traffic_loss_s": round(car.last_lap_traffic_loss_s, 3),
                "retired_reason": car.retired_reason,
            }
        )
    rc = state.race_control
    safety_car = None
    if rc.phase is RacePhase.SC and rc.sc_distance_m is not None:
        sc_fraction = rc.sc_distance_m / track.length_m
        sx, sy = geometry.point_at(sc_fraction)
        safety_car = {
            "x": round(sx, 1),
            "y": round(sy, 1),
            "heading_deg": round(geometry.heading_at(sc_fraction), 1),
            "lap_fraction": round(sc_fraction, 6),
        }
    return {
        "mode": state.mode,
        "event_id": state.setup.event_id,
        "world_id": state.world_id,
        "timeline_id": state.timeline_id,
        "seed": state.rng.seed,
        "status": state.world_status.value,
        "tick": state.sim_tick,
        "sim_clock_s": state.sim_clock_s,
        "speed": speed,
        "leader_lap": min(leader_car.laps_completed + 1, track.total_laps),
        "total_laps": track.total_laps,
        "engine_version": state.engine_version,
        "race_control": {
            "phase": rc.phase.value,
            "safety_car": safety_car,
            "sc_laps_left": rc.sc_laps_left if rc.phase is RacePhase.SC else None,
            "vsc_left_s": (
                round(rc.vsc_until_s - state.sim_clock_s, 1)
                if rc.phase is RacePhase.VSC and rc.vsc_until_s is not None
                else None
            ),
            "cause": rc.cause,
            "sc_count": rc.sc_count,
            "vsc_count": rc.vsc_count,
        },
        "weather": weather_view(state),
        "cars": cars,
        "events": [e.model_dump(mode="json") for e in recent_events],
    }


def weather_view(state: WorldState) -> dict[str, Any]:
    """Conditions as a team would see them, plus the (noisy) forecast; never the true future."""
    from f1ar.simulation.weather import best_tyre, crossovers, forecast_rain_probability

    params = state.setup.weather
    return {
        "mode": params.mode.value,
        "raining": state.weather.raining,
        "wetness": round(state.weather.wetness, 3),
        "best_tyre": best_tyre(params, state.weather.wetness).value,
        "crossovers": [round(x, 3) for x in crossovers(params)],
        "forecast_rain_10min": round(
            forecast_rain_probability(
                params, state.rng.seed, state.sim_clock_s, state.weather.raining
            ),
            2,
        ),
        "source_type": params.source_type.value,
    }


# --- recorded race (HISTORICAL_REPLAY) ------------------------------------------------------

_REPLAY_STATUS = {
    "READY": "CREATED",
    "PLAYING": "RUNNING",
    "PAUSED": "PAUSED",
    "FINISHED": "COMPLETED",
}


def _laps_down(
    state: str, lead: tuple[int, float, str], lead_laps: int, laps_done: int, frac: float
) -> int | None:
    """Whole laps behind the leader by track progress (not by completed-lap count).

    Right after the leader crosses the line everyone else has one lap fewer *completed* but
    is not a lap *down*; only a full lap of progress counts.
    """
    if state == "RETIRED":
        return None
    if state == "FINISHED":
        return lead_laps - laps_done
    return math.floor((lead[0] + lead[1]) - (laps_done + frac))


def replay_view(
    race: HistoricalRace,
    geometry: TrackGeometry,
    *,
    replay_s: float,
    status: str,
    speed: int,
    recent: int = 12,
) -> dict[str, Any]:
    """Same JSON shape as ``state_view`` so one page can show both modes.

    ``pace_mode`` carries the lap type instead (``IN LAP`` / ``OUT LAP`` / blank), since the
    recording has no pace intent. Gaps are approximate: progress difference times the leader's
    current lap time.
    """
    progress = race.progress_at(replay_s)
    t_abs = race.race_start_s + replay_s

    def sort_key(code: str) -> tuple[int, float, int]:
        laps, frac, state = progress[code]
        car = race.car(code)
        if state == "FINISHED":
            return (0, car.laps[-1].end_s - car.laps_completed * 1e6, car.grid_position)
        if state == "RETIRED":
            return (2, -(laps + frac), car.grid_position)
        return (1, -(laps + frac), car.grid_position)  # ties (e.g. the start): grid order

    order = sorted(progress, key=sort_key)
    lead = progress[order[0]]
    lead_car = race.car(order[0])
    lead_lap = lead_car.laps[min(lead[0], lead_car.laps_completed - 1)]
    lead_lap_time = lead_lap.end_s - lead_lap.start_s
    cars = []
    for pos, code in enumerate(order, start=1):
        laps_done, frac, state = progress[code]
        car = race.car(code)
        current = car.laps[laps_done] if laps_done < car.laps_completed else car.laps[-1]
        last = car.laps[laps_done - 1] if laps_done >= 1 else None
        x, y = geometry.point_at(frac)
        lap_kind = (
            "IN LAP" if current.pit_in else "OUT LAP" if current.pit_out and current.lap > 1 else ""
        )
        if state == "FINISHED":
            lead_finish = lead_car.laps[-1].end_s
            gap: float | None = (
                car.laps[-1].end_s - lead_finish
                if car.laps_completed == lead_car.laps_completed
                else None
            )
        elif state == "RETIRED":
            gap = None
        else:
            gap = ((lead[0] + lead[1]) - (laps_done + frac)) * lead_lap_time
        cars.append(
            {
                "position": pos,
                "driver_id": code,
                "code": code,
                "team_color": car.team_color or DEFAULT_COLOR,
                "x": round(x, 1),
                "y": round(y, 1),
                "heading_deg": round(geometry.heading_at(frac), 1),
                "lap": min(laps_done + 1, race.total_laps),
                "laps_completed": laps_done,
                "status": state,
                "compound": current.compound or "UNKNOWN",
                "tyre_age_laps": current.tyre_life_laps
                if current.tyre_life_laps is not None
                else 0,
                "pace_mode": lap_kind,
                "lap_pace_mode": lap_kind,
                "last_lap_time_s": last.lap_time_s if last else None,
                "gap_to_leader_s": None if gap is None else round(gap, 3),
                "laps_down": _laps_down(state, lead, lead_car.laps_completed, laps_done, frac),
                # official classification can differ from the road (e.g. post-race DSQ)
                "result_note": "DSQ"
                if car.status == "Disqualified" and state == "FINISHED"
                else None,
            }
        )
    events = [e for e in race.events() if e.session_time_s <= t_abs][-recent:]
    return {
        "mode": race.mode,
        "world_id": race.event_id,
        "timeline_id": "recorded",
        "seed": None,
        "status": _REPLAY_STATUS[status],
        "tick": None,
        "sim_clock_s": replay_s,
        "duration_s": race.duration_s,
        "speed": speed,
        "leader_lap": min(lead[0] + 1, race.total_laps),
        "total_laps": race.total_laps,
        "engine_version": None,
        "cars": cars,
        "events": [
            {
                "seq": i,
                "tick": None,
                "type": e.type,
                "driver_id": e.code,
                "payload": {"lap": e.lap, "detail": e.detail},
            }
            for i, e in enumerate(events)
        ],
    }


# --- driver card (profile + provenance) -----------------------------------------------------

ENGINE_TRAITS: tuple[str, ...] = (
    "race_pace_skill",
    "consistency",
    "tire_management",
    "composure",
    "wet_skill",
)
"""Persona traits that change the race today: lap model (pace, consistency, tyre management),
baseline strategy (tyre management, wet skill), incidents (consistency, composure), weather
penalty (wet skill, M4). The others are stored for later stages. Kept honest by a test that
searches the engine sources."""


def _note_source(note: str | None, default: SourceType) -> str:
    """Source type named by a ``RaceSetup.inputs`` note prefix (``"<source_type>: ..."``)."""
    if note is None:
        return default.value
    prefix = note.split(":", 1)[0].strip()
    return prefix if prefix in {s.value for s in SourceType} else default.value


def driver_view(state: WorldState, driver_id: str) -> dict[str, Any]:
    """A driver's pinned profile and race inputs, each value labelled with its source type.

    Read from the world itself (the profile revision this world was created with), not from
    the config files, so a save shows the values its race actually used.
    """
    entry = state.setup.driver(driver_id)
    profile = entry.profile
    meta = profile.metadata
    inputs = state.setup.inputs
    car = state.car(driver_id)
    return {
        "driver_id": driver_id,
        "code": entry.code,
        "display_name": profile.display_name,
        "team_id": entry.team_id,
        "identity_source_type": profile.identity_source_type.value,
        "grid_position": {
            "value": entry.grid_position,
            "source_type": SourceType.HISTORICAL_VERIFIED.value,
        },
        "start_compound": {
            "value": entry.start_compound.value,
            "source_type": _note_source(
                inputs.get("start_compounds"), SourceType.EDITORIAL_ASSUMPTION
            ),
            "note": inputs.get("start_compounds", "editorial_assumption: synthetic default"),
        },
        "car_pace_offset_s": {
            "value": entry.car_pace_offset_s,
            "source_type": _note_source(
                inputs.get("car_pace_offsets"), SourceType.EDITORIAL_ASSUMPTION
            ),
            "note": inputs.get("car_pace_offsets", "editorial_assumption: equal cars"),
        },
        "profile": {
            "revision": profile.profile_revision,
            "content_hash": profile.content_hash,
            "source_type": meta.numeric_values_source_type.value,
            "estimated": meta.estimated,
            "confidence": meta.confidence,
            "persona_label": meta.persona_label,
            "base_template": meta.base_template,
            "overridden_fields": list(meta.overridden_fields),
            "explanation": meta.explanation,
            "origin": meta.origin,
            "values": profile.archetype.model_dump(),
            "stage1_fields": list(STAGE1_FIELDS),
            "used_by_engine": list(ENGINE_TRAITS),
        },
        "policy": {"type": profile.policy.type, "version": profile.policy.version},
        "race": {
            "status": car.status.value,
            "compound": car.compound.value,
            "tyre_age_laps": car.tyre_age_laps,
            "pit_stops": car.pit_stops,
            "pace_mode": car.pace_mode.value,
            "controller": car.controller.value,
            "plan": plan_text(car.planned_stops),
            "source_type": SourceType.SIMULATED_DYNAMIC.value,
        },
    }
