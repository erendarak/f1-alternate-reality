"""JSON view models for the debug viewer: canonical state in, drawable data out (pure).

This is a spectator ("god") view, not a policy observation: it may show things a driver
can't know. Gaps are approximate display values derived from track position.
"""

from typing import Any

from f1ar.domain.race_state import CarStatus, RaceEvent, WorldState
from f1ar.presentation.track_geometry import TrackGeometry
from f1ar.simulation.lightweight_engine import standings

DEFAULT_COLOR = "9AA0A6"


def track_view(
    geometry: TrackGeometry,
    *,
    engine_length_m: float,
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
        if car.status is CarStatus.FINISHED and leader_car.finish_time_s is not None:
            gap = (car.finish_time_s or 0.0) - leader_car.finish_time_s
        else:
            gap = (_progress_m(state, leader) - _progress_m(state, driver_id)) / nominal_speed
        cars.append(
            {
                "position": pos,
                "driver_id": driver_id,
                "code": entry.code,
                "team_color": team_colors.get(driver_id, DEFAULT_COLOR),
                "x": round(x, 1),
                "y": round(y, 1),
                "heading_deg": round(geometry.heading_at(fraction), 1),
                "lap": min(car.laps_completed + 1, track.total_laps),
                "laps_completed": car.laps_completed,
                "status": car.status.value,
                "compound": car.compound.value,
                "tyre_age_laps": car.tyre_age_laps,
                "pace_mode": car.pace_mode.value,
                "lap_pace_mode": car.lap_pace_mode.value,
                "last_lap_time_s": car.last_lap_time_s,
                "gap_to_leader_s": round(gap, 3),
            }
        )
    return {
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
        "cars": cars,
        "events": [e.model_dump(mode="json") for e in recent_events],
    }
