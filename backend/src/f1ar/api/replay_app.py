"""Real-race replay viewer API (mode ``HISTORICAL_REPLAY``, D-018).

Same page and same endpoint paths as the simulation viewer, so the frontend needs no second
copy. Only playback commands exist: START/RESUME (play), PAUSE, SET_SPEED, STEP (seconds
while paused) and SEEK (to a lap or a time). There is nothing to save, fork or steer,
because this is recorded history.
"""

from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

from f1ar.api.debug_app import add_shared_routes
from f1ar.history.reference import HistoricalRace
from f1ar.history.replay import ReplayClock, ReplayError
from f1ar.presentation.track_geometry import TrackGeometry
from f1ar.presentation.view_model import replay_view

TICK_SECONDS = 0.5  # the page's "Step tick" = 0.5 s, same as the simulation


class ReplayCommand(BaseModel):
    model_config = ConfigDict(extra="ignore")  # the page also sends idempotency keys

    type: Literal["START", "RESUME", "PAUSE", "SET_SPEED", "STEP", "SEEK"]
    speed: int | None = None
    ticks: int = Field(default=1, ge=1, le=100_000)
    lap: int | None = Field(default=None, ge=1)
    seconds: float | None = Field(default=None, ge=0)


def create_replay_app(
    race: HistoricalRace,
    clock: ReplayClock,
    *,
    geometry: TrackGeometry,
    track_payload: dict[str, Any],
    car_sprite: Path | None = None,
) -> FastAPI:
    app = FastAPI(title="F1AR replay viewer")
    add_shared_routes(app, track_payload, car_sprite)

    def lap_start(lap: int) -> float:
        """Replay time at which the race leader started ``lap``."""
        if lap > race.total_laps:
            raise ReplayError(f"race has {race.total_laps} laps")
        starts = [c.laps[lap - 1].start_s for c in race.cars if c.laps_completed >= lap]
        return min(starts) - race.race_start_s

    @app.get("/api/debug/state")
    def state() -> dict[str, Any]:
        replay_s, status, speed = clock.snapshot()
        return replay_view(race, geometry, replay_s=replay_s, status=status, speed=speed)

    @app.post("/api/debug/command")
    def command(cmd: ReplayCommand) -> dict[str, Any]:
        try:
            match cmd.type:
                case "START" | "RESUME":
                    clock.play()
                case "PAUSE":
                    clock.pause()
                case "SET_SPEED":
                    clock.set_speed(cmd.speed or 0)
                case "STEP":
                    clock.step(cmd.ticks * TICK_SECONDS)
                case "SEEK":
                    target = lap_start(cmd.lap) if cmd.lap is not None else cmd.seconds
                    if target is None:
                        raise ReplayError("SEEK needs lap or seconds")
                    clock.seek(target)
            accepted, reason = True, None
        except ReplayError as exc:
            accepted, reason = False, str(exc)
        replay_s, status, speed = clock.snapshot()
        return {
            "type": cmd.type,
            "accepted": accepted,
            "reason": reason,
            "applied_tick": None,
            "replay_s": replay_s,
            "world_status": status,
            "speed": speed,
            "snapshot_id": None,
        }

    return app
