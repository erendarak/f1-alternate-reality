"""M0.5 debug viewer API (D-002): a thin HTTP layer over one live ``WorldRunner``.

The backend stays authoritative. The page only polls ``/api/debug/state`` and posts commands;
every command goes through the runner's queue and is applied at a tick boundary, exactly as
in the headless tools. This is a debug tool, replaced by the real API + React UI in M3.
"""

import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field

from f1ar.domain.race_state import PaceMode
from f1ar.presentation.track_geometry import TrackGeometry
from f1ar.presentation.view_model import state_view
from f1ar.simulation.world_runner import Command, CommandType, WorldRunner

STATIC = Path(__file__).with_name("static")
COMMAND_TIMEOUT_S = 2.0
RECENT_EVENTS = 12


class CommandRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: CommandType
    idempotency_key: str | None = None
    speed: int | None = None
    ticks: int = Field(default=1, ge=1, le=10_000)
    driver_id: str | None = None
    pace_mode: PaceMode | None = None
    label: str | None = None


def create_app(
    runner: WorldRunner,
    *,
    geometry: TrackGeometry,
    track_payload: dict[str, Any],
    team_colors: dict[str, str],
    run_background: bool = True,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        if run_background:
            runner.start_background()
        try:
            yield
        finally:
            if run_background:
                runner.stop_background()

    app = FastAPI(title="F1AR debug viewer", lifespan=lifespan)

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC / "debug_viewer.html")

    @app.get("/api/debug/track")
    def track() -> dict[str, Any]:
        return track_payload

    @app.get("/api/debug/state")
    def state() -> dict[str, Any]:
        events = runner.events
        return state_view(
            runner.state,
            geometry,
            team_colors=team_colors,
            speed=runner.speed,
            recent_events=events[-RECENT_EVENTS:],
        )

    @app.post("/api/debug/command")
    def command(req: CommandRequest) -> dict[str, Any]:
        cmd = Command(
            idempotency_key=req.idempotency_key or f"web-{uuid.uuid4().hex}",
            type=req.type,
            speed=req.speed,
            ticks=req.ticks,
            driver_id=req.driver_id,
            pace_mode=req.pace_mode,
            label=req.label,
        )
        future = runner.submit(cmd)
        if not run_background:
            runner.pump()  # tests: no thread, apply right away
        try:
            ack = future.result(timeout=COMMAND_TIMEOUT_S)
        except TimeoutError as exc:
            raise HTTPException(status_code=504, detail="runner did not acknowledge") from exc
        return ack.model_dump(mode="json")

    return app
