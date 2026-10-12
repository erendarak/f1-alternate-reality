"""M0.5 debug viewer API (D-002): a thin HTTP layer over one live ``WorldRunner``.

The backend stays authoritative. The page only polls ``/api/debug/state`` and posts commands;
every command goes through the runner's queue and is applied at a tick boundary, exactly as
in the headless tools. "Restart" never rewinds a world: it creates a brand-new world (same or
new seed) with a fresh runner. This is a debug tool, replaced by the real API + React UI in M3.
"""

import random
import threading
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field

from f1ar.domain.race_state import Compound, PaceMode, StrategyController, WorldState
from f1ar.persistence.sqlite_store import SnapshotStore
from f1ar.presentation.track_geometry import TrackGeometry
from f1ar.presentation.view_model import state_view
from f1ar.simulation.world_runner import Command, CommandType, WorldRunner

STATIC = Path(__file__).with_name("static")
COMMAND_TIMEOUT_S = 2.0
RECENT_EVENTS = 12

# Base sprite: palm-tree "Formula 1 inspired 2D pixel art car pack", Ferrari/Ferrari.png (16x16,
# nose up). Its bodywork is a single colour, recoloured per team in the browser; its white
# background becomes transparent. Terms: docs/LEGAL_AND_LICENSES.md.
SPRITE_PAINT_RGB = (211, 4, 0)
SPRITE_BACKGROUND_RGB = (255, 255, 255)
SPRITE_CREDIT = "Car sprite: palm-tree (itch.io), recoloured per team"

WorldFactory = Callable[[int, str], WorldState]
"""``(seed, world_id) -> new CREATED world``."""


def sprite_payload(car_sprite: Path | None) -> dict[str, Any] | None:
    """How the page recolours the base sprite (exact-match pixel replacement), or None."""
    if car_sprite is None:
        return None
    return {
        "url": "/assets/car.png",
        "paint_rgb": SPRITE_PAINT_RGB,
        "background_rgb": SPRITE_BACKGROUND_RGB,
        "credit": SPRITE_CREDIT,
    }


def add_shared_routes(app: FastAPI, track_payload: dict[str, Any], car_sprite: Path | None) -> None:
    """Page, car sprite and track: identical for the simulation and the replay viewer."""
    payload = {**track_payload, "car_sprite": sprite_payload(car_sprite)}

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC / "debug_viewer.html")

    @app.get("/assets/car.png", include_in_schema=False)
    def sprite() -> FileResponse:
        if car_sprite is None or not car_sprite.exists():
            raise HTTPException(status_code=404, detail="no car sprite configured")
        return FileResponse(car_sprite, media_type="image/png")

    @app.get("/api/debug/track")
    def track() -> dict[str, Any]:
        return payload


class CommandRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: CommandType
    idempotency_key: str | None = None
    speed: int | None = None
    ticks: int = Field(default=1, ge=1, le=10_000)
    driver_id: str | None = None
    pace_mode: PaceMode | None = None
    compound: Compound | None = None
    controller: StrategyController | None = None
    label: str | None = None


class RestartRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seed: int | None = Field(default=None, ge=0)
    """``None`` = keep the current seed (identical race); otherwise race with this seed."""
    new_seed: bool = False
    """Pick a fresh random seed (ignored if ``seed`` is given)."""


class _Session:
    """The one live runner; replaced wholesale on restart."""

    def __init__(
        self,
        factory: WorldFactory,
        seed: int,
        speed: int,
        store: SnapshotStore | None,
        background: bool,
    ) -> None:
        self.factory, self.store, self.background = factory, store, background
        self.lock = threading.Lock()
        self.generation = 0
        self.runner = self._make(seed, speed)

    def _make(self, seed: int, speed: int) -> WorldRunner:
        world = self.factory(seed, f"viewer_{self.generation}")
        return WorldRunner(world, speed=speed, store=self.store)

    def restart(self, seed: int) -> WorldRunner:
        with self.lock:
            old = self.runner
            if self.background:
                old.stop_background()
            self.generation += 1
            self.runner = self._make(seed, old.speed)
            if self.background:
                self.runner.start_background()
            return self.runner


def create_app(
    factory: WorldFactory,
    *,
    seed: int,
    speed: int,
    geometry: TrackGeometry,
    track_payload: dict[str, Any],
    team_colors: dict[str, str],
    store: SnapshotStore | None = None,
    car_sprite: Path | None = None,
    run_background: bool = True,
) -> FastAPI:
    session = _Session(factory, seed, speed, store, run_background)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        if run_background:
            session.runner.start_background()
        try:
            yield
        finally:
            if run_background:
                session.runner.stop_background()

    app = FastAPI(title="F1AR debug viewer", lifespan=lifespan)
    add_shared_routes(app, track_payload, car_sprite)

    @app.get("/api/debug/state")
    def state() -> dict[str, Any]:
        runner = session.runner
        return state_view(
            runner.state,
            geometry,
            team_colors=team_colors,
            speed=runner.speed,
            recent_events=runner.events[-RECENT_EVENTS:],
        )

    @app.post("/api/debug/command")
    def command(req: CommandRequest) -> dict[str, Any]:
        runner = session.runner
        cmd = Command(
            idempotency_key=req.idempotency_key or f"web-{uuid.uuid4().hex}",
            type=req.type,
            speed=req.speed,
            ticks=req.ticks,
            driver_id=req.driver_id,
            pace_mode=req.pace_mode,
            compound=req.compound,
            controller=req.controller,
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

    @app.post("/api/debug/restart")
    def restart(req: RestartRequest) -> dict[str, Any]:
        current = session.runner.state.rng.seed
        if req.seed is not None:
            new_seed = req.seed
        elif req.new_seed:
            new_seed = random.SystemRandom().randrange(1, 1_000_000)
        else:
            new_seed = current
        runner = session.restart(new_seed)
        return {"world_id": runner.state.world_id, "seed": new_seed, "status": "CREATED"}

    return app
