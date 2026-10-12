"""The M3 API application (plan §13.2): REST + WebSocket over one ``LiveSession``.

Contract: docs/API_CONTRACTS.md. Errors are JSON ``{"error": <code>, "detail": <text>}``:

========================  ======  =====================================================
error                     status  when
========================  ======  =====================================================
unknown_race / _world     404     no such race in the catalog / world in the save file
not_found                 404     unknown snapshot, timeline or driver
world_not_live            409     a live-only request for a world that isn't loaded
snapshot_unusable         409     checksum, schema, engine or policy mismatch (D-012)
command_rejected          409     fork's fork-point save was refused
command_timeout           504     the runner didn't acknowledge in time
invalid_request           422     body validation (FastAPI's own 422 shape) or bad value
========================  ======  =====================================================

Command endpoints answer with the command's ack: 200 accepted, 409 refused by the world.
"""

import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from f1ar.api.catalog import RaceCatalog, UnknownRaceError
from f1ar.api.routes_races import races_router
from f1ar.api.routes_worlds import worlds_router
from f1ar.api.session import (
    CommandRejectedError,
    CommandTimeoutError,
    LiveSession,
    UnknownWorldError,
    WorldNotLiveError,
)
from f1ar.api.websocket import stream_router
from f1ar.domain.race_state import WORLD_SCHEMA_VERSION
from f1ar.persistence.sqlite_store import SnapshotError, SnapshotNotFoundError, SnapshotStore
from f1ar.simulation.lightweight_engine import ENGINE_VERSION, POLICY_MANIFEST
from f1ar.simulation.world_runner import ALLOWED_SPEEDS, AutosavePolicy

API_VERSION = "1"


def _error(status: int, error: str, detail: str, **extra: Any) -> JSONResponse:
    return JSONResponse({"error": error, "detail": detail, **extra}, status_code=status)


def create_api(
    store: SnapshotStore,
    catalog: RaceCatalog,
    *,
    skipped_races: dict[str, str] | None = None,
    run_background: bool = True,
    clock: Callable[[], float] = time.monotonic,
    autosave: AutosavePolicy | None = None,
    auto_pause_on_disconnect: bool = True,
    stream_interval_s: float = 0.1,
    stream_heartbeat_s: float = 1.0,
) -> FastAPI:
    session = LiveSession(
        store,
        catalog,
        run_background=run_background,
        clock=clock,
        autosave=autosave,
        auto_pause_on_disconnect=auto_pause_on_disconnect,
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            session.shutdown()  # closing the app never loses the race (§4.7 test 8)

    app = FastAPI(title="F1 Alternate Reality API", version=API_VERSION, lifespan=lifespan)
    app.state.session = session

    @app.exception_handler(UnknownRaceError)
    async def unknown_race(_: Request, exc: UnknownRaceError) -> JSONResponse:
        return _error(404, "unknown_race", f"no simulated race {exc.args[0]!r}")

    @app.exception_handler(UnknownWorldError)
    async def unknown_world(_: Request, exc: UnknownWorldError) -> JSONResponse:
        return _error(404, "unknown_world", f"no world {exc.args[0]!r} in the save file")

    @app.exception_handler(SnapshotNotFoundError)
    async def unknown_snapshot(_: Request, exc: SnapshotNotFoundError) -> JSONResponse:
        return _error(404, "not_found", f"snapshot not found: {exc}")

    @app.exception_handler(KeyError)
    async def not_found(_: Request, exc: KeyError) -> JSONResponse:
        return _error(404, "not_found", f"not found: {exc.args[0] if exc.args else exc}")

    @app.exception_handler(WorldNotLiveError)
    async def not_live(_: Request, exc: WorldNotLiveError) -> JSONResponse:
        return _error(409, "world_not_live", str(exc), live_world_id=exc.live_world_id)

    @app.exception_handler(SnapshotError)
    async def unusable(_: Request, exc: SnapshotError) -> JSONResponse:
        return _error(409, "snapshot_unusable", str(exc))

    @app.exception_handler(CommandRejectedError)
    async def rejected(_: Request, exc: CommandRejectedError) -> JSONResponse:
        return _error(409, "command_rejected", str(exc), ack=exc.ack.model_dump(mode="json"))

    @app.exception_handler(CommandTimeoutError)
    async def timeout(_: Request, exc: CommandTimeoutError) -> JSONResponse:
        return _error(504, "command_timeout", str(exc))

    @app.exception_handler(ValueError)
    async def bad_value(_: Request, exc: ValueError) -> JSONResponse:
        return _error(422, "invalid_request", str(exc))

    @app.get("/api/health", tags=["meta"])
    def health() -> dict[str, Any]:
        live = session.live
        return {
            "status": "ok",
            "api_version": API_VERSION,
            "engine_version": ENGINE_VERSION,
            "world_schema_version": WORLD_SCHEMA_VERSION,
            "store_schema_version": store.schema_version,
            "policy_manifest": POLICY_MANIFEST,
            "allowed_speeds": list(ALLOWED_SPEEDS),
            "races": [r.race_id for r in catalog.all()],
            "live_world_id": None if live is None else live.world_id,
        }

    app.include_router(races_router(catalog, skipped_races or {}))
    app.include_router(worlds_router(session))
    app.include_router(
        stream_router(session, interval_s=stream_interval_s, heartbeat_s=stream_heartbeat_s)
    )
    return app
