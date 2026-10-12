"""World library and live-world control (plan §13.2).

Every state change goes through the runner's command queue and is applied at a tick boundary.
A command response is its acknowledgement: HTTP 200 if accepted, 409 if the world refused it
(the body is the ack either way). Success is never reported before the ack, and a SAVE is
acknowledged only after the database commit (US-09).
"""

import random
from typing import Any

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

from f1ar.api.schemas import (
    ActionRequest,
    ControlRequest,
    CreateWorldRequest,
    ForkRequest,
    LoadRequest,
    SaveRequest,
    SpeedRequest,
    StepRequest,
    control,
    save_command,
    speed_command,
    step_command,
)
from f1ar.api.session import LiveSession
from f1ar.api.views import (
    RECENT_EVENTS,
    event_json,
    live_view,
    timeline_events,
    timeline_of,
    world_view,
)
from f1ar.persistence.sqlite_store import SnapshotMeta
from f1ar.presentation.view_model import driver_view
from f1ar.simulation.world_runner import CommandAck, CommandType


def ack_response(ack: CommandAck, **extra: Any) -> JSONResponse:
    return JSONResponse(
        {**ack.model_dump(mode="json"), **extra}, status_code=200 if ack.accepted else 409
    )


def _meta(meta: SnapshotMeta) -> dict[str, Any]:
    return meta.model_dump(mode="json")


def worlds_router(session: LiveSession) -> APIRouter:
    router = APIRouter(prefix="/api/worlds", tags=["worlds"])
    store, catalog = session.store, session.catalog

    # --- library -----------------------------------------------------------------------

    @router.get("")
    def list_worlds() -> dict[str, Any]:
        live = session.live
        worlds = []
        for w in store.list_worlds():
            if w.snapshot_count == 0:
                continue  # a world whose snapshots were all deleted can't be opened
            worlds.append(
                {
                    **w.model_dump(mode="json"),
                    "race_name": catalog.get(w.event_id).name if w.event_id in catalog else None,
                    "race_available": w.event_id in catalog,
                    "live": live is not None and live.world_id == w.world_id,
                }
            )
        return {"live_world_id": None if live is None else live.world_id, "worlds": worlds}

    @router.post("", status_code=201)
    def create_world(body: CreateWorldRequest) -> dict[str, Any]:
        seed = body.seed if body.seed is not None else random.SystemRandom().randrange(1, 10**6)
        state = session.create(
            body.race_id, seed=seed, codes=body.drivers, synthetic=body.synthetic, speed=body.speed
        )
        live = session.require_live(state.world_id)
        return {"world_id": state.world_id, "seed": seed, "state": live_view(live)}

    @router.get("/{world_id}")
    def get_world(world_id: str) -> dict[str, Any]:
        live = session.live_world(world_id)
        if live is not None:
            return {"world_id": world_id, "live": True, "state": live_view(live)}
        session.require_known(world_id)
        latest = store.latest_valid(world_id)
        state = session.current_state(world_id)
        race = catalog.get(state.setup.event_id)
        view = world_view(state, race, speed=1)
        return {
            "world_id": world_id,
            "live": False,
            "latest_snapshot": None if latest is None else _meta(latest),
            "state": view,
        }

    @router.get("/{world_id}/timelines")
    def timelines(world_id: str) -> dict[str, Any]:
        session.require_known(world_id)
        snapshots = store.list_snapshots(world_id)
        live = session.live_world(world_id)
        return {
            "world_id": world_id,
            "live_timeline_id": None if live is None else live.runner.state.timeline_id,
            "timelines": [
                {
                    **t.model_dump(mode="json"),
                    "snapshots": [_meta(s) for s in snapshots if s.timeline_id == t.timeline_id],
                }
                for t in store.list_timelines(world_id)
            ],
        }

    @router.get("/{world_id}/events")
    def events(
        world_id: str,
        timeline_id: str | None = None,
        since_seq: int = Query(default=-1, ge=-1),
        limit: int = Query(default=RECENT_EVENTS, ge=1, le=5000),
    ) -> dict[str, Any]:
        session.require_known(world_id)
        timeline = timeline_of(session, world_id, timeline_id)
        found = [e for e in timeline_events(session, world_id, timeline.timeline_id)]
        newer = [e for e in found if e.seq > since_seq]
        return {
            "world_id": world_id,
            "timeline_id": timeline.timeline_id,
            "total": len(found),
            "events": [event_json(e) for e in newer[-limit:]],
        }

    @router.get("/{world_id}/drivers/{driver_id}")
    def driver(world_id: str, driver_id: str) -> dict[str, Any]:
        state = session.current_state(world_id)
        view = driver_view(state, driver_id)
        race = catalog.get(state.setup.event_id)
        entry = race.roster.entry(driver_id)
        view["team_name"] = entry.team_name
        view["car_number"] = entry.car_number
        view["team_color"] = race.assets.team_colors.get(driver_id)
        return view

    # --- live control ------------------------------------------------------------------

    @router.post("/{world_id}/start")
    def start(world_id: str, body: ControlRequest | None = None) -> JSONResponse:
        return ack_response(session.command(world_id, control(CommandType.START, body)))

    @router.post("/{world_id}/pause")
    def pause(world_id: str, body: ControlRequest | None = None) -> JSONResponse:
        return ack_response(session.command(world_id, control(CommandType.PAUSE, body)))

    @router.post("/{world_id}/resume")
    def resume(world_id: str, body: ControlRequest | None = None) -> JSONResponse:
        return ack_response(session.command(world_id, control(CommandType.RESUME, body)))

    @router.post("/{world_id}/speed")
    def speed(world_id: str, body: SpeedRequest) -> JSONResponse:
        return ack_response(session.command(world_id, speed_command(body)))

    @router.post("/{world_id}/step")
    def step(world_id: str, body: StepRequest) -> JSONResponse:
        return ack_response(session.command(world_id, step_command(body)))

    @router.post("/{world_id}/save")
    def save(world_id: str, body: SaveRequest | None = None) -> JSONResponse:
        ack = session.command(world_id, save_command(body))
        snapshot = (
            _meta(store.get_meta(ack.snapshot_id))
            if ack.accepted and ack.snapshot_id is not None
            else None
        )
        return ack_response(ack, snapshot=snapshot)

    @router.post("/{world_id}/actions")
    def action(world_id: str, body: ActionRequest) -> JSONResponse:
        return ack_response(session.command(world_id, body.to_command()))

    # --- load / fork -------------------------------------------------------------------

    @router.post("/{world_id}/load")
    def load(world_id: str, body: LoadRequest | None = None) -> dict[str, Any]:
        body = body or LoadRequest()
        state, loaded = session.load(world_id, body.snapshot_id, speed=body.speed)
        live = session.require_live(world_id)
        return {
            "world_id": world_id,
            "loaded_snapshot": _meta(loaded),
            "timeline_id": state.timeline_id,
            "branched": state.timeline_id != loaded.timeline_id,
            "state": live_view(live),
        }

    @router.post("/{world_id}/fork", status_code=201)
    def fork(world_id: str, body: ForkRequest) -> dict[str, Any]:
        result = session.fork(
            world_id,
            name=body.name,
            source_snapshot_id=body.source_snapshot_id,
            seed_mode=body.seed_mode,
            overrides=[a.to_command() for a in body.action_overrides],
            speed=body.speed,
        )
        live = session.require_live(world_id)
        return {
            "world_id": world_id,
            "timeline_id": result.state.timeline_id,
            "parent_timeline_id": result.state.parent_timeline_id,
            "source_snapshot_id": result.source_snapshot_id,
            "branch_point": _meta(result.branch_point),
            "override_acks": [a.model_dump(mode="json") for a in result.acks],
            "overrides_rejected": sum(not a.accepted for a in result.acks),
            "state": live_view(live),
        }

    return router
