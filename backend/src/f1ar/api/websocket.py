"""``WS /ws/worlds/{world_id}``: the live world's state, streamed (plan §13.2, §13.4).

The server pushes; the client only listens (commands go through REST, so they are
acknowledged and logged). Messages:

- ``{"type": "state", "stream_seq", "generation", "reset", "state", "events"}``: sent when the
  world changed (tick, status, speed, timeline, save status), and at least every heartbeat.
  ``stream_seq`` increases by one per message, so a client can drop anything older than what it
  has. ``events`` are the race events since the previous message (on the first message, and
  whenever ``reset`` is true, the most recent ones: the client should replace its log).
  ``generation`` changes when the live runner is replaced (fork, load).
- ``{"type": "closed", "reason", "live_world_id"}``: this world stopped being the live one; the
  server closes the socket afterwards (code 4000).
- ``{"type": "error", "error": "world_not_live" | "unknown_world"}``: on connect; then close
  (code 4404 / 4409).

When the last client of a running world disconnects, the world is paused (MVP policy, §13.4).
"""

import asyncio
import contextlib
from typing import Any

import anyio
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from starlette.concurrency import run_in_threadpool

from f1ar.api.session import LiveSession, UnknownWorldError
from f1ar.api.views import RECENT_EVENTS, event_json, live_view, timeline_events

CLOSE_NOT_LIVE = 4409
CLOSE_UNKNOWN = 4404
CLOSE_UNLOADED = 4000


def stream_router(
    session: LiveSession, *, interval_s: float = 0.1, heartbeat_s: float = 1.0
) -> APIRouter:
    router = APIRouter()

    async def receive_until_closed(ws: WebSocket) -> None:
        with contextlib.suppress(WebSocketDisconnect):
            while True:
                await ws.receive_text()  # client messages are ignored

    async def send_frames(ws: WebSocket, world_id: str) -> None:
        loop = asyncio.get_running_loop()
        stream_seq = 0
        generation = -1
        last_key: tuple[Any, ...] | None = None
        last_sent = 0.0
        last_event_seq = -1
        while True:
            live = session.live_world(world_id)
            if live is None:
                current = session.live
                await ws.send_json(
                    {
                        "type": "closed",
                        "reason": "world_unloaded",
                        "live_world_id": None if current is None else current.world_id,
                    }
                )
                await ws.close(code=CLOSE_UNLOADED)
                return
            runner = live.runner
            state = runner.state
            persistence = runner.persistence
            key = (
                live.generation,
                state.timeline_id,
                state.sim_tick,
                state.world_status,
                runner.speed,
                tuple(sorted((k, str(v)) for k, v in persistence.items())),
            )
            now = loop.time()
            if key != last_key or now - last_sent >= heartbeat_s:
                reset = live.generation != generation
                if reset:  # whole recent history of the timeline, stored part included
                    history = await run_in_threadpool(
                        timeline_events, session, world_id, state.timeline_id
                    )
                    new_events = tuple(history[-RECENT_EVENTS:])
                else:
                    new_events = tuple(e for e in runner.events if e.seq > last_event_seq)
                if new_events:
                    last_event_seq = new_events[-1].seq
                stream_seq += 1
                await ws.send_json(
                    {
                        "type": "state",
                        "stream_seq": stream_seq,
                        "generation": live.generation,
                        "reset": reset,
                        "state": live_view(live),
                        "events": [event_json(e) for e in new_events],
                    }
                )
                generation, last_key, last_sent = live.generation, key, now
            await asyncio.sleep(interval_s)

    @router.websocket("/ws/worlds/{world_id}")
    async def stream(ws: WebSocket, world_id: str) -> None:
        await ws.accept()
        if session.live_world(world_id) is None:
            try:
                session.require_known(world_id)
                error, code = "world_not_live", CLOSE_NOT_LIVE
            except UnknownWorldError:
                error, code = "unknown_world", CLOSE_UNKNOWN
            live = session.live
            await ws.send_json(
                {
                    "type": "error",
                    "error": error,
                    "live_world_id": None if live is None else live.world_id,
                }
            )
            await ws.close(code=code)
            return
        session.client_connected(world_id)
        receiver = asyncio.create_task(receive_until_closed(ws))
        sender = asyncio.create_task(send_frames(ws, world_id))
        try:
            await asyncio.wait({receiver, sender}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            # shielded: the handler itself may be cancelled on disconnect (the test client does
            # this); the client count must still drop, or the world is never auto-paused
            with anyio.CancelScope(shield=True):
                for task in (receiver, sender):
                    task.cancel()
                for task in (receiver, sender):
                    # a send on a socket the client already closed may fail any which way
                    with contextlib.suppress(asyncio.CancelledError, Exception):
                        await task
                await run_in_threadpool(session.client_disconnected, world_id)

    return router
