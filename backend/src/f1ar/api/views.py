"""JSON views shared by the REST routes and the WebSocket stream (display only, pure reads)."""

from typing import Any

from f1ar.api.catalog import RaceSpec
from f1ar.api.session import LiveSession, LiveWorld
from f1ar.domain.race_state import RaceEvent, WorldState
from f1ar.persistence.sqlite_store import TimelineMeta
from f1ar.presentation.view_model import state_view

RECENT_EVENTS = 50


def world_view(state: WorldState, race: RaceSpec, *, speed: int) -> dict[str, Any]:
    view = state_view(state, race.assets.geometry, team_colors=race.assets.team_colors, speed=speed)
    del view["events"]  # sent separately (REST: /events; stream: per message)
    view["race_name"] = race.name
    view["parent_timeline_id"] = state.parent_timeline_id
    return view


def live_view(live: LiveWorld) -> dict[str, Any]:
    runner = live.runner
    view = world_view(runner.state, live.race, speed=runner.speed)
    view["persistence"] = runner.persistence
    return view


def event_json(event: RaceEvent) -> dict[str, Any]:
    return event.model_dump(mode="json")


def timeline_of(session: LiveSession, world_id: str, timeline_id: str | None) -> TimelineMeta:
    """``timeline_id`` of ``world_id`` (default: the live one, else the newest saved one)."""
    timelines = session.store.list_timelines(world_id)
    if timeline_id is None:
        live = session.live_world(world_id)
        if live is not None:
            timeline_id = live.runner.state.timeline_id
        else:
            latest = session.store.latest_valid(world_id)
            timeline_id = latest.timeline_id if latest is not None else None
    for t in timelines:
        if t.timeline_id == timeline_id:
            return t
    raise KeyError(f"timeline {timeline_id} of world {world_id}")


def timeline_events(session: LiveSession, world_id: str, timeline_id: str) -> list[RaceEvent]:
    """A timeline's full history: stored (with its ancestors' prefix) plus, for the live
    timeline, the runner's events not yet written."""
    events = {e.seq: e for e in session.store.events(timeline_id)}
    live = session.live_world(world_id)
    if live is not None and live.runner.state.timeline_id == timeline_id:
        events.update((e.seq, e) for e in live.runner.events)
    return [events[seq] for seq in sorted(events)]
