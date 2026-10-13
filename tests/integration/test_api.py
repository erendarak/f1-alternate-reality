"""M3 API (D-027): REST + WebSocket over one live session, against a real save file.

The runner has no thread here and a frozen fake clock: ticks happen only through STEP, or when a
test moves the clock forward (the runner's own pacing), so every assertion is deterministic.
"""

import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from f1ar.api.catalog import RaceCatalog, RaceSpec
from f1ar.api.main import create_api
from f1ar.api.session import LiveSession
from f1ar.domain.driver_profiles import ProfileSet
from f1ar.domain.identities import RaceRoster
from f1ar.domain.race_state import RaceSetup, WorldStatus
from f1ar.persistence.sqlite_store import SnapshotStore
from f1ar.presentation.event_assets import EventAssets
from f1ar.presentation.track_geometry import TrackGeometry
from f1ar.presentation.view_model import track_view
from f1ar.simulation import lightweight_engine as engine
from f1ar.simulation.scenarios import build_setup
from f1ar.simulation.world_runner import Command, CommandType
from tests.conftest import HAM, MONZA_2024, VER

SQUARE = TrackGeometry.from_points([(0.0, 0.0), (250.0, 0.0), (250.0, 250.0), (0.0, 250.0)])


class FakeClock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


@pytest.fixture
def catalog(monza_roster: RaceRoster, monza_profiles: ProfileSet) -> RaceCatalog:
    by_code = {e.code: e.driver_id for e in monza_roster.entries}

    def build(codes: tuple[str, ...] | None, synthetic: bool, random_rain: bool) -> RaceSetup:
        ids = [VER, HAM] if codes is None else [by_code[c] for c in codes]
        return build_setup(monza_roster, monza_profiles, ids)

    assets = EventAssets(
        SQUARE,
        track_view(SQUARE, engine_length_m=5793.0, corners=[], rotation_deg=0.0, source="test"),
        {VER: "3671C6", HAM: "27F4D2"},
    )
    return RaceCatalog([RaceSpec(MONZA_2024, monza_roster, monza_profiles, assets, build, False)])


class Apps:
    """Start and stop API instances on one save file, like restarting the app."""

    def __init__(self, catalog: RaceCatalog, db: Path) -> None:
        self.catalog, self.db = catalog, db
        self.running: list[tuple[TestClient, SnapshotStore]] = []

    def start(self) -> TestClient:
        store = SnapshotStore(self.db)
        app = create_api(
            store,
            self.catalog,
            run_background=False,
            clock=FakeClock(),
            stream_interval_s=0.01,
            stream_heartbeat_s=0.2,
        )
        client = TestClient(app)
        client.__enter__()
        self.running.append((client, store))
        return client

    def stop(self, client: TestClient) -> None:
        """Stop an app the way Ctrl+C does (lifespan shutdown), then close its save file."""
        for i, (c, store) in enumerate(self.running):
            if c is client:
                c.__exit__(None, None, None)
                store.close()
                del self.running[i]
                return


@pytest.fixture
def apps(catalog: RaceCatalog, tmp_path: Path) -> Iterator[Apps]:
    a = Apps(catalog, tmp_path / "worlds.sqlite")
    yield a
    for client, _ in list(a.running):
        a.stop(client)


def create(client: TestClient, seed: int = 42) -> str:
    r = client.post("/api/worlds", json={"race_id": MONZA_2024, "seed": seed})
    assert r.status_code == 201, r.text
    world_id: str = r.json()["world_id"]
    return world_id


def post(client: TestClient, url: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
    r = client.post(url, json=body)
    assert r.status_code in (200, 201), r.text
    data: dict[str, Any] = r.json()
    return data


def state(client: TestClient, world_id: str) -> dict[str, Any]:
    data: dict[str, Any] = client.get(f"/api/worlds/{world_id}").json()["state"]
    return data


# --- catalog, health ----------------------------------------------------------------------


def test_health_and_races(apps: Apps) -> None:
    client = apps.start()
    health = client.get("/api/health").json()
    assert health["status"] == "ok" and health["engine_version"] == engine.ENGINE_VERSION
    assert health["races"] == [MONZA_2024] and health["live_world_id"] is None
    races = client.get("/api/races").json()["races"]
    assert races[0]["name"] == "2024 Italian GP" and races[0]["total_laps"] == 53
    race = client.get(f"/api/races/{MONZA_2024}").json()
    assert len(race["drivers"]) == 20
    assert race["drivers"][0]["source_type"] == "historical_verified"
    assert client.get(f"/api/races/{MONZA_2024}/track").json()["outline_length_m"] == 1000.0
    assert client.get("/api/races/1999_nowhere_gp").json()["error"] == "unknown_race"


# --- create / control ---------------------------------------------------------------------


def test_create_world_is_new_live_and_listed(apps: Apps) -> None:
    client = apps.start()
    a, b = create(client), create(client)
    assert a != b  # same race and seed, still a new world (US-01)
    listing = client.get("/api/worlds").json()
    assert listing["live_world_id"] == b
    assert [w["world_id"] for w in listing["worlds"]] == [b, a]
    assert [w["live"] for w in listing["worlds"]] == [True, False]
    assert listing["worlds"][0]["latest"]["label"] == "created"
    s = state(client, b)
    assert s["status"] == "CREATED" and s["tick"] == 0 and len(s["cars"]) == 2


def test_commands_are_acknowledged_by_the_runner(apps: Apps) -> None:
    client = apps.start()
    w = create(client)
    base = f"/api/worlds/{w}"
    assert post(client, f"{base}/start")["world_status"] == "RUNNING"
    refused = client.post(f"{base}/step", json={"ticks": 3})
    assert refused.status_code == 409 and refused.json()["accepted"] is False
    assert "PAUSED" in refused.json()["reason"]
    post(client, f"{base}/pause")
    assert post(client, f"{base}/step", {"ticks": 10})["applied_tick"] == 10
    assert post(client, f"{base}/speed", {"speed": 5})["speed"] == 5
    assert client.post(f"{base}/speed", json={"speed": 7}).status_code == 409
    pit = post(
        client,
        f"{base}/actions",
        {"action_type": "PIT_REQUEST", "driver_id": HAM, "compound": "HARD"},
    )
    assert pit["accepted"]
    car = next(c for c in state(client, w)["cars"] if c["driver_id"] == HAM)
    assert car["box"] == "HARD" and car["controller"] == "USER"
    missing = client.post(f"{base}/actions", json={"action_type": "PIT_REQUEST", "driver_id": VER})
    assert missing.status_code == 409 and "compound" in missing.json()["reason"]
    assert client.post(f"{base}/actions", json={"action_type": "TELEPORT"}).status_code == 422


def test_idempotency_key_applies_a_step_once(apps: Apps) -> None:
    client = apps.start()
    w = create(client)
    post(client, f"/api/worlds/{w}/start")
    post(client, f"/api/worlds/{w}/pause")
    body = {"ticks": 4, "idempotency_key": "double-click"}
    post(client, f"/api/worlds/{w}/step", body)
    again = post(client, f"/api/worlds/{w}/step", body)
    assert again["duplicate"] is True and state(client, w)["tick"] == 4


def test_running_world_advances_only_with_the_runner_pacing(apps: Apps) -> None:
    client = apps.start()
    w = create(client)
    post(client, f"/api/worlds/{w}/start")
    session: LiveSession = client.app.state.session  # type: ignore[attr-defined]
    live = session.require_live(w)
    clock = live.runner._clock  # the fake clock
    assert isinstance(clock, FakeClock)
    clock.t += 10.0  # 10 s of real time at x1 = 20 ticks of 0.5 s
    live.runner.pump()  # what the runner thread does
    ack = post(client, f"/api/worlds/{w}/pause")
    assert ack["applied_tick"] == 20 and ack["world_status"] == "PAUSED"
    clock.t += 60.0  # paused: no time is caught up
    assert post(client, f"/api/worlds/{w}/resume")["applied_tick"] == 20


def test_live_only_requests_on_a_stored_world_are_409(apps: Apps) -> None:
    client = apps.start()
    stored = create(client)
    live = create(client)
    r = client.post(f"/api/worlds/{stored}/start")
    assert r.status_code == 409
    assert r.json()["error"] == "world_not_live" and r.json()["live_world_id"] == live
    assert client.post("/api/worlds/world_nope/start").json()["error"] == "unknown_world"
    view = client.get(f"/api/worlds/{stored}").json()
    assert view["live"] is False and view["state"]["tick"] == 0  # read from the save


# --- save / load / fork (US-03, US-04) ----------------------------------------------------


def test_save_is_acknowledged_with_its_snapshot(apps: Apps) -> None:
    client = apps.start()
    w = create(client)
    post(client, f"/api/worlds/{w}/start")
    post(client, f"/api/worlds/{w}/pause")
    post(client, f"/api/worlds/{w}/step", {"ticks": 30})
    saved = post(client, f"/api/worlds/{w}/save", {"label": "before the stop"})
    assert saved["snapshot"]["label"] == "before the stop" and saved["snapshot"]["tick"] == 30
    timelines = client.get(f"/api/worlds/{w}/timelines").json()["timelines"]
    assert len(timelines) == 1
    labels = [s["label"] for s in timelines[0]["snapshots"]]
    assert "before the stop" in labels and "created" in labels


def test_play_pause_save_restart_load_resume_fork(apps: Apps) -> None:
    """The M3 user flow on the backend: nothing is lost by restarting the app, the loaded world
    continues exactly like the uninterrupted one, and a fork leaves its source untouched."""
    client = apps.start()
    w = create(client)
    base = f"/api/worlds/{w}"
    post(client, f"{base}/start")
    post(client, f"{base}/pause")
    post(client, f"{base}/step", {"ticks": 400})
    saved = post(client, f"{base}/save", {"label": "lap 2"})["snapshot"]
    session: LiveSession = client.app.state.session  # type: ignore[attr-defined]
    at_save = session.require_live(w).runner.state
    apps.stop(client)  # the whole backend goes away

    client = apps.start()
    listing = client.get("/api/worlds").json()
    assert listing["live_world_id"] is None and listing["worlds"][0]["world_id"] == w
    loaded = post(client, f"{base}/load", {"snapshot_id": saved["snapshot_id"]})
    assert loaded["branched"] is False  # nothing happened after the save: same timeline
    assert loaded["state"]["tick"] == 400 and loaded["state"]["status"] == "PAUSED"
    session = client.app.state.session  # type: ignore[attr-defined]
    live_state = session.require_live(w).runner.state
    assert live_state.canonical_hash() == at_save.canonical_hash()

    # resume: the loaded world continues exactly like the uninterrupted one
    post(client, f"{base}/step", {"ticks": 1000})
    expected = at_save
    for _ in range(1000):
        expected = engine.step(expected, allow_paused=True).state
    assert session.require_live(w).runner.state.canonical_hash() == expected.canonical_hash()
    main_timeline = session.require_live(w).runner.state.timeline_id

    # fork with a decision override
    fork = post(
        client,
        f"{base}/fork",
        {
            "name": "HAM boxes for hards",
            "action_overrides": [
                {"action_type": "PIT_REQUEST", "driver_id": HAM, "compound": "HARD"}
            ],
        },
    )
    assert fork["overrides_rejected"] == 0 and fork["parent_timeline_id"] == main_timeline
    assert fork["timeline_id"] != main_timeline and fork["state"]["tick"] == 1400
    ham = next(c for c in fork["state"]["cars"] if c["driver_id"] == HAM)
    assert ham["box"] == "HARD"
    source_point = session.store.load(fork["source_snapshot_id"])
    post(client, f"{base}/step", {"ticks": 500})

    # the source timeline is untouched by the branch
    assert session.store.load(fork["source_snapshot_id"]) == source_point
    main_events = session.store.events(main_timeline)
    assert all(e.tick <= 1400 for e in main_events)
    timelines = client.get(f"{base}/timelines").json()
    assert timelines["live_timeline_id"] == fork["timeline_id"]
    assert {t["timeline_id"] for t in timelines["timelines"]} == {
        main_timeline,
        fork["timeline_id"],
    }
    branch_log = session.require_live(w).runner.command_log  # the STEP isn't flushed yet
    assert [(c.command.type, c.command.actor_id) for c in branch_log] == [
        (CommandType.PIT_REQUEST, "USER"),
        (CommandType.STEP, "USER"),
    ]


def test_fork_a_stored_world_from_a_chosen_snapshot(apps: Apps) -> None:
    client = apps.start()
    w = create(client)
    post(client, f"/api/worlds/{w}/start")
    post(client, f"/api/worlds/{w}/pause")
    post(client, f"/api/worlds/{w}/step", {"ticks": 50})
    snap = post(client, f"/api/worlds/{w}/save", {"label": "early"})["snapshot"]
    create(client)  # another world becomes live; w is only stored now
    fork = post(
        client,
        f"/api/worlds/{w}/fork",
        {
            "name": "independent luck",
            "source_snapshot_id": snap["snapshot_id"],
            "seed_mode": "INDEPENDENT",
        },
    )
    assert fork["source_snapshot_id"] == snap["snapshot_id"] and fork["state"]["tick"] == 50
    assert fork["state"]["seed"] != 42
    assert client.get("/api/worlds").json()["live_world_id"] == w


def test_load_rejects_a_snapshot_of_another_world(apps: Apps) -> None:
    client = apps.start()
    a = create(client)
    a_created = client.get(f"/api/worlds/{a}/timelines").json()["timelines"][0]["snapshots"][0]
    b = create(client)
    r = client.post(f"/api/worlds/{b}/load", json={"snapshot_id": a_created["snapshot_id"]})
    assert r.status_code == 404 and r.json()["error"] == "not_found"


def test_load_after_a_crash_pauses_a_running_save(catalog: RaceCatalog, tmp_path: Path) -> None:
    """A save taken while RUNNING, with no clean shutdown after it (a crash), is loaded PAUSED by
    a logged SYSTEM command, on the same timeline."""
    store = SnapshotStore(tmp_path / "w.sqlite")
    crashed = LiveSession(store, catalog, run_background=False, clock=FakeClock())
    world = crashed.create(MONZA_2024, seed=7)
    crashed.command(world.world_id, Command(idempotency_key="s", type=CommandType.START))
    saved = crashed.command(world.world_id, Command(idempotency_key="v", type=CommandType.SAVE))
    # no shutdown: the process "dies" here

    fresh = LiveSession(store, catalog, run_background=False, clock=FakeClock())
    state, meta = fresh.load(world.world_id, saved.snapshot_id)
    assert meta.world_status is WorldStatus.RUNNING
    assert state.world_status is WorldStatus.PAUSED and state.timeline_id == meta.timeline_id
    fresh.shutdown()
    log = store.commands(state.timeline_id)
    assert (log[-1].type, log[-1].actor_id) == ("PAUSE", "SYSTEM")
    store.close()


def test_continue_after_a_crash_resumes_the_timeline_last_live(
    catalog: RaceCatalog, tmp_path: Path
) -> None:
    """Fork (branch gets the newest saves), go back to the main timeline, crash: "continue"
    must resume main, not the branch that happened to save last (found in M3 slice 3)."""
    store = SnapshotStore(tmp_path / "w.sqlite")
    first = LiveSession(store, catalog, run_background=False, clock=FakeClock())
    world = first.create(MONZA_2024, seed=3)
    w = world.world_id
    first.command(w, Command(idempotency_key="start", type=CommandType.START))
    first.command(w, Command(idempotency_key="pause", type=CommandType.PAUSE))
    first.command(w, Command(idempotency_key="step", type=CommandType.STEP, ticks=40))
    fork = first.fork(w, name="branch")
    first.command(w, Command(idempotency_key="b-step", type=CommandType.STEP, ticks=25))
    main_newest = [s for s in store.list_snapshots(w) if s.timeline_id == world.timeline_id][0]
    back, _ = first.load(w, main_newest.snapshot_id)  # back on main (newest saves are the branch's)
    assert back.timeline_id == world.timeline_id != fork.state.timeline_id
    # crash: no shutdown

    fresh = LiveSession(store, catalog, run_background=False, clock=FakeClock())
    resumed, _ = fresh.load(w)  # "Continue"
    assert resumed.timeline_id == world.timeline_id and resumed.sim_tick == 40
    fresh.shutdown()
    store.close()


def test_continue_without_snapshot_id_does_not_branch(apps: Apps) -> None:
    client = apps.start()
    w = create(client)
    post(client, f"/api/worlds/{w}/start")
    post(client, f"/api/worlds/{w}/pause")
    post(client, f"/api/worlds/{w}/step", {"ticks": 20})
    again = post(client, f"/api/worlds/{w}/load")  # "Continue" on the live world itself
    assert again["branched"] is False and again["state"]["tick"] == 20


# --- driver card ----------------------------------------------------------------------------


def test_driver_card_labels_every_source(apps: Apps) -> None:
    client = apps.start()
    w = create(client)
    card = client.get(f"/api/worlds/{w}/drivers/{HAM}").json()
    assert card["code"] == "HAM" and card["team_name"]
    assert card["identity_source_type"] == "historical_verified"
    assert card["grid_position"]["source_type"] == "historical_verified"
    assert card["profile"]["source_type"] == "editorial_assumption"
    assert card["start_compound"]["source_type"] == "editorial_assumption"  # synthetic setup
    assert card["race"]["source_type"] == "simulated_dynamic"
    assert set(card["profile"]["values"]) >= {"race_pace_skill", "tire_management"}
    assert client.get(f"/api/worlds/{w}/drivers/nobody_2024").status_code == 404


# --- events + stream ----------------------------------------------------------------------


def test_events_endpoint_includes_unsaved_live_events(apps: Apps) -> None:
    client = apps.start()
    w = create(client)
    post(client, f"/api/worlds/{w}/start")
    post(client, f"/api/worlds/{w}/pause")
    post(client, f"/api/worlds/{w}/actions", {"action_type": "DEPLOY_VSC", "duration_s": 30})
    events = client.get(f"/api/worlds/{w}/events").json()
    assert "VSC_DEPLOYED" in [e["type"] for e in events["events"]]
    seqs = [e["seq"] for e in events["events"]]
    assert seqs == sorted(seqs) == list(range(seqs[0], seqs[0] + len(seqs)))


def receive_until(ws: Any, predicate: Callable[[dict[str, Any]], bool]) -> dict[str, Any]:
    for _ in range(200):
        msg: dict[str, Any] = ws.receive_json()
        if predicate(msg):
            return msg
    raise AssertionError("no matching stream message")


def test_stream_pushes_state_changes_in_sequence(apps: Apps) -> None:
    client = apps.start()
    w = create(client)
    post(client, f"/api/worlds/{w}/start")
    post(client, f"/api/worlds/{w}/pause")
    with client.websocket_connect(f"/ws/worlds/{w}") as ws:
        first = ws.receive_json()
        assert first["type"] == "state" and first["stream_seq"] == 1 and first["reset"] is True
        assert first["state"]["status"] == "PAUSED"
        post(client, f"/api/worlds/{w}/step", {"ticks": 25})
        later = receive_until(ws, lambda m: m["state"]["tick"] == 25)
        assert later["stream_seq"] > first["stream_seq"] and later["reset"] is False
        post(client, f"/api/worlds/{w}/actions", {"action_type": "DEPLOY_SC", "laps": 3})
        sc = receive_until(
            ws, lambda m: any(e["type"] == "SAFETY_CAR_DEPLOYED" for e in m["events"])
        )
        assert sc["state"]["race_control"]["phase"] == "SC"

        # a fork replaces the runner: same world, new timeline, the client resets its log
        post(client, f"/api/worlds/{w}/fork", {"name": "branch"})
        branched = receive_until(ws, lambda m: m["reset"])
        assert branched["state"]["parent_timeline_id"] is not None
        assert any(e["type"] == "SAFETY_CAR_DEPLOYED" for e in branched["events"])  # history


def test_stream_closes_when_another_world_is_loaded(apps: Apps) -> None:
    client = apps.start()
    w = create(client)
    with client.websocket_connect(f"/ws/worlds/{w}") as ws:
        ws.receive_json()
        other = create(client)
        closed = receive_until(ws, lambda m: m["type"] == "closed")
        assert closed["live_world_id"] == other


def test_stream_refuses_worlds_that_are_not_live(apps: Apps) -> None:
    client = apps.start()
    stored = create(client)
    create(client)
    for world_id, error in ((stored, "world_not_live"), ("world_nope", "unknown_world")):
        with client.websocket_connect(f"/ws/worlds/{world_id}") as ws:
            assert ws.receive_json()["error"] == error
            with pytest.raises(WebSocketDisconnect):
                ws.receive_json()


def test_last_client_leaving_pauses_a_running_world(apps: Apps) -> None:
    client = apps.start()
    w = create(client)
    post(client, f"/api/worlds/{w}/start")
    with client.websocket_connect(f"/ws/worlds/{w}") as one:
        one.receive_json()
        with client.websocket_connect(f"/ws/worlds/{w}") as two:
            two.receive_json()
        time.sleep(0.2)
        assert state(client, w)["status"] == "RUNNING"  # one client still watching
    deadline = time.monotonic() + 5
    while state(client, w)["status"] != "PAUSED":
        assert time.monotonic() < deadline, "world was not auto-paused"
        time.sleep(0.05)
    session: LiveSession = client.app.state.session  # type: ignore[attr-defined]
    log = session.require_live(w).runner.command_log
    assert (log[-1].command.type, log[-1].command.actor_id) == (CommandType.PAUSE, "SYSTEM")
