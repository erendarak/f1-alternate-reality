"""Debug viewer: view model is a pure function of state; API only relays commands to the runner."""

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from f1ar.api.debug_app import create_app
from f1ar.domain.race_state import RaceSetup, WorldState
from f1ar.persistence.sqlite_store import SnapshotStore
from f1ar.presentation.track_geometry import TrackGeometry
from f1ar.presentation.view_model import state_view, track_view
from f1ar.simulation import lightweight_engine as engine
from f1ar.simulation.world_runner import WorldRunner
from tests.conftest import HAM, VER, run_ticks

# square "track" 1000 m around; engine length (5793 m) is mapped by fraction of a lap
SQUARE = TrackGeometry.from_points([(0.0, 0.0), (250.0, 0.0), (250.0, 250.0), (0.0, 250.0)])
COLORS = {VER: "3671C6", HAM: "27F4D2"}


def test_state_view_places_cars_by_lap_fraction(two_car_world: WorldState) -> None:
    state = run_ticks(two_car_world, 100)
    view = state_view(state, SQUARE, team_colors=COLORS, speed=5)
    assert [c["code"] for c in view["cars"]] == [
        state.setup.driver(d).code for d in engine.standings(state)
    ]
    for car in view["cars"]:
        fraction = state.car(car["driver_id"]).distance_m / state.setup.track.length_m
        assert (car["x"], car["y"]) == pytest.approx(SQUARE.point_at(fraction), abs=0.06)
        assert car["team_color"] == COLORS[car["driver_id"]]
    assert view["cars"][0]["gap_to_leader_s"] == 0.0
    assert view["status"] == "RUNNING" and view["tick"] == 100 and view["speed"] == 5


def test_state_view_does_not_change_state(two_car_world: WorldState) -> None:
    before = two_car_world.canonical_hash()
    state_view(two_car_world, SQUARE, team_colors={}, speed=1)
    assert two_car_world.canonical_hash() == before


def test_unknown_team_colour_falls_back(two_car_world: WorldState) -> None:
    view = state_view(two_car_world, SQUARE, team_colors={}, speed=1)
    assert {c["team_color"] for c in view["cars"]} == {"9AA0A6"}


def test_track_view_payload() -> None:
    payload = track_view(SQUARE, engine_length_m=5793.0, corners=[], rotation_deg=95.0, source="t")
    assert payload["outline"][0] == payload["start_finish"] == [0.0, 0.0]
    assert payload["outline_length_m"] == 1000.0 and payload["rotation_deg"] == 95.0


@pytest.fixture
def client(two_car_setup: RaceSetup, tmp_path: Path) -> Iterator[TestClient]:
    store = SnapshotStore(tmp_path / "v.sqlite")
    runner = WorldRunner(engine.create_world(two_car_setup, seed=42, world_id="w"), store=store)
    app = create_app(
        runner,
        geometry=SQUARE,
        track_payload=track_view(
            SQUARE, engine_length_m=5793.0, corners=[], rotation_deg=0.0, source="t"
        ),
        team_colors=COLORS,
        run_background=False,
    )
    with TestClient(app) as c:
        yield c
    store.close()


def test_page_and_track_are_served(client: TestClient) -> None:
    page = client.get("/")
    assert page.status_code == 200 and "F1AR Debug Viewer" in page.text
    assert client.get("/api/debug/track").json()["outline_length_m"] == 1000.0


def test_commands_go_through_the_runner(client: TestClient) -> None:
    assert client.get("/api/debug/state").json()["status"] == "CREATED"
    assert client.post("/api/debug/command", json={"type": "START"}).json()["accepted"]
    assert client.post("/api/debug/command", json={"type": "PAUSE"}).json()["accepted"]
    step = client.post("/api/debug/command", json={"type": "STEP", "ticks": 7}).json()
    assert step["accepted"] and step["applied_tick"] == 7
    push = client.post(
        "/api/debug/command",
        json={"type": "SET_PACE_MODE", "driver_id": HAM, "pace_mode": "PUSH"},
    ).json()
    assert push["accepted"]
    state = client.get("/api/debug/state").json()
    assert state["tick"] == 7 and state["status"] == "PAUSED"
    assert next(c for c in state["cars"] if c["driver_id"] == HAM)["pace_mode"] == "PUSH"
    assert state["events"][-1]["type"] == "PACE_MODE_CHANGED"


def test_save_and_rejections_are_reported(client: TestClient) -> None:
    rejected = client.post("/api/debug/command", json={"type": "RESUME"}).json()
    assert not rejected["accepted"] and rejected["reason"]
    client.post("/api/debug/command", json={"type": "START"})
    saved = client.post("/api/debug/command", json={"type": "SAVE"}).json()
    assert saved["accepted"] and saved["snapshot_id"].startswith("snap_")


def test_duplicate_idempotency_key_is_applied_once(client: TestClient) -> None:
    client.post("/api/debug/command", json={"type": "START"})
    client.post("/api/debug/command", json={"type": "PAUSE"})
    body = {"type": "STEP", "ticks": 3, "idempotency_key": "double-click"}
    client.post("/api/debug/command", json=body)
    again = client.post("/api/debug/command", json=body).json()
    assert again["duplicate"] and client.get("/api/debug/state").json()["tick"] == 3


def test_invalid_command_body_is_a_422(client: TestClient) -> None:
    assert client.post("/api/debug/command", json={"type": "TELEPORT"}).status_code == 422
