"""Replay clock (fake time) and replay viewer API over a synthetic recorded race."""

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from f1ar.api.replay_app import create_replay_app
from f1ar.history.reference import HistoricalRace
from f1ar.history.replay import ReplayClock, ReplayError
from f1ar.presentation.track_geometry import TrackGeometry
from f1ar.presentation.view_model import replay_view, track_view
from tests.unit.test_history_reference import tables

SQUARE = TrackGeometry.from_points([(0.0, 0.0), (250.0, 0.0), (250.0, 250.0), (0.0, 250.0)])


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def race() -> HistoricalRace:
    laps, results = tables()
    return HistoricalRace.from_tables("test", laps, results)


# --- clock ----------------------------------------------------------------------------------------


def test_clock_plays_pauses_and_never_catches_up() -> None:
    t = FakeClock()
    clock = ReplayClock(1000.0, speed=10, clock=t)
    assert clock.snapshot() == (0.0, "READY", 10)
    clock.play()
    t.now = 5.0
    assert clock.snapshot()[0] == 50.0
    clock.pause()
    t.now = 500.0
    assert clock.snapshot() == (50.0, "PAUSED", 10)
    clock.play()
    t.now = 501.0
    assert clock.snapshot()[0] == 60.0


def test_clock_speed_seek_step_and_finish() -> None:
    t = FakeClock()
    clock = ReplayClock(100.0, speed=10, clock=t)
    clock.seek(40.0)
    assert clock.snapshot() == (40.0, "PAUSED", 10)
    clock.step(2.5)
    assert clock.snapshot()[0] == 42.5
    clock.set_speed(50)
    clock.play()
    t.now = 10.0
    assert clock.snapshot() == (100.0, "FINISHED", 50)
    with pytest.raises(ReplayError):
        clock.play()
    clock.seek(0.0)
    assert clock.snapshot()[1] == "PAUSED"


@pytest.mark.parametrize(
    "bad", [lambda c: c.set_speed(3), lambda c: c.seek(-1), lambda c: c.seek(1e9)]
)
def test_clock_rejects_bad_commands(bad: object) -> None:
    clock = ReplayClock(100.0, clock=FakeClock())
    with pytest.raises(ReplayError):
        bad(clock)  # type: ignore[operator]


# --- view ------------------------------------------------------------------------------------


def test_replay_view_matches_the_recording(race: HistoricalRace) -> None:
    mid = replay_view(race, SQUARE, replay_s=45.0, status="PLAYING", speed=10)
    assert mid["mode"] == "HISTORICAL_REPLAY" and mid["status"] == "RUNNING"
    assert [c["code"] for c in mid["cars"]] == ["AAA", "BBB", "CCC"]
    aaa = mid["cars"][0]
    assert (aaa["x"], aaa["y"]) == pytest.approx(SQUARE.point_at(0.5), abs=0.06)

    later = replay_view(race, SQUARE, replay_s=150.0, status="PAUSED", speed=10)
    by_code = {c["code"]: c for c in later["cars"]}
    assert by_code["CCC"]["status"] == "RETIRED" and later["cars"][-1]["code"] == "CCC"
    # AAA pits at the end of lap 2 (its in-lap on mediums); lap 3 starts at +182 s on hards
    assert (by_code["AAA"]["compound"], by_code["AAA"]["pace_mode"]) == ("MEDIUM", "IN LAP")
    out_lap = {
        c["code"]: c
        for c in replay_view(race, SQUARE, replay_s=190.0, status="PAUSED", speed=10)["cars"]
    }
    assert (out_lap["AAA"]["compound"], out_lap["AAA"]["pace_mode"]) == ("HARD", "OUT LAP")

    end = replay_view(race, SQUARE, replay_s=race.duration_s, status="FINISHED", speed=10)
    assert end["status"] == "COMPLETED"
    assert [c["status"] for c in end["cars"]] == ["FINISHED", "FINISHED", "RETIRED"]
    assert end["cars"][1]["laps_down"] == 1
    assert {e["type"] for e in end["events"]} >= {"PIT_STOP", "RETIREMENT", "FINISHED"}


# --- API -------------------------------------------------------------------------------------


@pytest.fixture
def client(race: HistoricalRace) -> Iterator[TestClient]:
    t = FakeClock()
    clock = ReplayClock(race.duration_s, speed=10, clock=t)
    payload = track_view(SQUARE, engine_length_m=1000.0, corners=[], rotation_deg=0.0, source="t")
    app = create_replay_app(race, clock, geometry=SQUARE, track_payload=payload)
    with TestClient(app) as c:
        yield c


def test_replay_api_play_pause_seek(client: TestClient) -> None:
    assert client.get("/api/debug/state").json()["status"] == "CREATED"
    assert client.post("/api/debug/command", json={"type": "START"}).json()["accepted"]
    assert client.post("/api/debug/command", json={"type": "PAUSE"}).json()["accepted"]
    seek = client.post("/api/debug/command", json={"type": "SEEK", "lap": 2}).json()
    assert seek["accepted"] and seek["replay_s"] == 90.0  # leader AAA started lap 2 at +90 s
    step = client.post("/api/debug/command", json={"type": "STEP", "ticks": 4}).json()
    assert step["replay_s"] == 92.0
    state = client.get("/api/debug/state").json()
    assert state["leader_lap"] == 2 and state["tick"] is None


def test_replay_api_has_no_simulation_commands(client: TestClient) -> None:
    assert client.post("/api/debug/command", json={"type": "SAVE"}).status_code == 422
    assert client.post("/api/debug/command", json={"type": "SET_PACE_MODE"}).status_code == 422
    bad = client.post("/api/debug/command", json={"type": "SEEK", "lap": 99}).json()
    assert not bad["accepted"] and "laps" in bad["reason"]
    assert client.get("/api/debug/track").json()["car_sprite"] is None


def test_cars_just_behind_the_line_are_not_a_lap_down(race: HistoricalRace) -> None:
    """Regression: right after the leader crosses the line, the rest are behind, not lapped."""
    view = replay_view(race, SQUARE, replay_s=91.0, status="PAUSED", speed=10)
    bbb = next(c for c in view["cars"] if c["code"] == "BBB")
    assert bbb["laps_down"] == 0 and bbb["gap_to_leader_s"] is not None
    late = replay_view(race, SQUARE, replay_s=262.0, status="PAUSED", speed=10)
    assert next(c for c in late["cars"] if c["code"] == "BBB")["laps_down"] >= 0
