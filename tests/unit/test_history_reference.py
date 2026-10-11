"""Historical reference race from normalised tables (synthetic, offline)."""

import pandas as pd
import pytest

from f1ar.history.reference import MODE, HistoricalRace

# Three cars. A wins (3 laps, pits on lap 2). B finishes 2nd, lapped (2 laps).
# C retires after lap 1. Lap times: A 90/92/88, B 95/96, C 100.
SPEC = {
    1: ("AAA", [90.0, 92.0, 88.0], 2, "Finished", 1),
    2: ("BBB", [95.0, 96.0], None, "Lapped", 2),
    3: ("CCC", [100.0], None, "Retired", 3),
}


def tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for number, (code, times, pit_lap, _, _) in SPEC.items():
        t = 1000.0
        for i, lap_time in enumerate(times, start=1):
            rows.append(
                {
                    "driver_number": number,
                    "driver_code": code,
                    "lap_number": i,
                    "lap_time_s": lap_time,
                    "lap_start_time_s": t,
                    "lap_end_time_s": t + lap_time,
                    "pit_in_time_s": t + lap_time if i == pit_lap else pd.NA,
                    "pit_out_time_s": t if pit_lap and i == pit_lap + 1 else pd.NA,
                    "compound": "MEDIUM" if not pit_lap or i <= pit_lap else "HARD",
                    "tyre_life_laps": i,
                    "stint": 1,
                    "position": pd.NA,
                }
            )
            t += lap_time
    laps = pd.DataFrame(rows)
    results = pd.DataFrame(
        [
            {
                "driver_number": n,
                "driver_code": code,
                "team_name": "T",
                "team_color": "FF0000",
                "grid_position": grid,
                "status": status,
                "position": grid,
            }
            for n, (code, _, _, status, grid) in SPEC.items()
        ]
    )
    return laps, results


@pytest.fixture
def race() -> HistoricalRace:
    laps, results = tables()
    return HistoricalRace.from_tables("test", laps, results)


def test_basic_shape(race: HistoricalRace) -> None:
    assert race.mode == MODE == "HISTORICAL_REPLAY"
    assert race.total_laps == 3 and race.race_start_s == 1000.0
    assert race.car("AAA").laps_completed == 3 and race.car("CCC").laps_completed == 1
    assert race.duration_s == 270.0


def test_classification_uses_line_crossing_times(race: HistoricalRace) -> None:
    assert race.classification_after_lap(1) == [("AAA", 0.0), ("BBB", 5.0), ("CCC", 10.0)]
    # after lap 2: A at 1182, B at 1191; C retired
    assert race.classification_after_lap(2) == [("AAA", 0.0), ("BBB", 9.0), ("CCC", None)]
    # after lap 3 only A has 3 laps; lapped B is ahead of retired C
    assert race.classification_after_lap(3) == [("AAA", 0.0), ("BBB", None), ("CCC", None)]


def test_events(race: HistoricalRace) -> None:
    events = [(e.lap, e.type, e.code, e.detail) for e in race.events()]
    assert (2, "PIT_STOP", "AAA", "MEDIUM -> HARD") in events
    assert (1, "RETIREMENT", "CCC", "Retired after 1 laps") in events
    assert (2, "LAPPED_FINISH", "BBB", "P2, 2 laps") in events
    assert events[-1] == (3, "FINISHED", "AAA", "P1")
    times = [e.session_time_s for e in race.events()]
    assert times == sorted(times)


def test_progress_interpolates_within_a_lap(race: HistoricalRace) -> None:
    laps, frac, state = race.progress_at(45.0)["AAA"]
    assert (laps, state) == (0, "RUNNING") and frac == pytest.approx(0.5)
    laps, frac, state = race.progress_at(90.0 + 46.0)["AAA"]
    assert (laps, state) == (1, "RUNNING") and frac == pytest.approx(0.5)
    assert race.progress_at(200.0)["CCC"] == (1, 0.0, "RETIRED")
    assert race.progress_at(10_000.0)["AAA"] == (3, 0.0, "FINISHED")
    assert race.progress_at(10_000.0)["BBB"] == (2, 0.0, "FINISHED")


def test_feed_position_mismatch_is_reported() -> None:
    laps, results = tables()
    laps.loc[(laps["driver_code"] == "AAA") & (laps["lap_number"] == 1), "position"] = 2
    race = HistoricalRace.from_tables("test", laps, results)
    assert ("AAA", 1, 1, 2) in race.validate_positions()


def test_untimed_laps_are_skipped() -> None:
    laps, results = tables()
    laps.loc[0, "lap_end_time_s"] = pd.NA
    race = HistoricalRace.from_tables("test", laps, results)
    assert race.car("AAA").laps_completed == 2
