"""Opt-in: fetch the real 2024 Italian GP through FastF1 (uses .cache/fastf1 when warm).

Run with:  uv run pytest -m network
"""

import pytest

from f1ar.data.fastf1_adapter import load_race_session, tables_from_session
from f1ar.data.quality import build_quality_report
from f1ar.domain.identities import RaceRoster
from tests.conftest import REPO_ROOT

pytestmark = pytest.mark.network


def test_real_monza_2024_has_no_quality_failures(monza_roster: RaceRoster) -> None:
    session = load_race_session(2024, "Italian Grand Prix", REPO_ROOT / ".cache" / "fastf1")
    tables = tables_from_session(session)
    report = build_quality_report(tables, monza_roster, source={}, expected_total_laps=53)
    failures = [c for c in report.checks if c.status == "FAIL"]
    assert not failures, failures
    assert report.table_rows["laps"] == 1008
    assert tables.results.iloc[0]["driver_code"] == "LEC"  # used only as a data check


def test_real_race_reference_reproduces_the_timing_feed() -> None:
    from f1ar.history.reference import HistoricalRace

    session = load_race_session(
        2024, "Italian Grand Prix", REPO_ROOT / ".cache" / "fastf1", telemetry=False
    )
    tables = tables_from_session(session)
    race = HistoricalRace.from_tables("2024_italian_gp", tables.laps, tables.results)
    assert race.validate_positions() == []  # our lap-end ordering == feed positions, every lap
    final = race.classification_after_lap(53)
    assert [code for code, _ in final[:3]] == ["LEC", "PIA", "NOR"]
    assert final[1][1] is not None and 2.5 < final[1][1] < 2.9  # official gap 2.664 s
    assert {e.code for e in race.events() if e.type == "RETIREMENT"} == {"TSU"}
