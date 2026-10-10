from typing import Any

import pytest
from pydantic import ValidationError

from f1ar.data.jolpica_adapter import roster_from_results, slug
from f1ar.domain.identities import RaceRoster, load_roster
from f1ar.domain.provenance import SourceType
from tests.conftest import CONFIGS, MONZA_2024


def _result(number: str, driver_id: str, given: str, family: str, code: str, grid: str) -> Any:
    return {
        "number": number,
        "position": "1",
        "grid": grid,
        "status": "Finished",
        "Driver": {"driverId": driver_id, "givenName": given, "familyName": family, "code": code},
        "Constructor": {"constructorId": "haas", "name": "Haas F1 Team"},
    }


JOLPICA_FIXTURE = {
    "MRData": {
        "RaceTable": {
            "Races": [
                {
                    "season": "2024",
                    "round": "16",
                    "date": "2024-09-01",
                    "Circuit": {"circuitId": "monza"},
                    "Results": [
                        _result("20", "kevin_magnussen", "Kevin", "Magnussen", "MAG", "0"),
                        _result("27", "hulkenberg", "Nico", "Hülkenberg", "HUL", "10"),
                    ],
                }
            ]
        }
    }
}


def _build() -> RaceRoster:
    return roster_from_results(
        JOLPICA_FIXTURE, event_id="test_event", source_url="fixture", retrieved_utc="t"
    )


def test_slug_strips_accents() -> None:
    assert slug("Nico Hülkenberg") == "nico_hulkenberg"


def test_transform_builds_ids_and_puts_pit_lane_starters_last() -> None:
    roster = _build()
    assert roster.driver_ids() == ("nico_hulkenberg_2024", "kevin_magnussen_2024")
    hul = roster.entry("nico_hulkenberg_2024")
    assert hul.person_id == "nico_hulkenberg"
    assert hul.team_id == "haas_2024"
    assert hul.external_ids["jolpica_driver_id"] == "hulkenberg"
    assert roster.source_type is SourceType.HISTORICAL_VERIFIED


def test_finishing_results_are_not_part_of_the_roster() -> None:
    dumped = _build().model_dump()
    assert all("position" not in e and "status" not in e for e in dumped["entries"])


def test_duplicate_car_number_rejected() -> None:
    data = _build().model_dump(mode="json")
    data["entries"][1]["car_number"] = data["entries"][0]["car_number"]
    with pytest.raises(ValidationError, match="car_number"):
        RaceRoster.model_validate(data)


def test_committed_monza_roster(monza_roster: RaceRoster) -> None:
    assert monza_roster.season == 2024
    assert monza_roster.circuit_id == "monza"
    assert len(monza_roster.entries) == 20
    assert [e.grid_position for e in monza_roster.entries] == list(range(1, 21))
    assert monza_roster.entry("max_verstappen_2024").team_id == "red_bull_2024"
    assert monza_roster.source.name == "jolpica-f1"


def test_roster_is_never_reused_for_another_event() -> None:
    with pytest.raises(ValueError, match="race-specific"):
        load_roster(CONFIGS / "rosters" / f"{MONZA_2024}.json", expected_event_id="2026_italian_gp")
