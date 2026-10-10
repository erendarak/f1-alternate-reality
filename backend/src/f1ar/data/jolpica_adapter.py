"""Jolpica-F1 (Ergast successor) adapter: https://github.com/jolpica/jolpica-f1

Only the network call lives in ``fetch_json``; everything else is a pure transform so it can
be tested offline with fixtures.
"""

import json
import re
import unicodedata
import urllib.request
from typing import Any

from f1ar.domain.identities import RaceRoster
from f1ar.domain.provenance import SourceType

BASE_URL = "https://api.jolpi.ca/ergast/f1"
USER_AGENT = "f1ar/0.0.1 (non-commercial research project)"


def results_url(season: int, round_: int) -> str:
    return f"{BASE_URL}/{season}/{round_}/results.json"


def fetch_json(url: str, timeout_s: float = 30.0) -> Any:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        return json.loads(response.read().decode("utf-8"))


def slug(text: str) -> str:
    """ASCII snake_case slug: ``"Nico Hülkenberg"`` -> ``"nico_hulkenberg"``."""
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "_", ascii_text.lower()).strip("_")


def roster_from_results(
    payload: dict[str, Any], *, event_id: str, source_url: str, retrieved_utc: str
) -> RaceRoster:
    """Build a race roster from a Jolpica ``results.json`` payload.

    Only entry/start information is kept (identity, team, number, grid). Finishing results
    are reference data for a later milestone and are deliberately not part of the roster.
    """
    races = payload["MRData"]["RaceTable"]["Races"]
    if len(races) != 1:
        raise ValueError(f"expected exactly one race in payload, got {len(races)}")
    race = races[0]
    season = int(race["season"])

    entries = []
    for result in race["Results"]:
        driver = result["Driver"]
        constructor = result["Constructor"]
        person_id = slug(f"{driver['givenName']} {driver['familyName']}")
        entries.append(
            {
                "driver_id": f"{person_id}_{season}",
                "person_id": person_id,
                "code": driver["code"],
                "car_number": int(result["number"]),
                "given_name": driver["givenName"],
                "family_name": driver["familyName"],
                "team_id": f"{constructor['constructorId']}_{season}",
                "team_name": constructor["name"],
                "grid_position": int(result["grid"]),
                "external_ids": {
                    "jolpica_driver_id": driver["driverId"],
                    "jolpica_constructor_id": constructor["constructorId"],
                },
            }
        )
    # Grid order; pit-lane starters (grid 0) go last, ties broken by car number.
    entries.sort(key=lambda e: (e["grid_position"] == 0, e["grid_position"], e["car_number"]))

    return RaceRoster.model_validate(
        {
            "schema_version": 1,
            "event_id": event_id,
            "season": season,
            "round": int(race["round"]),
            "circuit_id": race["Circuit"]["circuitId"],
            "race_date": race["date"],
            "source_type": SourceType.HISTORICAL_VERIFIED,
            "source": {
                "name": "jolpica-f1",
                "url": source_url,
                "retrieved_utc": retrieved_utc,
                "license_review_status": "research_only",
            },
            "entries": entries,
        }
    )
