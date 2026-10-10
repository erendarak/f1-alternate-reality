from pathlib import Path

import pytest

from f1ar.domain.driver_profiles import ProfileSet, load_profiles
from f1ar.domain.identities import RaceRoster, load_roster

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS = REPO_ROOT / "configs"
MONZA_2024 = "2024_italian_gp"


@pytest.fixture(scope="session")
def monza_roster() -> RaceRoster:
    return load_roster(CONFIGS / "rosters" / f"{MONZA_2024}.json", expected_event_id=MONZA_2024)


@pytest.fixture(scope="session")
def monza_profiles(monza_roster: RaceRoster) -> ProfileSet:
    return load_profiles(
        CONFIGS / "drivers" / "seed_2024_monza_drivers.yaml",
        CONFIGS / "drivers" / "archetypes.yaml",
        monza_roster,
    )
