from pathlib import Path

import pytest

from f1ar.domain.driver_profiles import ProfileSet, load_profiles
from f1ar.domain.identities import RaceRoster, load_roster
from f1ar.domain.race_state import RaceSetup, WorldState, WorldStatus
from f1ar.simulation import lightweight_engine as engine
from f1ar.simulation.scenarios import build_setup

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS = REPO_ROOT / "configs"
MONZA_2024 = "2024_italian_gp"
VER = "max_verstappen_2024"
HAM = "lewis_hamilton_2024"


def run_ticks(state: WorldState, ticks: int) -> WorldState:
    for _ in range(ticks):
        state = engine.step(state).state
    return state


def run_to_end(state: WorldState) -> WorldState:
    while state.world_status is WorldStatus.RUNNING:
        state = engine.step(state).state
    return state


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


@pytest.fixture(scope="session")
def two_car_setup(monza_roster: RaceRoster, monza_profiles: ProfileSet) -> RaceSetup:
    return build_setup(monza_roster, monza_profiles, [VER, HAM])


@pytest.fixture
def two_car_world(two_car_setup: RaceSetup) -> WorldState:
    """Fresh RUNNING two-car world, seed 42."""
    return engine.start(engine.create_world(two_car_setup, seed=42, world_id="world_test"))
