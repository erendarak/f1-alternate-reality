"""Determinism (plan §4.5, §18.4). Save/load/fork tests build on these in steps 5-6."""

import subprocess
import sys

from f1ar.domain.driver_profiles import ProfileSet
from f1ar.domain.identities import RaceRoster
from f1ar.domain.race_state import RaceSetup, WorldState, WorldStatus
from f1ar.simulation import lightweight_engine as engine
from f1ar.simulation.scenarios import build_setup, default_traffic_params
from tests.conftest import HAM, REPO_ROOT, VER, run_ticks, run_to_end

LEC = "charles_leclerc_2024"


def _fresh(setup: RaceSetup, seed: int = 42) -> WorldState:
    return engine.start(engine.create_world(setup, seed=seed, world_id="world_test"))


def _lap_times(state: WorldState, driver_id: str) -> list[float]:
    times = []
    while state.world_status is WorldStatus.RUNNING:
        result = engine.step(state)
        times += [
            float(e.payload["lap_time_s"])
            for e in result.events
            if e.type == "LAP_COMPLETED" and e.driver_id == driver_id
        ]
        state = result.state
    return times


def test_same_seed_same_hash(two_car_setup: RaceSetup) -> None:
    a = run_ticks(_fresh(two_car_setup), 1000)
    b = run_ticks(_fresh(two_car_setup), 1000)
    assert a.canonical_hash() == b.canonical_hash()
    assert run_to_end(a).canonical_hash() == run_to_end(b).canonical_hash()


def test_different_seed_different_race(two_car_setup: RaceSetup) -> None:
    a = run_to_end(_fresh(two_car_setup, seed=42))
    b = run_to_end(_fresh(two_car_setup, seed=43))
    assert a.car(VER).finish_time_s != b.car(VER).finish_time_s


def test_hash_identical_in_a_fresh_process(two_car_setup: RaceSetup) -> None:
    """Same inputs in a brand-new interpreter (new hash seed, new imports) -> same state."""
    code = """
from tests.conftest import CONFIGS, MONZA_2024, VER, HAM, run_ticks
from f1ar.domain.identities import load_roster
from f1ar.domain.driver_profiles import load_profiles
from f1ar.simulation import lightweight_engine as engine
from f1ar.simulation.scenarios import build_setup, default_traffic_params
r = load_roster(CONFIGS / "rosters" / f"{MONZA_2024}.json", expected_event_id=MONZA_2024)
p = load_profiles(CONFIGS / "drivers" / "seed_2024_monza_drivers.yaml",
                  CONFIGS / "drivers" / "archetypes.yaml", r)
w = engine.start(engine.create_world(build_setup(r, p, [VER, HAM]), seed=42, world_id="world_test"))
print(run_ticks(w, 1000).canonical_hash())
"""
    out = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
        encoding="utf-8",
    ).stdout.strip()
    assert out == run_ticks(_fresh(two_car_setup), 1000).canonical_hash()


def test_common_random_numbers_across_scenarios(
    monza_roster: RaceRoster, monza_profiles: ProfileSet, isolated_setup: RaceSetup
) -> None:
    """Adding a third car doesn't change anyone else's luck: draws are keyed, not sequential.

    Traffic off: with traffic, a third car may legitimately hold VER up.
    """
    three = build_setup(
        monza_roster,
        monza_profiles,
        [VER, HAM, LEC],
        traffic=default_traffic_params(enabled=False),
    )
    two_car_laps = _lap_times(_fresh(isolated_setup), VER)
    assert len(two_car_laps) == isolated_setup.track.total_laps
    assert two_car_laps == _lap_times(_fresh(three), VER)


def test_profile_change_changes_outcome_only_for_that_driver(
    isolated_setup: RaceSetup,
) -> None:
    """test_profile_changes_transition_dynamics, end-to-end through the engine (no traffic)."""
    two_car_setup = isolated_setup
    ham = two_car_setup.driver(HAM)
    slower = ham.profile.model_copy(
        update={"archetype": ham.profile.archetype.model_copy(update={"race_pace_skill": 0.80})}
    )
    drivers = tuple(
        d.model_copy(update={"profile": slower}) if d.driver_id == HAM else d
        for d in two_car_setup.drivers
    )
    altered = two_car_setup.model_copy(update={"drivers": drivers})

    base = run_to_end(_fresh(two_car_setup))
    alt = run_to_end(_fresh(altered))
    alt_ham, base_ham = alt.car(HAM).finish_time_s, base.car(HAM).finish_time_s
    assert alt_ham is not None and base_ham is not None
    assert alt_ham > base_ham
    assert alt.car(VER).finish_time_s == base.car(VER).finish_time_s
