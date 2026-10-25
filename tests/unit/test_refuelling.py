"""H1b (engine v0.18.0, D-057): the refuelling era, 2000-2009."""

from pathlib import Path

import pytest

from f1ar.domain.driver_profiles import ProfileSet
from f1ar.domain.identities import RaceRoster
from f1ar.domain.race_state import CarStatus, Compound, RaceSetup, WorldState, WorldStatus
from f1ar.domain.refuelling import FUEL_S_PER_LAP, REFUEL_S_PER_LAP, fuel_on_board
from f1ar.domain.season_rules import rules_for
from f1ar.policy.refuel_strategy import best_plan_refuel
from f1ar.simulation import lightweight_engine as engine
from f1ar.simulation.scenarios import build_setup
from tests.conftest import CONFIGS


def _refuel_setup(roster: RaceRoster, profiles: ProfileSet, season: int) -> RaceSetup:
    setup = build_setup(roster, profiles, roster.driver_ids(), rules=rules_for(CONFIGS, season))
    return setup.model_copy(
        update={
            "pit": setup.pit.model_copy(update={"refuel_s_per_lap": REFUEL_S_PER_LAP}),
            "lap_model": setup.lap_model.model_copy(
                update={"fuel_s_per_lap_remaining": FUEL_S_PER_LAP}
            ),
            "race_control": setup.race_control.model_copy(update={"enabled": False}),
        }
    )


def _run(setup: RaceSetup, seed: int = 3) -> tuple[WorldState, list[dict[str, object]]]:
    state = engine.start(engine.create_world(setup, seed=seed, world_id="w"))
    stops: list[dict[str, object]] = []
    while state.world_status is WorldStatus.RUNNING:
        res = engine.step(state)
        state = res.state
        stops += [dict(e.payload) for e in res.events if e.type == "PIT_STOP"]
        assert all(c.fuel_laps is None or c.fuel_laps >= -1e-9 for c in state.cars)
    return state, stops


def test_the_plan_balances_fuel_weight_against_stops(
    monza_roster: RaceRoster, monza_profiles: ProfileSet
) -> None:
    setup = _refuel_setup(monza_roster, monza_profiles, 2004)
    driver = setup.drivers[0].profile.archetype
    plan = best_plan_refuel(
        setup,
        driver,
        laps_completed=0,
        compound=Compound.MEDIUM,
        compounds_used=(Compound.MEDIUM,),
        fuel_laps=None,
        free_first=True,
    )
    assert plan is not None and 1 <= len(plan.stops) <= 3
    assert plan.first_stint_laps == plan.stops[0].lap
    # mid-race the stint on the current fuel is fixed
    mid = best_plan_refuel(
        setup,
        driver,
        laps_completed=10,
        compound=Compound.MEDIUM,
        compounds_used=(Compound.MEDIUM,),
        fuel_laps=8.0,
        free_first=False,
    )
    assert mid is not None and mid.first_stint_laps == 8 and mid.stops[0].lap == 18
    # lighter fuel effect: fewer stops
    light = setup.model_copy(
        update={"lap_model": setup.lap_model.model_copy(update={"fuel_s_per_lap_remaining": 0.0})}
    )
    few = best_plan_refuel(
        light,
        driver,
        laps_completed=0,
        compound=Compound.MEDIUM,
        compounds_used=(Compound.MEDIUM,),
        fuel_laps=None,
        free_first=True,
    )
    assert few is not None and len(few.stops) <= len(plan.stops)


def test_cars_burn_and_refuel_and_never_run_dry(
    monza_roster: RaceRoster, monza_profiles: ProfileSet
) -> None:
    setup = _refuel_setup(monza_roster, monza_profiles, 2004)
    start = engine.create_world(setup, seed=3, world_id="w")
    assert all(c.fuel_laps is not None and c.fuel_laps >= 1 for c in start.cars)
    final, stops = _run(setup)
    assert stops and all(float(str(s["fuel_laps_added"])) > 0 for s in stops)
    assert all(
        abs(float(str(s["refuel_s"])) - float(str(s["fuel_laps_added"])) * REFUEL_S_PER_LAP) < 0.02
        for s in stops
    )
    finishers = [c for c in final.cars if c.status is CarStatus.FINISHED]
    assert finishers and all(c.pit_stops >= 1 for c in finishers)


def test_2005_stops_are_fuel_only_and_fixed_start_fuel_is_kept(
    monza_roster: RaceRoster, monza_profiles: ProfileSet
) -> None:
    setup = _refuel_setup(monza_roster, monza_profiles, 2005)
    first = setup.drivers[0]
    drivers = tuple(
        d.model_copy(update={"start_fuel_laps": 20}) if d.driver_id == first.driver_id else d
        for d in setup.drivers
    )
    setup = setup.model_copy(update={"drivers": drivers})
    start = engine.create_world(setup, seed=5, world_id="w")
    assert start.car(first.driver_id).fuel_laps == 20.0
    assert start.car(first.driver_id).planned_stops[0].lap == 20
    final, stops = _run(setup, seed=5)
    assert stops and all(s["from_compound"] == s["to_compound"] for s in stops)
    assert all(len(c.compounds_used) == 1 for c in final.cars)
    assert any(s["lap"] == 20 for s in stops)


@pytest.mark.parametrize("season", [2010, 2024])
def test_seasons_without_refuelling_carry_no_fuel_state(
    monza_roster: RaceRoster, monza_profiles: ProfileSet, season: int
) -> None:
    setup = build_setup(
        monza_roster, monza_profiles, monza_roster.driver_ids(), rules=rules_for(CONFIGS, season)
    )
    assert all(c.fuel_laps is None for c in engine.create_world(setup, seed=1, world_id="w").cars)


def test_fuel_on_board_follows_the_stint(tmp_path: Path) -> None:
    assert fuel_on_board(20, 1) == 20 and fuel_on_board(20, 20) == 1 and fuel_on_board(20, 25) == 1
