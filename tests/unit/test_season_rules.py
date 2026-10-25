"""H1 (engine v0.16.0, D-053): sporting rules per season 2010-2025 and what the engine and the
scoring do with them."""

import pytest

from f1ar.domain.driver_profiles import ProfileSet
from f1ar.domain.identities import RaceRoster
from f1ar.domain.race_state import RacePhase, RaceSetup, WorldState, WorldStatus
from f1ar.domain.season_rules import SEASON_2024, load_catalogue, rules_for
from f1ar.simulation import lightweight_engine as engine
from f1ar.simulation.incidents import IncidentKind, debris_neutralisation, neutralisation_after
from f1ar.simulation.scenarios import build_setup
from f1ar.training.objectives import race_results
from tests.conftest import CONFIGS


def _field(roster: RaceRoster, profiles: ProfileSet, season: int, **rc: float) -> RaceSetup:
    setup = build_setup(roster, profiles, roster.driver_ids(), rules=rules_for(CONFIGS, season))
    return setup.model_copy(update={"race_control": setup.race_control.model_copy(update=rc)})


def test_catalogue_covers_2000_2025_with_the_documented_rules() -> None:
    every = load_catalogue(CONFIGS)
    assert sorted(every) == list(range(2000, 2026))
    rules = {s: r for s, r in every.items() if s >= 2010}
    assert all(r.points == (25, 18, 15, 12, 10, 8, 6, 4, 2, 1) for r in rules.values())
    assert all(not r.refuelling and r.min_distinct_dry_compounds == 2 for r in rules.values())
    assert [s for s, r in rules.items() if not r.vsc] == [2010, 2011, 2012, 2013, 2014]
    assert [s for s, r in rules.items() if not r.drs] == [2010]
    # H1b: the refuelling era
    old = {s: r for s, r in every.items() if s < 2010}
    assert all(r.refuelling and not r.vsc and not r.drs for r in old.values())
    assert [s for s, r in old.items() if r.points == (10, 6, 4, 3, 2, 1)] == [2000, 2001, 2002]
    assert all(len(r.points) == 8 for s, r in old.items() if s >= 2003)
    assert [s for s, r in old.items() if not r.tyre_changes_allowed] == [2005]
    assert [s for s, r in old.items() if r.min_distinct_dry_compounds == 2] == [2007, 2008, 2009]
    assert [s for s, r in old.items() if r.pit_lane_closed_under_sc] == [2007, 2008]
    assert [r.race_fuel_from_qualifying for s, r in sorted(old.items())] == ["none"] * 3 + [
        "all"
    ] * 3 + ["top10"] * 4
    assert [s for s, r in old.items() if "Michelin" in r.tyre_supplier] == list(range(2001, 2007))
    assert (
        not rules[2010].pit_lane_closed_under_sc and rules[2010].race_fuel_from_qualifying == "none"
    )
    assert [s for s, r in rules.items() if r.fastest_lap_point] == [
        2019,
        2020,
        2021,
        2022,
        2023,
        2024,
    ]
    assert [s for s, r in rules.items() if r.double_points_final_round] == [2014]
    assert rules[2010].tyre_supplier == "Bridgestone" and rules[2011].tyre_supplier == "Pirelli"
    assert rules[2013].top10_start_tyres == "q3_tyres"
    assert rules[2021].top10_start_tyres == "q2_tyres"
    assert rules[2022].top10_start_tyres == "free"
    assert rules[2021].shortened_race_points == "half_below_75pct"
    assert rules[2022].shortened_race_points == "sliding_scale_2022"
    # the built-in default equals the catalogue's 2024 apart from where it came from
    assert rules[2024].model_copy(update={"sources": ()}) == SEASON_2024.model_copy(
        update={"sources": ()}
    )
    with pytest.raises(ValueError, match="no sporting rules for 1999"):
        rules_for(CONFIGS, 1999)


def test_points_with_fastest_lap_and_double_points() -> None:
    rules = load_catalogue(CONFIGS)
    assert rules[2024].points_for(1, fastest_lap=True) == 26
    assert rules[2024].points_for(11, fastest_lap=True) == 0  # outside the top 10: no point
    assert rules[2025].points_for(1, fastest_lap=True) == 25
    assert rules[2014].points_for(1, final_round=True) == 50
    assert rules[2014].points_for(10, final_round=True) == 2
    assert rules[2015].points_for(1, final_round=True) == 25


def test_no_vsc_before_2015_double_yellows_instead(
    monza_roster: RaceRoster, monza_profiles: ProfileSet
) -> None:
    rc = {"failure_sc_probability": 0.3, "failure_vsc_probability": 1.0, "debris_per_lap": 1.0}
    old = _field(monza_roster, monza_profiles, 2014, **rc)
    new = _field(monza_roster, monza_profiles, 2015, **rc)
    driver = old.drivers[0].driver_id
    old_n = [neutralisation_after(old, s, driver, 10, IncidentKind.FAILURE) for s in range(200)]
    new_n = [neutralisation_after(new, s, driver, 10, IncidentKind.FAILURE) for s in range(200)]
    assert all(n is None or n.phase is RacePhase.SC for n in old_n)
    assert any(n is not None and n.phase is RacePhase.VSC for n in new_n)
    # the safety cars are the same draws in both eras: only the VSC cases change
    assert [n is not None for n in old_n] == [
        n is not None and n.phase is RacePhase.SC for n in new_n
    ]
    old_debris = [debris_neutralisation(old, s, 5) for s in range(100)]
    assert all(n is None or n.phase is RacePhase.SC for n in old_debris)


def test_race_director_cannot_deploy_a_vsc_in_2014(
    monza_roster: RaceRoster, monza_profiles: ProfileSet
) -> None:
    setup = _field(monza_roster, monza_profiles, 2014)
    state = engine.start(engine.create_world(setup, seed=1, world_id="w"))
    with pytest.raises(engine.ActionRejectedError, match="no virtual safety car in 2014"):
        engine.deploy_neutralisation(state, RacePhase.VSC)
    assert engine.deploy_neutralisation(state, RacePhase.SC).state.race_control.phase is (
        RacePhase.SC
    )


def _finish(state: WorldState) -> WorldState:
    while state.world_status is WorldStatus.RUNNING:
        state = engine.step(state).state
    return state


def test_fastest_lap_point_and_dry_compound_rule_follow_the_season(
    monza_roster: RaceRoster, monza_profiles: ProfileSet
) -> None:
    rc = {"enabled": False}
    s2024 = _field(monza_roster, monza_profiles, 2024, **rc)
    s2018 = _field(monza_roster, monza_profiles, 2018, **rc)
    assert s2018.pit.min_distinct_dry_compounds == 2
    final = _finish(engine.start(engine.create_world(s2024, seed=3, world_id="a")))
    best = {c.driver_id: c.best_lap_time_s for c in final.cars}
    assert all(b is not None and b > 0 for b in best.values())
    holder = min(best, key=lambda d: best[d] or 0.0)
    with_point = race_results(final)
    without = race_results(final.model_copy(update={"setup": s2018}))
    pos = with_point[holder].position
    assert with_point[holder].points - without[holder].points == (1 if pos <= 10 else 0)
    others = [d for d in with_point if d != holder]
    assert all(with_point[d].points == without[d].points for d in others)


def test_a_pit_lane_starter_starts_behind_the_grid(
    monza_roster: RaceRoster, monza_profiles: ProfileSet
) -> None:
    """H2a (D-054): Jolpica's grid 0 = pit-lane start; it was sorted to the front before."""
    from f1ar.domain.race_state import PIT_LANE_SLOT, start_slot

    assert start_slot(0) == PIT_LANE_SLOT and start_slot(1) == 1
    first = monza_roster.entries[0]
    entries = [
        e.model_copy(update={"grid_position": 0}) if e.driver_id == first.driver_id else e
        for e in monza_roster.entries
    ]
    roster = monza_roster.model_copy(update={"entries": tuple(entries)})
    setup = build_setup(roster, monza_profiles, roster.driver_ids())
    assert setup.drivers[-1].driver_id == first.driver_id
    state = engine.start(engine.create_world(setup, seed=2, world_id="w"))
    assert engine.standings(state)[-1] == first.driver_id
