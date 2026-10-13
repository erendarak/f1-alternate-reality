"""Pit stops, tyre choice, baseline strategy and the chequered-flag rule (engine v0.4.0)."""

from itertools import product
from statistics import median

import pytest

from f1ar.domain.driver_profiles import ProfileSet
from f1ar.domain.identities import RaceRoster
from f1ar.domain.race_state import (
    DRY_COMPOUNDS,
    CancelPitStop,
    CarStatus,
    Compound,
    PaceMode,
    RaceEvent,
    RaceSetup,
    RequestPitStop,
    SetStrategyController,
    StrategyController,
    TrackSpec,
    WorldState,
    WorldStatus,
)
from f1ar.policy.baseline_strategy import _stint_cost, plan_stops
from f1ar.policy.observation import build_observation
from f1ar.presentation.track_geometry import TrackGeometry
from f1ar.presentation.view_model import state_view
from f1ar.simulation import lightweight_engine as engine
from f1ar.simulation.lap_model import wear_per_lap_s
from f1ar.simulation.scenarios import build_setup, load_event
from f1ar.simulation.timing import LapTimingSheet, result_gap
from tests.conftest import CONFIGS, HAM, MONZA_2024, VER, run_to_end


def _run_until(state: WorldState, driver_id: str, laps: int) -> WorldState:
    """Step until ``driver_id`` has completed ``laps`` laps (stops right after the crossing)."""
    while state.car(driver_id).laps_completed < laps:
        state = engine.step(state).state
    return state


def _full_race_events(state: WorldState) -> tuple[WorldState, list[str], list[RaceEvent]]:
    events: list[RaceEvent] = []
    while state.world_status is WorldStatus.RUNNING:
        result = engine.step(state)
        state = result.state
        events += result.events
    return state, [e.type for e in events], events


# --- baseline strategy ----------------------------------------------------------------------


def test_every_baseline_car_has_a_legal_plan(two_car_world: WorldState) -> None:
    total = two_car_world.setup.track.total_laps
    for car in two_car_world.cars:
        assert car.controller is StrategyController.BASELINE
        assert 1 <= len(car.planned_stops) <= 2
        laps = [s.lap for s in car.planned_stops]
        assert laps == sorted(laps) and laps[0] >= 1 and laps[-1] < total
        assert len({car.compound, *(s.compound for s in car.planned_stops)}) >= 2


def test_baseline_race_obeys_the_tyre_rule(two_car_world: WorldState) -> None:
    final, types, events = _full_race_events(two_car_world)
    assert "TYRE_RULE_BREACHED" not in types
    for car in final.cars:
        assert car.pit_stops >= 1 and len(set(car.compounds_used)) >= 2
    stops = [e for e in events if e.type == "PIT_STOP"]
    planned = {c.driver_id: [s.lap for s in c.planned_stops] for c in two_car_world.cars}
    for d in (VER, HAM):
        # The first stop happens exactly where the initial plan put it (no noise in the plan).
        assert next(e.payload["lap"] for e in stops if e.driver_id == d) == planned[d][0]


def _brute_force_cost(setup: RaceSetup, *, start: Compound, pace_mode: PaceMode) -> float:
    """Cheapest legal 0/1/2-stop plan by full enumeration (reference for the planner)."""
    driver = setup.drivers[0].profile.archetype
    total = setup.track.total_laps
    pit = setup.pit.lane_loss_s + setup.pit.standstill_median_s

    def stint(c: Compound, n: int) -> float:
        rate = wear_per_lap_s(setup.lap_model, driver, compound=c, pace_mode=pace_mode)
        return _stint_cost(setup.lap_model.compounds[c].offset_s, rate, n)

    best = [stint(start, total)] if setup.pit.min_distinct_dry_compounds <= 1 else []
    for c2 in DRY_COMPOUNDS:
        if len({start, c2}) >= 2:
            best += [stint(start, a) + pit + stint(c2, total - a) for a in range(1, total)]
    for c2, c3 in product(DRY_COMPOUNDS, DRY_COMPOUNDS):
        if len({start, c2, c3}) < 2:
            continue
        for a in range(1, total - 1):
            for b in range(1, total - a):
                best.append(stint(start, a) + 2 * pit + stint(c2, b) + stint(c3, total - a - b))
    return min(best)


@pytest.mark.parametrize("start", DRY_COMPOUNDS)
@pytest.mark.parametrize("pace_mode", [PaceMode.CONSERVE, PaceMode.PUSH])
@pytest.mark.parametrize("pit_loss", [2.0, 22.0])
def test_planner_finds_the_brute_force_optimum(
    two_car_setup: RaceSetup, start: Compound, pace_mode: PaceMode, pit_loss: float
) -> None:
    short = TrackSpec(track_id="t", length_m=5000.0, total_laps=24, source_note="test")
    setup = two_car_setup.model_copy(
        update={
            "track": short,
            "pit": two_car_setup.pit.model_copy(update={"lane_loss_s": pit_loss}),
        }
    )
    driver = setup.drivers[0].profile.archetype
    plan = plan_stops(
        setup,
        driver,
        laps_completed=0,
        compound=start,
        tyre_wear_s=0.0,
        pace_mode=pace_mode,
        compounds_used=(start,),
    )
    # Recompute the plan's cost with the same stint formula.
    pit = setup.pit.lane_loss_s + setup.pit.standstill_median_s
    bounds = [0, *(s.lap for s in plan), short.total_laps]
    compounds = [start, *(s.compound for s in plan)]
    cost = len(plan) * pit
    for c, lo, hi in zip(compounds, bounds, bounds[1:], strict=False):
        rate = wear_per_lap_s(setup.lap_model, driver, compound=c, pace_mode=pace_mode)
        cost += _stint_cost(setup.lap_model.compounds[c].offset_s, rate, hi - lo)
    assert cost == pytest.approx(_brute_force_cost(setup, start=start, pace_mode=pace_mode))


def test_cheap_stops_mean_more_stops(two_car_setup: RaceSetup) -> None:
    def stops(lane_loss: float) -> int:
        setup = two_car_setup.model_copy(
            update={"pit": two_car_setup.pit.model_copy(update={"lane_loss_s": lane_loss})}
        )
        world = engine.create_world(setup, seed=1, world_id="w")
        return len(world.car(HAM).planned_stops)

    assert stops(0.0) == 2 and stops(60.0) == 1


def test_no_stop_needed_once_two_compounds_were_used(two_car_setup: RaceSetup) -> None:
    driver = two_car_setup.drivers[0].profile.archetype
    late = plan_stops(
        two_car_setup,
        driver,
        laps_completed=50,
        compound=Compound.HARD,
        tyre_wear_s=0.5,
        pace_mode=PaceMode.BALANCED,
        compounds_used=(Compound.MEDIUM, Compound.HARD),
    )
    assert late == ()


# --- pit stop mechanics ---------------------------------------------------------------------


def test_requested_stop_happens_at_the_next_crossing(two_car_world: WorldState) -> None:
    state = _run_until(two_car_world, HAM, 2)
    result = engine.apply_action(state, RequestPitStop(driver_id=HAM, compound=Compound.HARD))
    assert [e.type for e in result.events] == ["CONTROLLER_CHANGED", "PIT_REQUESTED"]
    state = result.state
    car = state.car(HAM)
    assert car.controller is StrategyController.USER and car.planned_stops == ()
    assert engine.pending_stop(car) is Compound.HARD

    before = car
    events: list[RaceEvent] = []
    while state.car(HAM).laps_completed < 3:
        r = engine.step(state)
        state, events = r.state, events + list(r.events)
    (stop,) = [e for e in events if e.type == "PIT_STOP"]
    assert stop.payload["lap"] == 3 and stop.payload["to_compound"] == "HARD"
    after = state.car(HAM)
    assert after.compound is Compound.HARD and after.tyre_age_laps == 0
    assert after.tyre_wear_s == 0.0 and after.pit_stops == 1 and after.pit_request is None
    assert after.compounds_used == (before.compound, Compound.HARD)
    assert after.pit_hold_s == pytest.approx(stop.payload["pit_loss_s"])
    assert after.pit_hold_s == pytest.approx(
        state.setup.pit.lane_loss_s + float(stop.payload["standstill_s"])
    )
    # The out-lap contains the hold; the car stays parked at the line while it lasts.
    assert after.lap_target_s > after.pit_hold_s + 0.9 * state.setup.lap_model.base_lap_s
    for _ in range(int(after.pit_hold_s / state.setup.tick_seconds) - 1):
        state = engine.step(state).state
        assert state.car(HAM).distance_m == 0.0
    state = run_to_end(state)
    assert state.car(HAM).pit_stops == 1  # a USER car never gets baseline stops


def test_stop_is_slower_by_about_the_pit_loss(two_car_world: WorldState) -> None:
    """Same seed: the out-lap differs from the no-stop lap by hold + fresh-tyre effects."""
    state = _run_until(two_car_world, HAM, 2)
    stay = engine.apply_action(
        state, SetStrategyController(driver_id=HAM, controller=StrategyController.USER)
    )
    box = engine.apply_action(
        stay.state, RequestPitStop(driver_id=HAM, compound=stay.state.car(HAM).compound)
    )
    a, b = _run_until(stay.state, HAM, 3), _run_until(box.state, HAM, 3)
    out_lap_delta = b.car(HAM).lap_target_s - a.car(HAM).lap_target_s
    hold = b.car(HAM).pit_hold_s
    # Fresh tyres of the same compound: only the worn-off wear is gained back.
    assert out_lap_delta == pytest.approx(hold - a.car(HAM).tyre_wear_s, abs=1e-9)


def test_illegal_pit_requests_are_rejected(two_car_world: WorldState) -> None:
    total = two_car_world.setup.track.total_laps
    last_lap = _run_until(two_car_world, HAM, total - 1)
    with pytest.raises(engine.ActionRejectedError, match="final lap"):
        engine.apply_action(last_lap, RequestPitStop(driver_id=HAM, compound=Compound.SOFT))


def test_cancel_and_take_over_semantics(two_car_world: WorldState) -> None:
    first_stop = two_car_world.car(HAM).planned_stops[0]
    # Nothing pending yet: cancelling is a no-op and the baseline keeps control.
    noop = engine.apply_action(two_car_world, CancelPitStop(driver_id=HAM))
    assert noop.events == () and noop.state == two_car_world

    in_lap = _run_until(two_car_world, HAM, first_stop.lap - 1)
    assert engine.pending_stop(in_lap.car(HAM)) is first_stop.compound
    # Taking over keeps the stop the baseline had called for this lap...
    taken = engine.apply_action(
        in_lap, SetStrategyController(driver_id=HAM, controller=StrategyController.USER)
    ).state
    assert taken.car(HAM).pit_request is first_stop.compound
    # ...and cancelling it keeps the car out.
    cancelled = engine.apply_action(taken, CancelPitStop(driver_id=HAM))
    assert [e.type for e in cancelled.events] == ["PIT_CANCELLED"]
    after = _run_until(cancelled.state, HAM, first_stop.lap)
    assert after.car(HAM).pit_stops == 0

    # Handing back to the baseline replans from the current state.
    back = engine.apply_action(
        after, SetStrategyController(driver_id=HAM, controller=StrategyController.BASELINE)
    )
    assert back.events[0].type == "CONTROLLER_CHANGED"
    plan = back.state.car(HAM).planned_stops
    assert plan and plan[0].lap > first_stop.lap
    final = run_to_end(back.state)
    assert final.car(HAM).pit_stops >= 1


def test_user_car_without_stops_breaches_the_tyre_rule(two_car_world: WorldState) -> None:
    state = engine.apply_action(
        two_car_world, SetStrategyController(driver_id=HAM, controller=StrategyController.USER)
    ).state
    final, types, events = _full_race_events(state)
    breaches = [e for e in events if e.type == "TYRE_RULE_BREACHED"]
    assert [e.driver_id for e in breaches] == [HAM]
    assert final.car(HAM).pit_stops == 0 and final.car(VER).pit_stops >= 1


def test_standstill_distribution() -> None:
    roster, profiles = load_event(CONFIGS, MONZA_2024)
    setup = build_setup(roster, profiles, [VER])
    draws = [engine.standstill_s(setup, 7, VER, lap) for lap in range(1, 2001)]
    assert median(draws) == pytest.approx(setup.pit.standstill_median_s, rel=0.05)
    assert min(draws) > 0.0 and max(draws) <= setup.pit.standstill_max_s
    assert sum(d > 4.0 for d in draws) / len(draws) < 0.05  # slow stops are rare
    assert engine.standstill_s(setup, 7, VER, 5) == draws[4]  # keyed, so repeatable


def test_pit_state_is_in_the_observation(two_car_world: WorldState) -> None:
    state = engine.apply_action(
        two_car_world, RequestPitStop(driver_id=HAM, compound=Compound.HARD)
    ).state
    f = build_observation(state, HAM).features
    assert f["car.boxing_this_lap"] == 1.0 and f["car.pit_stops"] == 0.0
    assert f["car.distinct_compounds_used"] == 1.0
    assert build_observation(state, VER).features["car.boxing_this_lap"] == 0.0


# --- chequered flag / lapped cars -----------------------------------------------------------


@pytest.fixture
def slow_backmarker(monza_roster: RaceRoster, monza_profiles: ProfileSet) -> WorldState:
    """VER's team given a 10 s/lap slower car: he ends up several laps down."""
    setup = build_setup(
        monza_roster,
        monza_profiles,
        [VER, HAM],
        car_offsets_s={monza_roster.entry(VER).team_id: 10.0},
    )
    return engine.start(engine.create_world(setup, seed=5, world_id="w"))


def test_lapped_car_finishes_at_the_flag(slow_backmarker: WorldState) -> None:
    final, types, events = _full_race_events(slow_backmarker)
    total = final.setup.track.total_laps
    flag = final.chequered_flag_s
    assert flag is not None and types.count("CHEQUERED_FLAG") == 1
    (flag_event,) = [e for e in events if e.type == "CHEQUERED_FLAG"]
    assert flag_event.driver_id == HAM and flag_event.payload["flag_time_s"] == flag
    ham, ver = final.car(HAM), final.car(VER)
    assert ham.laps_completed == total and ham.finish_time_s == flag
    assert ver.status is CarStatus.FINISHED and ver.laps_completed < total - 1
    assert ver.finish_time_s is not None and flag <= ver.finish_time_s < flag + 100.0
    # Exactly one crossing after the flag: the previous one was before it.
    ver_crossings = [
        float(e.payload["crossing_time_s"])
        for e in events
        if e.type == "LAP_COMPLETED" and e.driver_id == VER
    ]
    assert ver_crossings[-2] < flag <= ver_crossings[-1]
    assert engine.standings(final) == (HAM, VER)


def test_standings_after_the_flag_put_lead_lap_cars_first(slow_backmarker: WorldState) -> None:
    state = slow_backmarker
    while state.chequered_flag_s is None:
        state = engine.step(state).state
    # HAM has finished, VER (laps down) is still running: HAM is ahead regardless of distance.
    assert state.car(VER).status is CarStatus.RUNNING
    assert engine.standings(state) == (HAM, VER)
    with pytest.raises(engine.ActionRejectedError, match="chequered flag"):
        engine.apply_action(state, RequestPitStop(driver_id=VER, compound=Compound.SOFT))


def test_timing_sheet_and_results_handle_lapped_cars(slow_backmarker: WorldState) -> None:
    code_of = {d.driver_id: d.code for d in slow_backmarker.setup.drivers}
    total = slow_backmarker.setup.track.total_laps
    sheet = LapTimingSheet(code_of, total)
    rows: list[str] = []
    state = slow_backmarker
    while state.world_status is WorldStatus.RUNNING:
        result = engine.step(state)
        state = result.state
        rows += sheet.feed(result.events)
    # One row per lap including the last, which only the leader completed.
    assert [int(r.split()[1].split("/")[0]) for r in rows] == list(range(1, total + 1))
    assert rows[-1].split()[2:] == ["HAM", rows[-1].split()[3]]  # only HAM on lap 53
    stop_lap = next(
        lap for lap in range(1, total + 1) if any("P" in c for c in rows[lap - 1].split()[2:])
    )
    assert stop_lap >= 1
    laps_down = state.car(HAM).laps_completed - state.car(VER).laps_completed
    assert result_gap(state, VER, HAM) == f"+{laps_down} laps"


def test_state_view_shows_pit_state(two_car_world: WorldState) -> None:
    geometry = TrackGeometry.from_points([(0.0, 0.0), (250.0, 0.0), (250.0, 250.0), (0.0, 250.0)])
    stop = two_car_world.car(HAM).planned_stops[0]
    in_lap = _run_until(two_car_world, HAM, stop.lap - 1)
    cars = {
        c["driver_id"]: c for c in state_view(in_lap, geometry, team_colors={}, speed=1)["cars"]
    }
    assert cars[HAM]["box"] == stop.compound.value and not cars[HAM]["in_pit"]
    assert cars[HAM]["controller"] == "BASELINE" and cars[HAM]["plan"].startswith(f"L{stop.lap}")
    in_pit = _run_until(in_lap, HAM, stop.lap)
    cars = {
        c["driver_id"]: c for c in state_view(in_pit, geometry, team_colors={}, speed=1)["cars"]
    }
    assert cars[HAM]["in_pit"] and cars[HAM]["box"] is None and cars[HAM]["pit_stops"] == 1
    assert cars[HAM]["laps_down"] == 0


# --- non-linear wear (M4, D-033) -----------------------------------------------------------------


def _curved(setup: RaceSetup, quad: float, laps: int) -> RaceSetup:
    compounds = {
        c: p.model_copy(update={"deg_quad_s_per_lap2": quad})
        for c, p in setup.lap_model.compounds.items()
    }
    return setup.model_copy(
        update={
            "lap_model": setup.lap_model.model_copy(update={"compounds": compounds}),
            "track": TrackSpec(track_id="t", length_m=5000.0, total_laps=laps, source_note="t"),
        }
    )


def _stint_by_laps(
    setup: RaceSetup, c: Compound, mode: PaceMode, n: int, age0: int, w0: float
) -> float:
    """Lap-by-lap cost of n laps on compound c from tyre age age0 with wear w0 (reference)."""
    driver = setup.drivers[0].profile.archetype
    cost, wear = 0.0, w0
    for k in range(n):
        cost += setup.lap_model.compounds[c].offset_s + wear
        wear += wear_per_lap_s(
            setup.lap_model, driver, compound=c, pace_mode=mode, tyre_age_laps=age0 + k
        )
    return cost


@pytest.mark.parametrize("quad", [0.004, -0.0015])
@pytest.mark.parametrize("age0", [0, 7])
def test_planner_finds_the_brute_force_optimum_with_curved_wear(
    two_car_setup: RaceSetup, quad: float, age0: int
) -> None:
    setup = _curved(two_car_setup, quad, 22)
    driver = setup.drivers[0].profile.archetype
    start, mode, w0 = Compound.MEDIUM, PaceMode.BALANCED, 0.35 if age0 else 0.0
    total = setup.track.total_laps
    pit = setup.pit.lane_loss_s + setup.pit.standstill_median_s

    def first(n: int) -> float:
        return _stint_by_laps(setup, start, mode, n, age0, w0)

    def fresh(c: Compound, n: int) -> float:
        return _stint_by_laps(setup, c, mode, n, 0, 0.0)

    options = []
    for c2 in DRY_COMPOUNDS:
        if c2 != start:
            options += [first(a) + pit + fresh(c2, total - a) for a in range(1, total)]
    for c2, c3 in product(DRY_COMPOUNDS, DRY_COMPOUNDS):
        if len({start, c2, c3}) >= 2:
            for a in range(1, total - 1):
                for b in range(1, total - a):
                    options.append(first(a) + 2 * pit + fresh(c2, b) + fresh(c3, total - a - b))
    plan = plan_stops(
        setup,
        driver,
        laps_completed=0,
        compound=start,
        tyre_wear_s=w0,
        pace_mode=mode,
        compounds_used=(start,),
        tyre_age_laps=age0,
    )
    bounds = [0, *(s.lap for s in plan), total]
    compounds = [start, *(s.compound for s in plan)]
    cost = len(plan) * pit + first(bounds[1])
    for c, lo, hi in zip(compounds[1:], bounds[1:], bounds[2:], strict=False):
        cost += fresh(c, hi - lo)
    assert cost == pytest.approx(min(options))


def test_strong_wear_curve_means_more_stops(two_car_setup: RaceSetup) -> None:
    def stops(quad: float) -> int:
        world = engine.create_world(_curved(two_car_setup, quad, 53), seed=1, world_id="w")
        return len(world.car(HAM).planned_stops)

    assert stops(0.0) <= stops(0.01) and stops(0.01) == 2
