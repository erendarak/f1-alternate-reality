"""Incidents, retirements, VSC and the safety car (engine v0.6.0, M2 slice 3)."""

import math
from collections import Counter
from dataclasses import dataclass

import pytest

from f1ar.domain.driver_profiles import ProfileSet
from f1ar.domain.identities import RaceRoster
from f1ar.domain.race_state import (
    CarStatus,
    PaceMode,
    RaceEvent,
    RacePhase,
    RaceSetup,
    SetPaceMode,
    WorldState,
    WorldStatus,
)
from f1ar.policy.observation import build_observation
from f1ar.simulation import lightweight_engine as engine
from f1ar.simulation.incidents import IncidentKind, lap_incident, neutralisation_after
from f1ar.simulation.scenarios import build_setup, default_race_control_params
from f1ar.simulation.timing import LapTimingSheet, result_gap
from tests.conftest import HAM, VER, run_to_end

CODES = ("NOR", "PIA", "LEC", "SAI", "HAM", "VER")


@dataclass
class Race:
    states: list[WorldState]
    events: list[RaceEvent]

    @property
    def final(self) -> WorldState:
        return self.states[-1]

    def first(self, type_: str) -> RaceEvent:
        return next(e for e in self.events if e.type == type_)


def _setup(roster: RaceRoster, profiles: ProfileSet, **race_control: object) -> RaceSetup:
    """Six cars; only crashes (no mistakes or failures) unless overridden."""
    params = default_race_control_params().model_copy(
        update={
            "mistake_per_lap": 0.0,
            "failure_per_lap": 0.0,
            "crash_per_lap": 0.004,
            "first_lap_crash_multiplier": 1.0,
            **race_control,
        }
    )
    ids = [e.driver_id for e in roster.entries if e.code in CODES]
    return build_setup(roster, profiles, ids, race_control=params)


def _run(setup: RaceSetup, seed: int) -> Race:
    state = engine.start(engine.create_world(setup, seed=seed, world_id="w"))
    states: list[WorldState] = [state]
    events: list[RaceEvent] = []
    while state.world_status is WorldStatus.RUNNING:
        result = engine.step(state)
        state = result.state
        states.append(state)
        events += result.events
    return Race(states, events)


def _first_race_with(setup: RaceSetup, event_type: str, min_lap: int = 3) -> Race:
    """The first seed whose race has ``event_type`` caused by a retirement on lap >= min_lap."""
    for seed in range(200):
        race = _run(setup, seed)
        retired = [e for e in race.events if e.type == "CAR_RETIRED"]
        if (
            retired
            and int(retired[0].payload["lap"]) >= min_lap
            and any(e.type == event_type for e in race.events)
        ):
            return race
    raise AssertionError(f"no seed produced {event_type}")


# --- the incident model ------------------------------------------------------------------------


def test_incidents_off_means_none(isolated_world: WorldState) -> None:
    car = isolated_world.car(HAM)
    assert all(lap_incident(isolated_world.setup, 1, car, lap) is None for lap in range(1, 54))


def test_pushing_crashes_more_than_conserving(two_car_world: WorldState) -> None:
    setup = two_car_world.setup.model_copy(
        update={
            "race_control": two_car_world.setup.race_control.model_copy(
                update={"crash_per_lap": 0.05, "failure_per_lap": 0.0, "mistake_per_lap": 0.0}
            )
        }
    )

    def crash_rate(mode: PaceMode) -> float:
        car = two_car_world.car(HAM).model_copy(update={"lap_pace_mode": mode})
        hits = sum(
            1
            for lap in range(2, 6002)
            if (i := lap_incident(setup, 3, car, lap)) is not None and i.kind is IncidentKind.CRASH
        )
        return hits / 6000

    p = setup.race_control
    from f1ar.simulation.incidents import skill_factor

    base = p.crash_per_lap * skill_factor(setup, HAM)
    assert crash_rate(PaceMode.BALANCED) == pytest.approx(base, rel=0.15)
    assert crash_rate(PaceMode.PUSH) == pytest.approx(base * p.push_risk_multiplier, rel=0.15)
    assert crash_rate(PaceMode.CONSERVE) == pytest.approx(
        base * p.conserve_risk_multiplier, rel=0.2
    )


def test_neutralisation_choices_and_lengths(two_car_world: WorldState) -> None:
    setup = two_car_world.setup
    p = setup.race_control
    seen: Counter[RacePhase | None] = Counter()
    for lap in range(1, 4001):
        n = neutralisation_after(setup, 9, HAM, lap, IncidentKind.CRASH)
        seen[None if n is None else n.phase] += 1
        if n is not None and n.phase is RacePhase.SC:
            assert p.sc_min_laps <= n.sc_laps <= p.sc_max_laps
        if n is not None and n.phase is RacePhase.VSC:
            assert p.vsc_min_s <= n.vsc_s <= p.vsc_max_s
    sc = p.crash_sc_probability
    assert seen[RacePhase.SC] / 4000 == pytest.approx(sc, abs=0.03)
    assert seen[RacePhase.VSC] / 4000 == pytest.approx((1 - sc) * p.crash_vsc_probability, abs=0.03)


# --- retirements -------------------------------------------------------------------------------


def test_a_crashed_car_retires_where_it_stopped(
    monza_roster: RaceRoster, monza_profiles: ProfileSet
) -> None:
    setup = _setup(
        monza_roster, monza_profiles, crash_sc_probability=0.0, crash_vsc_probability=0.0
    )
    race = _first_race_with(setup, "CAR_RETIRED")
    event = race.first("CAR_RETIRED")
    out = str(event.driver_id)
    assert event.payload["reason"] == "CRASH"
    final = race.final
    car = final.car(out)
    assert car.status is CarStatus.RETIRED and car.retired_reason == "CRASH"
    assert car.distance_m == pytest.approx(float(event.payload["lap_fraction"]) * 5793.0)
    assert car.laps_completed == int(event.payload["laps"])
    assert final.world_status is WorldStatus.COMPLETED
    assert engine.standings(final)[-1] == out  # retired cars are classified last
    winner = engine.standings(final)[0]
    assert result_gap(final, out, winner).startswith("DNF (CRASH, lap ")
    with pytest.raises(engine.ActionRejectedError, match="RETIRED"):
        retired_at = next(s for s in race.states if s.car(out).status is CarStatus.RETIRED)
        engine.apply_action(retired_at, SetPaceMode(driver_id=out, pace_mode=PaceMode.PUSH))
    # The timing sheet doesn't wait for the retired car.
    sheet = LapTimingSheet({d.driver_id: d.code for d in setup.drivers}, setup.track.total_laps)
    rows = sheet.feed(race.events)
    assert len(rows) == setup.track.total_laps


# --- safety car --------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sc_race(monza_roster: RaceRoster, monza_profiles: ProfileSet) -> Race:
    setup = _setup(monza_roster, monza_profiles, crash_sc_probability=1.0)
    return _first_race_with(setup, "SAFETY_CAR_DEPLOYED")


def _sc_window(race: Race) -> list[WorldState]:
    """States from the first tick under SC up to (not including) the restart."""
    return [
        s
        for s in race.states
        if s.race_control.phase is RacePhase.SC and s.race_control.sc_count == 1
    ]


def test_safety_car_deploys_ahead_of_the_leader(sc_race: Race) -> None:
    under = _sc_window(sc_race)
    first = under[0]
    length = first.setup.track.length_m
    road = [
        c for c in first.cars if c.status is CarStatus.RUNNING and c.lap_elapsed_s >= c.pit_hold_s
    ]
    leader = max(road, key=lambda c: (c.laps_completed, c.distance_m))
    sc = first.race_control.sc_distance_m
    assert sc is not None
    deployed = sc_race.first("SAFETY_CAR_DEPLOYED")
    assert first.race_control.cause == deployed.driver_id
    # Placed sc_deploy_ahead_m ahead of the leader one tick earlier, then moved one tick.
    ahead = (sc - leader.distance_m) % length
    assert 0.0 < ahead <= first.setup.race_control.sc_deploy_ahead_m + 1.0


def test_no_overtaking_and_the_field_closes_up_behind_the_safety_car(sc_race: Race) -> None:
    under = _sc_window(sc_race)
    spacing = under[0].setup.race_control.sc_spacing_m
    length = under[0].setup.track.length_m

    def queue(s: WorldState) -> list[tuple[str, float]]:
        """(driver, metres behind the safety car) for cars on the road, front first."""
        sc = s.race_control.sc_distance_m
        assert sc is not None
        cars = [
            (c.driver_id, (sc - c.distance_m) % length)
            for c in s.cars
            if c.status is CarStatus.RUNNING and c.lap_elapsed_s >= c.pit_hold_s
        ]
        return sorted(cars, key=lambda x: x[1])

    for before, after in zip(under, under[1:], strict=False):
        q0, q1 = queue(before), queue(after)
        common = {d for d, _ in q0} & {d for d, _ in q1}
        order0 = [d for d, _ in q0 if d in common]
        order1 = [d for d, _ in q1 if d in common]
        assert order0 == order1, f"order changed under SC at tick {after.sim_tick}"
        # Gaps never shrink below the spacing. Cars that were closer than that when the safety
        # car came out (racing nose to tail), or that just left the pit lane together, are held
        # back until the gap has opened up.
        pos0 = dict(q0)
        for (a, x), (b, y) in zip(q1, q1[1:], strict=False):
            if a in pos0 and b in pos0:
                assert y - x >= min(spacing, pos0[b] - pos0[a]) - 1e-6
    sc_ticks = {s.sim_tick for s in under}
    assert not [e for e in sc_race.events if e.type == "OVERTAKE" and e.tick in sc_ticks]

    # By the time it pulls in, the train is compact: everyone within ~2 spacings of the next.
    gaps = [
        b2 - b1 for (_, b1), (_, b2) in zip(queue(under[-1]), queue(under[-1])[1:], strict=False)
    ]
    assert max(gaps) < 2.5 * spacing


def test_safety_car_laps_then_restart_without_jumps(sc_race: Race) -> None:
    deployed = sc_race.first("SAFETY_CAR_DEPLOYED")
    sc_in = sc_race.first("SAFETY_CAR_IN")
    restart = sc_race.first("RESTART")
    assert sc_in.tick == restart.tick > deployed.tick
    under = _sc_window(sc_race)
    crossings = sum(
        1
        for a, b in zip(under, under[1:], strict=False)
        if (b.race_control.sc_distance_m or 0.0) < (a.race_control.sc_distance_m or 0.0)
    )
    assert crossings + 1 == int(deployed.payload["laps"])  # the last crossing ends it
    i = next(k for k, s in enumerate(sc_race.states) if s.sim_tick == restart.tick)
    at_restart, after = sc_race.states[i], sc_race.states[i + 1]
    assert at_restart.race_control.phase is RacePhase.GREEN
    length = at_restart.setup.track.length_m
    for c in at_restart.cars:
        if c.status is not CarStatus.RUNNING or c.lap_elapsed_s < c.pit_hold_s:
            continue
        n = after.car(c.driver_id)
        moved = (n.distance_m - c.distance_m) % length
        assert 0.0 < moved < 45.0  # one tick of racing speed, no teleport


def test_cheap_stops_under_the_safety_car(sc_race: Race) -> None:
    under = _sc_window(sc_race)
    window = under[0].setup.race_control.neutralised_pit_window_laps
    deployed_state = under[0]
    expected = {
        c.driver_id
        for c in deployed_state.cars
        if c.status is CarStatus.RUNNING
        and c.planned_stops
        and c.planned_stops[0].lap - c.laps_completed <= window + 1
        and c.pit_stops == 0
    }
    stops_under_sc = {
        e.driver_id for e in sc_race.events if e.type == "PIT_STOP" and e.payload["under"] == "SC"
    }
    if expected:
        assert stops_under_sc  # at least one car took the cheap stop
    assert stops_under_sc <= {c.driver_id for c in deployed_state.cars}


def test_observation_shows_the_safety_car(sc_race: Race) -> None:
    under = _sc_window(sc_race)[5]
    running = next(c for c in under.cars if c.status is CarStatus.RUNNING)
    f = build_observation(under, running.driver_id).features
    assert f["race.safety_car"] == 1.0 and f["race.vsc"] == 0.0


# --- VSC ---------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def vsc_race(monza_roster: RaceRoster, monza_profiles: ProfileSet) -> Race:
    setup = _setup(
        monza_roster, monza_profiles, crash_sc_probability=0.0, crash_vsc_probability=1.0
    )
    return _first_race_with(setup, "VSC_DEPLOYED")


def test_vsc_slows_everyone_by_the_same_fraction(vsc_race: Race) -> None:
    k = vsc_race.final.setup.race_control.vsc_speed_fraction
    dt = vsc_race.final.setup.tick_seconds
    length = vsc_race.final.setup.track.length_m
    checked = 0
    for a, b in zip(vsc_race.states, vsc_race.states[1:], strict=False):
        if a.race_control.phase is not RacePhase.VSC:
            continue
        for ca in a.cars:
            cb = b.car(ca.driver_id)
            if (
                ca.status is not CarStatus.RUNNING
                or cb.status is not CarStatus.RUNNING
                or cb.laps_completed != ca.laps_completed
                or ca.lap_elapsed_s < ca.pit_hold_s
            ):
                continue
            racing_speed = length / (ca.lap_target_s - ca.pit_hold_s)
            assert cb.distance_m - ca.distance_m == pytest.approx(k * racing_speed * dt, rel=1e-6)
            checked += 1
    assert checked > 100


def test_vsc_ends_after_its_duration(vsc_race: Race) -> None:
    deployed = vsc_race.first("VSC_DEPLOYED")
    ended = vsc_race.first("VSC_ENDED")
    dt = vsc_race.final.setup.tick_seconds
    lasted = (ended.tick - deployed.tick) * dt
    assert (
        float(deployed.payload["duration_s"])
        <= lasted
        <= float(deployed.payload["duration_s"]) + dt
    )


def test_pit_lane_runs_at_full_speed_under_vsc(two_car_world: WorldState) -> None:
    """A car in the pit lane under VSC loses its hold in sim time, the others crawl."""
    state = engine.step(two_car_world).state
    vsc = state.race_control.model_copy(
        update={"phase": RacePhase.VSC, "vsc_until_s": 1e9, "since_s": state.sim_clock_s}
    )
    ham = state.car(HAM).model_copy(update={"pit_hold_s": 20.0, "lap_elapsed_s": 5.0})
    cars = tuple(ham if c.driver_id == HAM else c for c in state.cars)
    state = state.model_copy(update={"race_control": vsc, "cars": cars})
    after = engine.step(state).state
    dt = state.setup.tick_seconds
    assert after.car(HAM).lap_elapsed_s == pytest.approx(5.0 + dt)  # sim time
    k = state.setup.race_control.vsc_speed_fraction
    assert after.race_clock_s - state.race_clock_s == pytest.approx(k * dt)
    assert after.car(VER).lap_elapsed_s - state.car(VER).lap_elapsed_s == pytest.approx(k * dt)


# --- determinism with race control -------------------------------------------------------------


def test_save_and_reload_under_the_safety_car_continues_identically(sc_race: Race) -> None:
    under = _sc_window(sc_race)[10]
    reloaded = WorldState.model_validate_json(under.model_dump_json())
    assert reloaded == under
    a, b = under, reloaded
    while a.world_status is WorldStatus.RUNNING:
        a, b = engine.step(a).state, engine.step(b).state
    assert a.canonical_hash() == b.canonical_hash() == sc_race.final.canonical_hash()


def test_race_with_incidents_is_reproducible(
    monza_roster: RaceRoster, monza_profiles: ProfileSet
) -> None:
    setup = build_setup(monza_roster, monza_profiles, [e.driver_id for e in monza_roster.entries])

    def race(seed: int) -> tuple[str, list[RaceEvent]]:
        state = engine.start(engine.create_world(setup, seed=seed, world_id="w"))
        events: list[RaceEvent] = []
        while state.world_status is WorldStatus.RUNNING:
            result = engine.step(state)
            state = result.state
            events += result.events
        assert not math.isnan(state.race_clock_s)
        return state.canonical_hash(), events

    assert race(2024) == race(2024)


# --- race director ----------------------------------------------------------------------------


def test_race_director_safety_car(isolated_world: WorldState) -> None:
    """A user-deployed safety car runs the full cycle even with random incidents switched off."""
    state = isolated_world
    while state.car(VER).laps_completed < 5:
        state = engine.step(state).state
    result = engine.deploy_neutralisation(state, RacePhase.SC, sc_laps=2)
    (event,) = result.events
    assert event.type == "SAFETY_CAR_DEPLOYED" and event.driver_id is None
    assert event.payload == {"laps": 2, "sc_number": 1, "actor_id": "USER"}
    state = result.state
    assert state.race_control.phase is RacePhase.SC and state.sim_tick == result.state.sim_tick
    with pytest.raises(engine.ActionRejectedError, match="already out"):
        engine.deploy_neutralisation(state, RacePhase.VSC)
    events: list[RaceEvent] = []
    while state.race_control.phase is RacePhase.SC:
        step = engine.step(state)
        state, events = step.state, events + list(step.events)
    assert [e.type for e in events if e.type in ("SAFETY_CAR_IN", "RESTART")] == [
        "SAFETY_CAR_IN",
        "RESTART",
    ]
    final = run_to_end(state)
    assert final.world_status is WorldStatus.COMPLETED


def test_race_director_vsc_and_rejections(isolated_world: WorldState) -> None:
    with pytest.raises(engine.ActionRejectedError, match="not GREEN"):
        engine.deploy_neutralisation(isolated_world, RacePhase.GREEN)
    vsc = engine.deploy_neutralisation(isolated_world, RacePhase.VSC, vsc_s=30.0).state
    assert vsc.race_control.phase is RacePhase.VSC
    assert vsc.race_control.vsc_until_s == pytest.approx(isolated_world.sim_clock_s + 30.0)
    upgraded = engine.deploy_neutralisation(vsc, RacePhase.SC)  # VSC -> SC is allowed
    assert upgraded.state.race_control.phase is RacePhase.SC
    with pytest.raises(engine.ActionRejectedError, match="race is over"):
        engine.deploy_neutralisation(run_to_end(isolated_world), RacePhase.SC)


def test_runner_deploy_commands(isolated_setup: RaceSetup) -> None:
    from f1ar.simulation.world_runner import Command, CommandType, WorldRunner

    runner = WorldRunner(
        engine.create_world(isolated_setup, seed=1, world_id="w"), clock=lambda: 0.0
    )
    runner.submit(Command(idempotency_key="s", type=CommandType.START))
    runner.pump(0.0)
    vsc = runner.submit(Command(idempotency_key="v", type=CommandType.DEPLOY_VSC, duration_s=45.0))
    runner.pump(0.0)
    assert vsc.result().accepted and runner.state.race_control.phase is RacePhase.VSC
    sc = runner.submit(Command(idempotency_key="c", type=CommandType.DEPLOY_SC, laps=3))
    runner.pump(0.0)
    assert sc.result().accepted and runner.state.race_control.sc_laps_left == 3
    again = runner.submit(Command(idempotency_key="c2", type=CommandType.DEPLOY_SC))
    runner.pump(0.0)
    assert not again.result().accepted and "already out" in (again.result().reason or "")
