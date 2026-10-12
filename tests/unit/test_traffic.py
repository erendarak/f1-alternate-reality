"""Traffic: dirty air, held up, overtakes, blue flags (engine v0.5.0, M2 slice 2)."""

import math

import pytest

from f1ar.domain.driver_profiles import ProfileSet
from f1ar.domain.identities import RaceRoster
from f1ar.domain.race_state import (
    CarState,
    CarStatus,
    Compound,
    RaceSetup,
    RequestPitStop,
    WorldState,
    WorldStatus,
)
from f1ar.simulation import lightweight_engine as engine
from f1ar.simulation.scenarios import build_setup, default_pit_params
from f1ar.simulation.timeline import fork_world
from f1ar.simulation.traffic import PASS_MARGIN_S, Ahead, resolve
from tests.conftest import HAM, VER, run_ticks, run_to_end

LEC = "charles_leclerc_2024"


def _car(two_car_world: WorldState, driver_id: str, **update: object) -> CarState:
    return two_car_world.car(driver_id).model_copy(update=update)


# --- resolve() in isolation --------------------------------------------------------------------
#
# HAM joins the track at t=100 (lap start 100). VER is ahead on the road, having left the line
# at ``ver_start`` with a lap of ``ver_lap`` seconds.


def _resolve(
    world: WorldState, *, ham_lap: float, ver_start: float, ver_lap: float, **ver: object
) -> tuple[float, list[tuple[str, dict[str, float | int | str]]], dict[str, float]]:
    me = _car(world, HAM, lap_target_s=ham_lap, laps_completed=10)
    ahead = Ahead(
        _car(world, VER, lap_target_s=ver_lap, **{"laps_completed": 10, **ver}), ver_start
    )
    out = resolve(world.setup, world.rng.seed, me, 100.0, 100.0, [ahead])
    return out.lap_target_s, out.events, out.carried_losses


def test_no_interaction_when_they_dont_meet(two_car_world: WorldState) -> None:
    lap, events, carried = _resolve(two_car_world, ham_lap=85.0, ver_start=95.0, ver_lap=85.0)
    assert (lap, events, carried) == (85.0, [], {})


def test_dirty_air_only_inside_the_window(two_car_world: WorldState) -> None:
    p = two_car_world.setup.traffic
    close = 100.0 - (p.dirty_air_window_s - 0.1)
    lap, events, _ = _resolve(two_car_world, ham_lap=85.0, ver_start=close, ver_lap=84.0)
    assert lap == pytest.approx(85.0 + p.dirty_air_loss_s) and events == []


def test_small_advantage_means_held_at_the_minimum_gap(two_car_world: WorldState) -> None:
    p = two_car_world.setup.traffic
    # VER left 5 s earlier and is 5.1 s slower: HAM would arrive 0.1 s ahead. Margin to the
    # held position = 0.1 + min_gap, below the threshold only if threshold > that; force it.
    world = two_car_world.model_copy(
        update={
            "setup": two_car_world.setup.model_copy(
                update={"traffic": p.model_copy(update={"overtake_threshold_s": 5.0})}
            )
        }
    )
    lap, events, carried = _resolve(world, ham_lap=85.0, ver_start=95.0, ver_lap=90.1)
    assert events == [] and carried == {}
    assert 100.0 + lap == pytest.approx(95.0 + 90.1 + p.min_gap_s)


def test_big_advantage_is_a_completed_pass(two_car_world: WorldState) -> None:
    p = two_car_world.setup.traffic
    # HAM would arrive 10 s ahead of VER: the pass is (practically) certain.
    lap, events, carried = _resolve(two_car_world, ham_lap=80.0, ver_start=95.0, ver_lap=95.0)
    ((kind, payload),) = events
    assert kind == "OVERTAKE" and payload["passed"] == VER and payload["lap"] == 11
    assert float(payload["probability"]) > 0.999
    assert lap == pytest.approx(80.0 + p.attacker_loss_s)
    assert 100.0 + lap <= 95.0 + 95.0 - PASS_MARGIN_S
    assert carried == {VER: p.defender_loss_s}


def test_marginal_pass_still_finishes_ahead(two_car_world: WorldState) -> None:
    """A pass just above the threshold arrives PASS_MARGIN ahead (the DRS/slipstream stand-in)."""
    p = two_car_world.setup.traffic
    world = two_car_world.model_copy(
        update={
            "setup": two_car_world.setup.model_copy(
                update={"traffic": p.model_copy(update={"overtake_scale_s": 1e-6})}
            )
        }
    )
    # Free arrival 0.05 s after VER: margin = min_gap - 0.05 > threshold -> certain pass.
    lap, events, _ = _resolve(world, ham_lap=90.05, ver_start=95.0, ver_lap=95.0)
    assert [k for k, _ in events] == ["OVERTAKE"]
    assert 100.0 + lap == pytest.approx(95.0 + 95.0 - PASS_MARGIN_S)


def test_lapped_car_yields_under_blue_flags(two_car_world: WorldState) -> None:
    p = two_car_world.setup.traffic
    lap, events, carried = _resolve(
        two_car_world, ham_lap=85.0, ver_start=95.0, ver_lap=95.0, laps_completed=9
    )
    assert [k for k, _ in events] == ["BLUE_FLAG"] and events[0][1]["yielded"] == VER
    assert lap == pytest.approx(85.0 + p.blue_flag_attacker_loss_s)
    assert carried == {VER: p.blue_flag_defender_loss_s}


def test_disabled_traffic_changes_nothing(isolated_world: WorldState) -> None:
    lap, events, carried = _resolve(isolated_world, ham_lap=80.0, ver_start=99.9, ver_lap=95.0)
    assert (lap, events, carried) == (80.0, [], {})


def test_overtake_probability_matches_the_formula(two_car_world: WorldState) -> None:
    p = two_car_world.setup.traffic
    margin = p.overtake_threshold_s + p.overtake_scale_s  # -> P = 1 - 1/e
    passes = 0
    n = 3000
    for lap in range(n):
        me = _car(two_car_world, HAM, lap_target_s=85.0, laps_completed=lap)
        their_end = 100.0 + 85.0 + margin - p.min_gap_s
        ver = _car(two_car_world, VER, lap_target_s=their_end - 50.0, laps_completed=lap)
        out = resolve(two_car_world.setup, 7, me, 100.0, 100.0, [Ahead(ver, 50.0)])
        passes += any(k == "OVERTAKE" for k, _ in out.events)
    assert passes / n == pytest.approx(1 - math.exp(-1), abs=0.03)


# --- in the engine -----------------------------------------------------------------------------


def _no_stop_setup(setup: RaceSetup) -> RaceSetup:
    pit = default_pit_params().model_copy(
        update={"min_distinct_dry_compounds": 0, "lane_loss_s": 500.0}
    )
    return setup.model_copy(update={"pit": pit})


def test_a_faster_car_that_cant_pass_is_stuck_behind(two_car_setup: RaceSetup) -> None:
    """HAM starts ahead in a car 0.5 s/lap slower; overtaking impossible; nobody stops.

    (With only a tiny advantage, smaller than the dirty-air loss, VER couldn't even stay close:
    he'd fall out of the window and catch up again, which is the intended dirty-air effect.)
    """
    no_pass = two_car_setup.traffic.model_copy(update={"overtake_threshold_s": 100.0})
    drivers = tuple(
        d.model_copy(update={"car_pace_offset_s": 0.5}) if d.driver_id == HAM else d
        for d in two_car_setup.drivers
    )
    setup = _no_stop_setup(two_car_setup).model_copy(
        update={"traffic": no_pass, "drivers": drivers}
    )
    assert setup.driver(HAM).grid_position < setup.driver(VER).grid_position

    stuck, crossings = _race_with_crossings(setup)
    free_setup = setup.model_copy(update={"traffic": no_pass.model_copy(update={"enabled": False})})
    free = run_to_end(engine.start(engine.create_world(free_setup, seed=4, world_id="w")))

    assert all(c.pit_stops == 0 for c in stuck.cars)
    assert engine.standings(free)[0] == VER  # on an empty track VER wins
    assert engine.standings(stuck) == (HAM, VER)  # in traffic he never gets by
    # Many laps end with VER exactly min_gap behind HAM: held up, not just slower.
    held = sum(
        1
        for lap, by in crossings.items()
        if by[VER] - by[HAM] == pytest.approx(setup.traffic.min_gap_s, abs=1e-6)
    )
    assert held >= 15


def _race_with_crossings(setup: RaceSetup) -> tuple[WorldState, dict[int, dict[str, float]]]:
    state = engine.start(engine.create_world(setup, seed=4, world_id="w"))
    crossings: dict[int, dict[str, float]] = {}
    while state.world_status is WorldStatus.RUNNING:
        result = engine.step(state)
        state = result.state
        for e in result.events:
            if e.type == "LAP_COMPLETED" and e.driver_id is not None:
                lap = int(e.payload["lap"])
                crossings.setdefault(lap, {})[e.driver_id] = float(e.payload["crossing_time_s"])
    return state, crossings


def _progress(state: WorldState) -> dict[str, float]:
    """Road progress of cars driving on track (not parked in the pit lane, not finished)."""
    length = state.setup.track.length_m
    return {
        c.driver_id: c.laps_completed + c.distance_m / length
        for c in state.cars
        if c.status is CarStatus.RUNNING and c.lap_elapsed_s > c.pit_hold_s
    }


@pytest.mark.parametrize("seed", [7, 11, 42])
def test_road_order_and_overtake_events_agree(
    monza_roster: RaceRoster, monza_profiles: ProfileSet, seed: int
) -> None:
    """Full 20-car race, both directions:

    - whenever one car moves ahead of another on the road (neither in the pit lane), the
      engine decided it with an OVERTAKE or BLUE_FLAG event;
    - every OVERTAKE really happens: the attacker crosses the line ahead of the defender at
      the end of that lap (regression: a pass followed by a hold used to leave the attacker
      level with the car it had "passed").
    """
    teams = sorted({e.team_id for e in monza_roster.entries})
    setup = build_setup(
        monza_roster,
        monza_profiles,
        [e.driver_id for e in monza_roster.entries],
        car_offsets_s={t: 0.3 * i for i, t in enumerate(teams)},  # backmarkers get lapped
    )
    state = engine.start(engine.create_world(setup, seed=seed, world_id="w"))
    decided: set[tuple[str, str]] = set()
    unexplained: list[tuple[int, str, str]] = []
    overtakes: list[tuple[str, str, int]] = []
    crossing: dict[tuple[str, int], float] = {}
    counts = {"OVERTAKE": 0, "BLUE_FLAG": 0}
    while state.world_status is WorldStatus.RUNNING:
        before = _progress(state)
        result = engine.step(state)
        state = result.state
        for e in result.events:
            if e.driver_id is None:
                continue
            if e.type == "LAP_COMPLETED":
                crossing[e.driver_id, int(e.payload["lap"])] = float(e.payload["crossing_time_s"])
            if e.type in counts:
                counts[e.type] += 1
                rival = str(e.payload.get("passed", e.payload.get("yielded")))
                decided.add((e.driver_id, rival))
                if e.type == "OVERTAKE":
                    overtakes.append((e.driver_id, rival, int(e.payload["lap"])))
        after = _progress(state)
        for a in after.keys() & before.keys():
            for b in after.keys() & before.keys():
                if before[a] < before[b] and after[a] > after[b] and (a, b) not in decided:
                    unexplained.append((state.sim_tick, a, b))
    assert counts["OVERTAKE"] > 0 and counts["BLUE_FLAG"] > 0
    assert unexplained == []
    not_ahead = [
        (a, d, lap)
        for a, d, lap in overtakes
        if (a, lap) in crossing and (d, lap) in crossing and not crossing[a, lap] < crossing[d, lap]
    ]
    assert not_ahead == []


def test_traffic_costs_time_and_is_observable(two_car_world: WorldState) -> None:
    final = run_to_end(two_car_world)
    for c in final.cars:
        assert c.traffic_loss_s == 0.0 and c.carried_loss_s == 0.0  # reset at the flag
    mid = run_ticks(two_car_world, 3 * 168)
    from f1ar.policy.observation import build_observation

    f = build_observation(mid, HAM).features
    assert f["car.last_lap_traffic_loss_s"] == mid.car(HAM).last_lap_traffic_loss_s


def test_with_traffic_a_decision_can_change_a_rivals_race(
    monza_roster: RaceRoster, monza_profiles: ProfileSet
) -> None:
    """Same snapshot, same luck; LEC boxing early changes other cars' races through traffic,
    and the counterfactual is still exactly reproducible."""
    setup = build_setup(monza_roster, monza_profiles, [e.driver_id for e in monza_roster.entries])
    at_lap_10 = engine.pause(
        run_ticks(engine.start(engine.create_world(setup, seed=42, world_id="w")), 10 * 168)
    )

    def finish(box: bool) -> WorldState:
        branch = engine.resume(fork_world(at_lap_10, new_timeline_id=f"t_{box}"))
        if box:
            action = RequestPitStop(driver_id=LEC, compound=Compound.HARD)
            branch = engine.apply_action(branch, action).state
        return run_to_end(branch)

    base, boxed, boxed_again = finish(False), finish(True), finish(True)
    changed = [
        c.driver_id
        for c in boxed.cars
        if c.driver_id != LEC and c.finish_time_s != base.car(c.driver_id).finish_time_s
    ]
    assert changed  # interaction: someone else's race moved
    assert boxed.cars == boxed_again.cars  # still deterministic
