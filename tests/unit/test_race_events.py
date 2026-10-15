"""W4 (engine v0.11.0, D-042): red flags, debris neutralisations, pit crews and slow stops, team
reliability, time penalties, and the fitted race-events config."""

from pathlib import Path

import pytest

from f1ar.calibration.race_events_fit import EVENTS_FILE, load_events, penalty_kind
from f1ar.domain.driver_profiles import ProfileSet
from f1ar.domain.identities import RaceRoster
from f1ar.domain.race_state import (
    CarStatus,
    Compound,
    RaceEvent,
    RaceSetup,
    WorldState,
    WorldStatus,
)
from f1ar.policy.baseline_strategy import free_change_compound
from f1ar.simulation import lightweight_engine as engine
from f1ar.simulation.incidents import IncidentKind, lap_incident, neutralisation_after
from f1ar.simulation.scenarios import build_setup, setup_from_codes, with_race_events
from tests.conftest import CONFIGS, HAM, MONZA_2024, VER


def _field(roster: RaceRoster, profiles: ProfileSet, **rc: float) -> RaceSetup:
    setup = build_setup(roster, profiles, roster.driver_ids())
    return setup.model_copy(update={"race_control": setup.race_control.model_copy(update=rc)})


def _run(state: WorldState) -> tuple[WorldState, list[RaceEvent]]:
    events: list[RaceEvent] = []
    while state.world_status is WorldStatus.RUNNING:
        res = engine.step(state)
        state = res.state
        events += res.events
    return state, events


def test_a_red_flag_gives_everyone_new_tyres_and_a_safety_car(
    monza_roster: RaceRoster, monza_profiles: ProfileSet
) -> None:
    setup = _field(monza_roster, monza_profiles, red_flag_per_crash=1.0, crash_per_lap=0.02)
    state = engine.start(engine.create_world(setup, seed=4, world_id="w"))
    red_at = None
    while state.world_status is WorldStatus.RUNNING and red_at is None:
        before = state
        res = engine.step(state)
        state = res.state
        if any(e.type == "RED_FLAG" for e in res.events):
            red_at = (before, state, res.events)
    assert red_at is not None
    before, after, events = red_at
    assert after.race_control.phase.value == "SC"
    changed = [e for e in events if e.type == "PIT_STOP" and e.payload["under"] == "RED_FLAG"]
    running = [c for c in after.cars if c.status is CarStatus.RUNNING]
    assert len(changed) == len(running) > 0
    for c in running:
        assert c.tyre_age_laps == 0 and c.tyre_life_used == 0.0 and c.tyre_wear_s == 0.0
        assert c.pit_stops == before.car(c.driver_id).pit_stops  # not counted as a stop
        assert c.compound in c.compounds_used


def test_debris_neutralises_without_a_retirement(
    monza_roster: RaceRoster, monza_profiles: ProfileSet
) -> None:
    setup = _field(
        monza_roster,
        monza_profiles,
        debris_per_lap=1.0,
        crash_per_lap=0.0,
        failure_per_lap=0.0,
        first_lap_crash_multiplier=1.0,
    )
    state, events = _run(engine.start(engine.create_world(setup, seed=1, world_id="w")))
    deployed = [e for e in events if e.type in ("SAFETY_CAR_DEPLOYED", "VSC_DEPLOYED")]
    assert deployed and all(e.driver_id is None for e in deployed)
    assert not any(e.type == "CAR_RETIRED" for e in events)


def test_pit_crews_and_slow_stops(two_car_setup: RaceSetup) -> None:
    fast = two_car_setup.drivers[0].model_copy(update={"pit_crew_offset_s": -0.4})
    slow = two_car_setup.drivers[1].model_copy(update={"pit_crew_offset_s": 0.4})
    setup = two_car_setup.model_copy(update={"drivers": (fast, slow)})
    a = [engine.standstill_s(setup, 1, fast.driver_id, lap) for lap in range(1, 200)]
    b = [engine.standstill_s(setup, 1, slow.driver_id, lap) for lap in range(1, 200)]
    assert sorted(b)[100] - sorted(a)[100] == pytest.approx(0.8, abs=0.25)
    bad = setup.model_copy(
        update={"pit": setup.pit.model_copy(update={"slow_stop_probability": 1.0})}
    )
    assert min(engine.standstill_s(bad, 1, fast.driver_id, lap) for lap in range(1, 50)) >= (
        bad.pit.slow_stop_min_s
    )


def test_team_reliability_scales_failures(two_car_setup: RaceSetup) -> None:
    rc = two_car_setup.race_control.model_copy(
        update={"failure_per_lap": 0.5, "crash_per_lap": 0.0, "mistake_per_lap": 0.0}
    )
    never = two_car_setup.drivers[0].model_copy(update={"failure_multiplier": 0.0})
    setup = two_car_setup.model_copy(
        update={"race_control": rc, "drivers": (never, two_car_setup.drivers[1])}
    )
    car = engine.create_world(setup, seed=1, world_id="w").car(never.driver_id)
    assert all(lap_incident(setup, s, car, 5) is None for s in range(1, 60))
    other = engine.create_world(setup, seed=1, world_id="w").car(two_car_setup.drivers[1].driver_id)
    kinds = [lap_incident(setup, s, other, 5) for s in range(1, 60)]
    assert any(k is not None and k.kind is IncidentKind.FAILURE for k in kinds)


def test_red_flag_needs_a_crash(two_car_setup: RaceSetup) -> None:
    rc = two_car_setup.race_control.model_copy(update={"red_flag_per_crash": 1.0})
    setup = two_car_setup.model_copy(update={"race_control": rc})
    crash = neutralisation_after(setup, 1, VER, 3, IncidentKind.CRASH)
    failure = neutralisation_after(setup, 1, VER, 3, IncidentKind.FAILURE)
    assert crash is not None and crash.red_flag
    assert failure is None or not failure.red_flag


def test_time_penalties_count_in_the_result(
    monza_roster: RaceRoster, monza_profiles: ProfileSet
) -> None:
    setup = build_setup(monza_roster, monza_profiles, [VER, HAM])
    rc = setup.race_control.model_copy(update={"enabled": True, "penalty_per_lap": 0.5})
    setup = setup.model_copy(update={"race_control": rc})
    state, events = _run(engine.start(engine.create_world(setup, seed=3, world_id="w")))
    pens = [e for e in events if e.type == "TIME_PENALTY"]
    assert pens
    for c in state.cars:
        if c.status is CarStatus.FINISHED:
            mine = sum(float(e.payload["seconds"]) for e in pens if e.driver_id == c.driver_id)
            assert c.penalty_s == pytest.approx(mine)
            done = next(
                e for e in events if e.type == "CAR_FINISHED" and e.driver_id == c.driver_id
            )
            assert float(done.payload["penalty_s"]) == pytest.approx(mine)


def test_free_change_compound(two_car_setup: RaceSetup) -> None:
    car = engine.create_world(two_car_setup, seed=1, world_id="w").car(VER)
    assert free_change_compound(two_car_setup, car, 0.0) in (
        Compound.SOFT,
        Compound.MEDIUM,
        Compound.HARD,
    )
    assert free_change_compound(two_car_setup, car, 1.0) in (Compound.INTERMEDIATE, Compound.WET)


def test_penalty_messages_are_classified() -> None:
    m = "FIA STEWARDS: 10 SECOND TIME PENALTY FOR CAR 20 (MAG) - CAUSING A COLLISION"
    assert penalty_kind(m) == "collision"
    assert penalty_kind("FIA STEWARDS: 5 SECOND TIME PENALTY FOR CAR 2 - FALSE START") == (
        "false_start"
    )
    assert penalty_kind("FIA STEWARDS: 5 SECOND TIME PENALTY FOR CAR 2 - TRACK LIMITS") == "other"
    assert penalty_kind("FIA STEWARDS: NO FURTHER INVESTIGATION") is None


def test_fitted_race_events_are_applied_by_team(
    monza_roster: RaceRoster, monza_profiles: ProfileSet, tmp_path: Path
) -> None:
    cfg = load_events(CONFIGS)
    assert cfg is not None, f"{EVENTS_FILE} missing"
    setup = with_race_events(build_setup(monza_roster, monza_profiles, [VER, HAM]), CONFIGS)
    assert "race_events" in setup.inputs and "estimated_from_data" in setup.inputs["race_events"]
    ver = setup.driver(VER)
    assert ver.pit_crew_offset_s == pytest.approx(cfg["crews"][ver.team_id]["pit_crew_offset_s"])
    assert setup.race_control.red_flag_per_crash == pytest.approx(
        cfg["fitted"]["red_flag_per_crash"]
    )
    assert setup.pit.slow_stop_probability > 0
    missing = with_race_events(setup, tmp_path)
    assert missing.inputs["race_events"].startswith("editorial_assumption")
    real = setup_from_codes(monza_roster, monza_profiles, ["VER", "HAM"])
    assert "race_events" not in real.inputs  # synthetic setups keep the old model
    assert MONZA_2024
