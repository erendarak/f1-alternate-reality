"""A1b (engine v0.21.0, D-062): rule teams that undercut and cover; per-team practice."""

from f1ar.domain.provenance import SourceType
from f1ar.domain.race_state import Compound, PlannedStop, RaceSetup, RivalRules, WorldState
from f1ar.policy.rival_rules import gap_ahead_s, rival_stop
from f1ar.simulation import lightweight_engine as engine
from tests.conftest import HAM, VER

RULES = RivalRules(source_type=SourceType.EDITORIAL_ASSUMPTION, note="test")


def _state(setup: RaceSetup, *, ham_cross: float, ham_pitted: bool = False) -> WorldState:
    """VER about to complete lap 11 (started it at 1000 s); HAM's last crossing at ``ham_cross``."""
    s = engine.create_world(setup.model_copy(update={"rival_rules": RULES}), seed=1, world_id="w")
    ver = s.car(VER).model_copy(
        update={
            "laps_completed": 10,
            "lap_started_at_s": 1000.0,
            "planned_stops": (PlannedStop(lap=13, compound=Compound.HARD),),
        }
    )
    ham_laps = 11 if not ham_pitted else 10
    ham = s.car(HAM).model_copy(
        update={
            "laps_completed": ham_laps,
            "lap_started_at_s": ham_cross,
            "pit_hold_s": 25.0 if ham_pitted else 0.0,
            "tyre_age_laps": 0 if ham_pitted else 5,
            "pit_stops": 1 if ham_pitted else 0,
        }
    )
    cars = tuple(ver if c.driver_id == VER else ham for c in s.cars)
    return s.model_copy(update={"cars": cars})


def test_gap_and_undercut_after_two_stuck_laps(two_car_setup: RaceSetup) -> None:
    s = _state(two_car_setup, ham_cross=1089.4)
    ver = s.car(VER)
    assert abs((gap_ahead_s(s, ver, 1090.0) or 0.0) - 0.6) < 1e-9
    stop, stuck, _ = rival_stop(s, ver, 1090.0)
    assert stop is None and stuck == 1  # first lap stuck: not yet
    stop, stuck, why = rival_stop(s, ver.model_copy(update={"stuck_laps": 1}), 1090.0)
    assert stop is Compound.HARD and "undercut" in why
    far = _state(two_car_setup, ham_cross=1085.0)
    assert rival_stop(far, far.car(VER).model_copy(update={"stuck_laps": 5}), 1090.0)[0] is None


def test_cover_a_rival_that_just_pitted_behind(two_car_setup: RaceSetup) -> None:
    s = _state(two_car_setup, ham_cross=1001.5, ham_pitted=True)  # 1.5 s behind at the line
    stop, _, why = rival_stop(s, s.car(VER), 1090.0)
    assert stop is Compound.HARD and why.startswith("cover")
    late = s.model_copy(
        update={
            "cars": tuple(
                c.model_copy(
                    update={"planned_stops": (PlannedStop(lap=20, compound=Compound.HARD),)}
                )
                if c.driver_id == VER
                else c
                for c in s.cars
            )
        }
    )
    assert rival_stop(late, late.car(VER), 1090.0)[0] is None  # stop too far away to cover
    assert rival_stop(s.model_copy(update={"setup": two_car_setup}), s.car(VER), 1090.0)[0] is None
