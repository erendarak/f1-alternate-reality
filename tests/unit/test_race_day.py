"""H3 (engine v0.19.0, D-058): race-day priors from history only, hidden truth vs knowledge."""

import math

from f1ar.calibration.race_day_prior import PriorParams, RaceRecord, prior_for, surprise_size
from f1ar.domain.provenance import SourceType
from f1ar.domain.race_state import Compound, RaceSetup, TyreKnowledge
from f1ar.policy import baseline_strategy as bs
from f1ar.simulation import lightweight_engine as engine
from f1ar.simulation.race_day import drawn_truth, scaled, with_oracle
from tests.conftest import HAM, VER, run_to_end
from tests.unit.test_tyre_life import _with_life

LIFE = {"SOFT": 15.0, "MEDIUM": 20.0, "HARD": 25.0}


def _record(season: int, rnd: int, circuit: str, scale: float, wear: float = 1.0) -> RaceRecord:
    return RaceRecord(
        event_id=f"{season}_{circuit}",
        season=season,
        round=rnd,
        circuit_id=circuit,
        life_raw={c: v * scale for c, v in LIFE.items()},
        measured={c: True for c in LIFE},
        wear=wear,
        split="dry_train",
    )


def _history() -> list[RaceRecord]:
    return [
        _record(s, r, circuit, scale)
        for s in (2010, 2011, 2012)
        for r, (circuit, scale) in enumerate((("monza", 1.4), ("monaco", 0.8), ("spa", 1.0)), 1)
    ]


def test_prior_uses_only_races_before_the_target() -> None:
    history = _history()
    target = _record(2013, 1, "monza", 3.0)
    later = [_record(2013, 2, "monaco", 9.0), _record(2014, 1, "monza", 0.1)]
    p = PriorParams()
    a = prior_for(target, history, p)
    b = prior_for(target, [*history, target, *later], p)
    assert a is not None and b is not None
    assert a.values == b.values  # its own value and later races don't move it
    assert a.track_years == 3 and a.season_races == 0
    assert prior_for(history[0], history, p) is None  # nothing before the first race


def test_prior_learns_the_track_effect() -> None:
    p = PriorParams()
    fast = prior_for(_record(2013, 1, "monza", 1.0), _history(), p)
    slow = prior_for(_record(2013, 1, "monaco", 1.0), _history(), p)
    new = prior_for(_record(2013, 1, "baku", 1.0), _history(), p)
    assert fast is not None and slow is not None and new is not None
    assert fast.values["MEDIUM"] > new.values["MEDIUM"] > slow.values["MEDIUM"]
    assert new.track_years == 0


def test_surprise_size_is_zero_for_a_perfect_prior() -> None:
    from f1ar.calibration.race_day_prior import Prior, Residual

    r = _record(2019, 1, "monza", 1.0)
    exact = Prior({q: r.value(q) for q in ("SOFT", "MEDIUM", "HARD", "WEAR")}, 1, 0)
    s = surprise_size([Residual(r, exact)] * 3)
    assert s["life_shared_sd"] == 0.0 and s["wear_sd"] == 0.0 and s["life_compound_sd"] == 0.0


def _hidden(setup: RaceSetup, life: float, wear: float) -> RaceSetup:
    """Teams believe the life is x ``life`` and the wear x ``wear`` of the truth."""
    seen = scaled(setup.lap_model, {c: life for c in LIFE}, wear)
    k = TyreKnowledge(
        source_type=SourceType.EDITORIAL_ASSUMPTION, note="test", lap_model=seen, wear_scale=wear
    )
    return setup.model_copy(update={"knowledge": k})


def test_team_view_reads_the_tyres_through_the_prior(two_car_setup: RaceSetup) -> None:
    setup = _hidden(_with_life(two_car_setup), life=2.0, wear=0.5)
    car = engine.create_world(setup, seed=1, world_id="w").car(VER)
    car = car.model_copy(
        update={"compound": Compound.MEDIUM, "tyre_life_used": 0.6, "tyre_wear_s": 1.0}
    )
    seen_setup, seen = bs.team_view(setup, car)
    assert math.isclose(seen.tyre_life_used, 0.3)  # twice the life: half of it used
    assert math.isclose(seen.tyre_wear_s, 0.5)
    assert seen_setup.lap_model is setup.knowledge.lap_model  # type: ignore[union-attr]
    oracle = with_oracle(setup, (VER,))
    assert bs.team_view(oracle, car) == (oracle, car)


def test_knowledge_changes_plans_and_an_all_oracle_race_is_the_truth_race(
    two_car_setup: RaceSetup,
) -> None:
    truth = _with_life(two_car_setup)
    hidden = _hidden(truth, life=2.5, wear=0.3)  # teams think the tyres last much longer
    plain = run_to_end(engine.start(engine.create_world(truth, seed=3, world_id="w")))
    oracle = with_oracle(hidden, (VER, HAM))
    same = run_to_end(engine.start(engine.create_world(oracle, seed=3, world_id="w")))
    assert [c.finish_time_s for c in same.cars] == [c.finish_time_s for c in plain.cars]
    start = engine.create_world(hidden, seed=3, world_id="w")
    told = engine.create_world(truth, seed=3, world_id="w")
    # believing in long-lasting tyres, the team picks the softer tyre for its second stint
    assert start.car(VER).planned_stops != told.car(VER).planned_stops
    prior = run_to_end(engine.start(start))
    assert [c.finish_time_s for c in prior.cars] != [c.finish_time_s for c in plain.cars]


def test_drawn_truth_is_keyed_and_scaled_by_the_surprise() -> None:
    entry = {"season": 2019, "prior": {"life_raw": dict(LIFE), "wear_multiplier": 1.2}}
    s = {
        "life_shared_sd": 0.2,
        "life_compound_sd": 0.1,
        "wear_sd": 0.5,
        "life_wear_correlation": -0.2,
    }
    a = drawn_truth(entry, s, "2019_x", 1)
    assert a == drawn_truth(entry, s, "2019_x", 1)
    assert a != drawn_truth(entry, s, "2019_x", 2)
    none = {**s, "life_shared_sd": 0.0, "life_compound_sd": 0.0, "wear_sd": 0.0}
    assert drawn_truth(entry, none, "2019_x", 1) == (LIFE, 1.2)
