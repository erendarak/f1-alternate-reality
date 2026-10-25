"""A1 (engine v0.20.0, D-061): the timing screen, the field's in-race estimate, re-planning."""

import numpy as np

from f1ar.domain.race_state import Compound, RaceSetup, TimingObservation, WorldState
from f1ar.policy.race_estimator import estimate, material
from f1ar.simulation import lightweight_engine as engine
from f1ar.simulation.lap_model import cliff_s
from tests.conftest import HAM, VER, run_to_end
from tests.unit.test_race_day import _hidden
from tests.unit.test_tyre_life import _with_life


def _learning(setup: RaceSetup, life: float, wear: float, trend: float = 0.0) -> RaceSetup:
    hidden = _hidden(_with_life(setup), life=life, wear=wear)
    k = hidden.knowledge
    assert k is not None
    lm = hidden.lap_model.model_copy(update={"track_trend_s_per_lap": trend})
    return hidden.model_copy(
        update={"lap_model": lm, "knowledge": k.model_copy(update={"learning": True})}
    )


def _synthetic(
    setup: RaceSetup, beta: float, tau: float, u: float, n_cars: int = 2
) -> tuple[TimingObservation, ...]:
    """Clean laps from known truth: wear factor 1 + beta, trend tau, life factor e^u (MEDIUM)."""
    k = setup.knowledge
    assert k is not None
    prior = k.lap_model
    rng = np.random.default_rng(1)
    obs = []
    ids = [d.driver_id for d in setup.drivers][:n_cars]
    life = prior.compounds[Compound.MEDIUM].life_laps
    assert life is not None
    for i, d in enumerate(ids):
        form = 0.3 * (i - 0.5)
        stop = 18 + 9 * i
        for lap in range(2, 50):
            age = lap - 1 if lap <= stop else lap - stop - 1
            wear = 0.04 * age + 0.0005 * age * age
            used = age / life
            # the truth's life is prior x e^u: its cliff starts at used = e^u
            true_cliff = (
                prior.cliff_s_per_lap * max(0.0, used * np.exp(-u) - 1.0) * life * np.exp(u)
            )
            r = beta * wear + tau * lap + form + true_cliff - cliff_s(prior, Compound.MEDIUM, used)
            obs.append(
                TimingObservation(
                    driver_id=d,
                    lap=lap,
                    compound=Compound.MEDIUM,
                    wear_s=wear,
                    life_used=used,
                    residual_s=r + rng.normal(0.0, 0.1),
                    noise_sd_s=0.1,
                )
            )
    return tuple(obs)


def test_estimate_recovers_wear_trend_and_life(two_car_setup: RaceSetup) -> None:
    setup = _learning(two_car_setup, life=1.0, wear=1.0)
    est = estimate(setup, _synthetic(setup, beta=0.5, tau=0.01, u=0.0))
    assert abs(est.wear_factor - 1.5) < 0.1
    assert abs(est.trend_s_per_lap - 0.01) < 0.005
    short = estimate(setup, _synthetic(setup, beta=0.0, tau=0.0, u=np.log(0.6)))
    assert abs(short.life_factor[Compound.MEDIUM] - 0.6) < 0.1
    # the shared factor carries some of it to the unseen compounds
    assert short.life_factor[Compound.HARD] < 1.0
    assert estimate(setup, ()).life_factor[Compound.SOFT] == 1.0


def test_material_change_rule(two_car_setup: RaceSetup) -> None:
    setup = _learning(two_car_setup, life=1.0, wear=1.0)
    a = estimate(setup, ())
    assert material(None, a)
    assert not material(a, a)
    b = a.model_copy(update={"wear_factor": a.wear_factor * 1.05})
    assert material(a, b)


def _run(setup: RaceSetup, seed: int = 3) -> WorldState:
    return run_to_end(engine.start(engine.create_world(setup, seed=seed, world_id="w")))


def test_learning_field_records_timing_estimates_and_replans(two_car_setup: RaceSetup) -> None:
    setup = _learning(two_car_setup, life=2.5, wear=0.3, trend=0.01)
    end = _run(setup)
    assert len(end.timing) > 20
    k = end.setup.knowledge
    assert k is not None and k.estimate is not None and k.belief_lap_model is not None
    assert k.estimate.observations <= len(end.timing)
    # the teams believed the tyres last 2.5x longer: what they saw pulled the life estimate down
    assert k.estimate.life_factor[Compound.MEDIUM] < 1.0
    prior_only = setup.model_copy(update={"knowledge": k.model_copy(update={"learning": False})})
    assert _run(prior_only).timing == ()
    assert _run(setup) == end  # deterministic, estimate included
    again = WorldState.model_validate(end.model_dump(mode="json"))
    assert again == end  # the timing screen and the estimate survive a save


def test_trend_zero_and_no_learning_change_nothing(two_car_setup: RaceSetup) -> None:
    plain = _with_life(two_car_setup)
    zero = plain.model_copy(
        update={"lap_model": plain.lap_model.model_copy(update={"track_trend_s_per_lap": 0.0})}
    )
    a, b = _run(plain), _run(zero)
    assert [c.finish_time_s for c in a.cars] == [c.finish_time_s for c in b.cars]
    trended = plain.model_copy(
        update={"lap_model": plain.lap_model.model_copy(update={"track_trend_s_per_lap": 0.02})}
    )
    c = _run(trended)
    assert c.car(VER).finish_time_s != a.car(VER).finish_time_s
    assert c.car(HAM).finish_time_s is not None
