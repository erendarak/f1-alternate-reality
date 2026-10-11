"""Profile -> transition dynamics (plan §3.6 A, test_profile_changes_transition_dynamics)."""

import statistics

from f1ar.domain.driver_profiles import DriverArchetype
from f1ar.domain.race_state import Compound, PaceMode
from f1ar.simulation.lap_model import (
    degradation_s,
    expected_lap_time_s,
    noise_sd_s,
    sample_lap_time_s,
    wear_per_lap_s,
)
from f1ar.simulation.scenarios import default_lap_model

PARAMS = default_lap_model()
NEUTRAL = DriverArchetype.model_validate(dict.fromkeys(DriverArchetype.model_fields, 0.5))


def _with(**changes: float) -> DriverArchetype:
    return NEUTRAL.model_copy(update=changes)


def _expected(driver: DriverArchetype, **kw: object) -> float:
    args: dict[str, object] = {
        "compound": Compound.MEDIUM,
        "tyre_wear_s": 0.5,
        "pace_mode": PaceMode.BALANCED,
        "laps_remaining": 30,
    } | kw
    return expected_lap_time_s(PARAMS, driver, **args)  # type: ignore[arg-type]


def test_higher_pace_skill_is_faster() -> None:
    assert _expected(_with(race_pace_skill=0.9)) < _expected(_with(race_pace_skill=0.6))


def test_better_tyre_management_degrades_less() -> None:
    kw = {"compound": Compound.SOFT, "tyre_age_laps": 20, "pace_mode": PaceMode.BALANCED}
    careful = degradation_s(PARAMS, _with(tire_management=0.95), **kw)  # type: ignore[arg-type]
    rough = degradation_s(PARAMS, _with(tire_management=0.40), **kw)  # type: ignore[arg-type]
    assert 0 < careful < rough


def test_push_is_faster_now_but_wears_faster() -> None:
    d = NEUTRAL
    assert _expected(d, pace_mode=PaceMode.PUSH) < _expected(d, pace_mode=PaceMode.BALANCED)
    push = wear_per_lap_s(PARAMS, d, compound=Compound.MEDIUM, pace_mode=PaceMode.PUSH)
    conserve = wear_per_lap_s(PARAMS, d, compound=Compound.MEDIUM, pace_mode=PaceMode.CONSERVE)
    assert push > conserve > 0


def test_switching_mode_never_rewrites_past_wear() -> None:
    """Regression for engine v0.1.0: the mode only changes the lap-time offset and FUTURE wear.

    On 40-lap-old tyres, switching to PUSH must make the very next lap faster, because the
    40 laps of wear already accumulated are the same whatever mode comes next.
    """
    worn = degradation_s(
        PARAMS, NEUTRAL, compound=Compound.MEDIUM, tyre_age_laps=40, pace_mode=PaceMode.BALANCED
    )
    assert _expected(NEUTRAL, tyre_wear_s=worn, pace_mode=PaceMode.PUSH) < _expected(
        NEUTRAL, tyre_wear_s=worn, pace_mode=PaceMode.BALANCED
    )
    assert _expected(NEUTRAL, tyre_wear_s=worn, pace_mode=PaceMode.CONSERVE) > _expected(
        NEUTRAL, tyre_wear_s=worn, pace_mode=PaceMode.BALANCED
    )


def test_fuel_burn_makes_later_laps_faster_all_else_equal() -> None:
    assert _expected(NEUTRAL, laps_remaining=5) < _expected(NEUTRAL, laps_remaining=50)


def test_risk_appetite_does_not_change_lap_dynamics() -> None:
    """risk_appetite is a preference for the strategist, not a transition-model skill."""
    assert _expected(_with(risk_appetite=0.95)) == _expected(_with(risk_appetite=0.05))
    assert wear_per_lap_s(
        PARAMS, _with(risk_appetite=0.95), compound=Compound.SOFT, pace_mode=PaceMode.PUSH
    ) == wear_per_lap_s(
        PARAMS, _with(risk_appetite=0.05), compound=Compound.SOFT, pace_mode=PaceMode.PUSH
    )


def test_consistency_controls_lap_time_spread() -> None:
    def spread(consistency: float) -> float:
        driver = _with(consistency=consistency)
        laps = [
            sample_lap_time_s(
                PARAMS,
                driver,
                seed=1,
                driver_id="x",
                lap_number=n,
                compound=Compound.MEDIUM,
                tyre_wear_s=0.0,
                pace_mode=PaceMode.BALANCED,
                laps_remaining=30,
            )
            for n in range(1, 2001)
        ]
        return statistics.pstdev(laps)

    assert noise_sd_s(PARAMS, _with(consistency=0.95)) < noise_sd_s(PARAMS, _with(consistency=0.6))
    assert spread(0.95) < spread(0.60)
