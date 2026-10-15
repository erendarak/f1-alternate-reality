"""W1 (engine v0.10.0, D-041): tyre life, management vs push, the cliff, the fresh-tyre peak,
the baseline's pace control and stop margin, and the same-lap fit."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from f1ar.calibration.lap_model_fit import LIFE_FILE, life_lap_model, load_life
from f1ar.calibration.tyre_life_fit import fit_wear, green_laps
from f1ar.domain.race_state import (
    Compound,
    CompoundParams,
    PaceMode,
    PaceModeParams,
    RaceSetup,
    SetPaceMode,
    SetStrategyController,
    StrategyController,
    WorldState,
    WorldStatus,
)
from f1ar.policy import baseline_strategy as bs
from f1ar.simulation import lightweight_engine as engine
from f1ar.simulation.lap_model import cliff_s, life_use_per_lap, sample_lap_time_s
from tests.conftest import CONFIGS, HAM, VER


def _with_life(setup: RaceSetup, life: float = 20.0, cliff: float = 0.5) -> RaceSetup:
    lm = setup.lap_model
    compounds = dict(lm.compounds)
    for c, mult in ((Compound.SOFT, 0.6), (Compound.MEDIUM, 1.0), (Compound.HARD, 1.3)):
        compounds[c] = lm.compounds[c].model_copy(update={"life_laps": life * mult})
    modes = {
        PaceMode.CONSERVE: PaceModeParams(offset_s=0.35, wear_multiplier=0.75),
        PaceMode.BALANCED: PaceModeParams(offset_s=0.0, wear_multiplier=1.0),
        PaceMode.PUSH: PaceModeParams(offset_s=-0.3, wear_multiplier=1.4),
    }
    return setup.model_copy(
        update={
            "lap_model": lm.model_copy(
                update={
                    "compounds": compounds,
                    "pace_modes": modes,
                    "cliff_s_per_lap": cliff,
                    "fresh_tyre_bonus_s": -0.2,
                    "neutralised_life_factor": 0.35,
                }
            )
        }
    )


def _run(state: WorldState) -> WorldState:
    while state.world_status is WorldStatus.RUNNING:
        state = engine.step(state).state
    return state


# --- the lap model -------------------------------------------------------------------------


def test_without_a_life_nothing_changes(isolated_setup: RaceSetup) -> None:
    lm = isolated_setup.lap_model
    driver = isolated_setup.driver(VER).profile.archetype
    assert life_use_per_lap(lm, driver, compound=Compound.MEDIUM, pace_mode=PaceMode.PUSH) == 0
    assert cliff_s(lm, Compound.MEDIUM, 5.0) == 0.0
    assert not bs.has_tyre_life(isolated_setup)
    laps = np.arange(30)
    assert not bs.management_cost(isolated_setup, driver, Compound.MEDIUM, laps, 1.0).any()
    world = engine.create_world(isolated_setup, seed=1, world_id="w")
    car = world.car(VER).model_copy(update={"pace_mode": PaceMode.PUSH})
    assert bs.baseline_pace_mode(isolated_setup, car) is PaceMode.PUSH  # left alone


def test_life_use_cliff_and_fresh_peak(isolated_setup: RaceSetup) -> None:
    setup = _with_life(isolated_setup)
    lm = setup.lap_model
    driver = setup.driver(VER).profile.archetype
    bal = life_use_per_lap(lm, driver, compound=Compound.MEDIUM, pace_mode=PaceMode.BALANCED)
    push = life_use_per_lap(lm, driver, compound=Compound.MEDIUM, pace_mode=PaceMode.PUSH)
    cons = life_use_per_lap(lm, driver, compound=Compound.MEDIUM, pace_mode=PaceMode.CONSERVE)
    assert push == pytest.approx(1.4 * bal) and cons == pytest.approx(0.75 * bal)
    assert cliff_s(lm, Compound.MEDIUM, 0.99) == 0.0
    # 10 % past a 20-lap life = 2 laps of overrun at 0.5 s per lap
    assert cliff_s(lm, Compound.MEDIUM, 1.1) == pytest.approx(0.5 * 0.1 * 20.0)
    kw = {
        "seed": 3,
        "driver_id": VER,
        "lap_number": 12,
        "compound": Compound.MEDIUM,
        "tyre_wear_s": 0.3,
        "pace_mode": PaceMode.BALANCED,
        "laps_remaining": 20,
    }
    plain = sample_lap_time_s(lm, driver, **kw)  # type: ignore[arg-type]
    fresh = sample_lap_time_s(lm, driver, fresh_lap=True, **kw)  # type: ignore[arg-type]
    worn = sample_lap_time_s(lm, driver, tyre_life_used=1.1, **kw)  # type: ignore[arg-type]
    assert fresh - plain == pytest.approx(-0.2) and worn - plain == pytest.approx(1.0)


# --- the planner ---------------------------------------------------------------------------


def test_management_cost_pushes_spare_life_and_manages_long_stints(
    isolated_setup: RaceSetup,
) -> None:
    setup = _with_life(isolated_setup)
    driver = setup.driver(VER).profile.archetype
    u = life_use_per_lap(
        setup.lap_model, driver, compound=Compound.MEDIUM, pace_mode=PaceMode.BALANCED
    )
    life = 1.0 / u
    n = np.arange(0, int(2.5 * life))
    m = bs.management_cost(setup, driver, Compound.MEDIUM, n, 1.0)
    short, exact, longer = int(life * 0.5), int(life), int(life * 1.2)
    assert m[short] < 0  # spare life is spent pushing
    assert m[short] == pytest.approx(-0.3 * short)  # half a life: every lap pushed
    assert abs(m[exact]) < 1.0  # about the life: barely any push or management
    assert m[longer] > 0  # managed
    beyond = int(life / 0.75) + 5  # more than full management can stretch: the cliff
    assert m[beyond] > beyond * 0.35
    f = bs.stint_costs(setup, driver, Compound.MEDIUM, len(n) - 1)
    assert (np.diff(f, 2) >= -1e-9).all()  # convex: the planner's split stays exact


def test_baseline_pace_follows_the_plan(isolated_setup: RaceSetup) -> None:
    setup = _with_life(isolated_setup)
    world = engine.create_world(setup, seed=1, world_id="w")
    car = world.car(VER).model_copy(update={"planned_stops": ()})
    total = setup.track.total_laps
    # a whole race on one set: manage
    assert bs.baseline_pace_mode(setup, car) is PaceMode.CONSERVE
    # a stop planned two laps from now on a fresh set: push
    from f1ar.domain.race_state import PlannedStop

    soon = car.model_copy(update={"planned_stops": (PlannedStop(lap=2, compound=Compound.HARD),)})
    assert bs.baseline_pace_mode(setup, soon) is PaceMode.PUSH
    assert total > 2


def test_stop_margin_means_fewer_or_equal_stops(
    isolated_setup: RaceSetup, monkeypatch: pytest.MonkeyPatch
) -> None:
    setup = _with_life(isolated_setup, life=14.0)
    car = engine.create_world(setup, seed=1, world_id="w").car(VER)
    monkeypatch.setattr(bs, "STOP_MARGIN_S", 0.0)
    free = len(bs.plan_for_car(setup, car))
    monkeypatch.setattr(bs, "STOP_MARGIN_S", 60.0)
    costly = len(bs.plan_for_car(setup, car))
    assert 1 <= costly <= free


# --- the engine ----------------------------------------------------------------------------


def test_engine_tracks_life_resets_it_at_stops_and_manages(isolated_setup: RaceSetup) -> None:
    setup = _with_life(isolated_setup)
    state = engine.start(engine.create_world(setup, seed=2, world_id="w"))
    seen_life, modes = [], set()
    while state.world_status is WorldStatus.RUNNING:
        before = state
        res = engine.step(state)
        state = res.state
        for e in res.events:
            if e.type == "LAP_COMPLETED" and e.driver_id == VER:
                seen_life.append(state.car(VER).tyre_life_used)
                modes.add(state.car(VER).lap_pace_mode)
            if e.type == "PIT_STOP" and e.driver_id == VER:
                assert state.car(VER).tyre_life_used == 0.0
                assert before.car(VER).tyre_life_used > 0.3
    assert max(seen_life) < 1.15  # the baseline never runs deep into the cliff
    assert state.car(VER).pit_stops >= 1 and len(modes) >= 2  # it manages / pushes


def test_user_pace_stops_the_automatic_management(isolated_setup: RaceSetup) -> None:
    setup = _with_life(isolated_setup)
    state = engine.start(engine.create_world(setup, seed=2, world_id="w"))
    state = engine.apply_action(
        state, SetPaceMode(driver_id=VER, pace_mode=PaceMode.PUSH, actor_id="user")
    ).state
    assert state.car(VER).pace_by_user and state.car(VER).controller is StrategyController.BASELINE
    for _ in range(4000):
        state = engine.step(state).state
    assert state.car(VER).pace_mode is PaceMode.PUSH  # never overridden
    state = engine.apply_action(
        state,
        SetStrategyController(
            driver_id=VER, controller=StrategyController.BASELINE, actor_id="user"
        ),
    ).state
    assert not state.car(VER).pace_by_user
    assert state.car(HAM).pace_by_user is False


def test_life_runs_are_deterministic(isolated_setup: RaceSetup) -> None:
    setup = _with_life(isolated_setup)
    a = _run(engine.start(engine.create_world(setup, seed=9, world_id="w")))
    b = _run(engine.start(engine.create_world(setup, seed=9, world_id="w")))
    assert a.model_dump() == b.model_dump()


# --- calibration ---------------------------------------------------------------------------


def test_same_lap_fit_recovers_known_values() -> None:
    """Synthetic race: a per-lap term (fuel + evolution), driver terms, compound offsets and
    wear slopes, a fresh-tyre peak: the regression recovers them."""
    rng = np.random.default_rng(0)
    rows = []
    off = {"SOFT": -0.3, "MEDIUM": 0.0, "HARD": 0.25}
    lin = {"SOFT": 0.08, "MEDIUM": 0.05, "HARD": 0.03}
    lap_term = {lap: 90.0 - 0.06 * lap + rng.normal(0, 0.2) for lap in range(1, 61)}
    for i in range(20):
        plan = [("SOFT", 15), ("HARD", 45)] if i % 3 == 0 else [("MEDIUM", 25), ("HARD", 35)]
        if i % 3 == 1:
            plan = [("MEDIUM", 20), ("SOFT", 18), ("HARD", 22)]
        lap = 1
        for stint, (c, n) in enumerate(plan, start=1):
            for k in range(1, n + 1):
                if lap > 60:
                    break
                t = lap_term[lap] + 0.1 * i + off[c] + lin[c] * (k - 1)
                t += -0.15 if (k == 2 and stint > 1) else 0.0
                rows.append(
                    {
                        "event_id": "x",
                        "driver_code": f"D{i}",
                        "lap_number": lap,
                        "lap_time_s": t + rng.normal(0, 0.05),
                        "compound": c,
                        "tyre_age_start": 0,
                        "stint": stint,
                        "stint_lap": k,
                        "pit_in_lap": k == n,
                        "pit_out_lap": k == 1 and stint > 1,
                        "track_status_start": "1",
                        "gap_ahead_s": 5.0,
                    }
                )
                lap += 1
    fit = fit_wear(green_laps(pd.DataFrame(rows)))
    assert fit.offsets["SOFT"] == pytest.approx(-0.3, abs=0.05)
    assert fit.offsets["HARD"] == pytest.approx(0.25, abs=0.05)
    for c in lin:
        assert fit.lin[c] == pytest.approx(lin[c], abs=0.01)
    assert fit.fresh_bonus_s == pytest.approx(-0.15, abs=0.08)


def test_fitted_life_config_builds_both_model_variants(isolated_setup: RaceSetup) -> None:
    loaded = load_life(CONFIGS)
    assert loaded is not None, f"{LIFE_FILE} missing"
    cfg, sha = loaded
    field = [d.profile.archetype for d in isolated_setup.drivers]
    model = life_lap_model(isolated_setup.lap_model, cfg, sha, "2024_italian_gp", field, 0.05)
    for c in (Compound.SOFT, Compound.MEDIUM, Compound.HARD):
        assert model.params.compounds[c].life_laps is not None
        assert isinstance(model.params.compounds[c], CompoundParams)
    soft, hard = model.params.compounds[Compound.SOFT], model.params.compounds[Compound.HARD]
    assert soft.life_laps is not None and hard.life_laps is not None
    assert soft.life_laps < hard.life_laps
    delta = float(cfg["globals"]["delta_s"])
    assert hard.offset_s - soft.offset_s == pytest.approx(2 * delta)
    seen = model.estimation_params.compounds
    assert seen[Compound.SOFT].offset_s == pytest.approx(
        cfg["season_wear"]["same_lap_offsets"]["SOFT"]
    )
    assert "editorial_assumption" in model.note and "in-sample" in model.note
    assert Path(CONFIGS / LIFE_FILE).exists()
