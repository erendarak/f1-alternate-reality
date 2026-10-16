"""M5 slice 4 (D-048): racecraft intent ATTACK / NORMAL / DEFEND, its actions and rl_obs_v3."""

import numpy as np
import pytest

from f1ar.domain.race_state import (
    CarState,
    RacecraftIntent,
    RaceSetup,
    SetRacecraftIntent,
    SetStrategyController,
    StrategyController,
    WorldState,
)
from f1ar.simulation import lightweight_engine as engine
from f1ar.simulation.incidents import lap_incident
from f1ar.simulation.traffic import Ahead, resolve
from f1ar.training.actions import (
    LEGACY_N_ACTIONS,
    N_ACTIONS,
    PER_PIT,
    agent_commands,
    decode,
    encode,
    intent_of,
    is_stop,
    legacy_to_current,
)
from f1ar.training.features import RL_V3_NAMES, rl_observation
from f1ar.training.objectives import ChampionshipContext, Objective, TeamOrder
from f1ar.training.policy_registry import NumpyPolicy
from f1ar.training.scenarios import randomise_character
from tests.conftest import HAM, VER, run_ticks


def test_action_layout_helpers() -> None:
    assert N_ACTIONS == 54 and PER_PIT == 9
    for a in range(N_ACTIONS):
        assert encode(*decode(a)) == a
        assert intent_of(a) is decode(a)[2]
        assert bool(is_stop(a)) == (decode(a)[0] is not None)
    assert np.array_equal(is_stop(np.arange(N_ACTIONS)), np.arange(N_ACTIONS) >= PER_PIT)
    # an 18-action (slice 1-3b) policy raced NORMAL
    mapped = {legacy_to_current(a) for a in range(LEGACY_N_ACTIONS)}
    assert len(mapped) == LEGACY_N_ACTIONS
    assert all(intent_of(a) is RacecraftIntent.NORMAL for a in mapped)


def test_legacy_policies_only_choose_normal_intent() -> None:
    rng = np.random.default_rng(0)
    shapes = [(8, 10), (LEGACY_N_ACTIONS, 8)]
    policy = NumpyPolicy(
        weights=tuple(rng.normal(0, 1, s) for s in shapes),
        biases=tuple(rng.normal(0, 1, s[0]) for s in shapes),
    )
    obs = rng.normal(size=10)
    p = policy.probabilities(obs, np.ones(N_ACTIONS, dtype=bool))
    assert p.shape == (N_ACTIONS,) and p.sum() == pytest.approx(1.0)
    assert all(p[a] == 0.0 for a in range(N_ACTIONS) if intent_of(a) is not RacecraftIntent.NORMAL)


def test_intent_takes_effect_from_the_next_lap(two_car_world: WorldState) -> None:
    state = run_ticks(two_car_world, 300)
    result = engine.apply_action(
        state, SetRacecraftIntent(driver_id=VER, intent=RacecraftIntent.ATTACK, actor_id="AGENT:x")
    )
    (event,) = result.events
    assert event.type == "INTENT_CHANGED" and event.payload["to"] == "ATTACK"
    car = result.state.car(VER)
    assert car.intent is RacecraftIntent.ATTACK and car.lap_intent is RacecraftIntent.NORMAL
    laps = car.laps_completed
    s = result.state
    while s.car(VER).laps_completed == laps:
        s = engine.step(s).state
    assert s.car(VER).lap_intent is RacecraftIntent.ATTACK
    # handing the car back to the baseline resets it to NORMAL
    back = engine.apply_action(
        s, SetStrategyController(driver_id=VER, controller=StrategyController.BASELINE)
    ).state
    assert back.car(VER).intent is RacecraftIntent.NORMAL


def test_agent_commands_carry_the_intent(two_car_world: WorldState) -> None:
    car = two_car_world.car(VER)
    cmds = agent_commands(car, encode(None, car.pace_mode, RacecraftIntent.DEFEND), "AGENT:x")
    assert [type(c) for c in cmds] == [SetRacecraftIntent]
    assert agent_commands(car, encode(None, car.pace_mode), "AGENT:x") == []


def _encounter(setup: RaceSetup, me: CarState, other: CarState) -> list[str]:
    """Outcomes of 300 keyed overtake draws with VER reaching HAM with a margin just above
    the normal threshold (a ~50 % pass)."""
    p = setup.traffic
    margin = p.overtake_threshold_s + 0.7 * p.overtake_scale_s
    out = []
    for lap in range(300):
        m = me.model_copy(update={"laps_completed": lap, "lap_target_s": 80.0})
        o = other.model_copy(update={"laps_completed": lap, "lap_target_s": 80.0})
        # they meet: my end = their end + min_gap - margin
        start = p.min_gap_s - margin
        res = resolve(setup, 7, m, start, [Ahead(o, 0.0, 1.0)])
        out.append("pass" if any(k == "OVERTAKE" for k, _ in res.events) else "held")
    return out


def test_attack_passes_more_and_defend_passes_less(two_car_world: WorldState) -> None:
    setup = two_car_world.setup
    me, other = two_car_world.car(VER), two_car_world.car(HAM)
    normal = _encounter(setup, me, other).count("pass")
    attack = _encounter(
        setup, me.model_copy(update={"lap_intent": RacecraftIntent.ATTACK}), other
    ).count("pass")
    defend = _encounter(
        setup, me, other.model_copy(update={"lap_intent": RacecraftIntent.DEFEND})
    ).count("pass")
    assert attack > normal > defend


def test_attack_and_defend_raise_the_incident_risk(two_car_world: WorldState) -> None:
    setup = two_car_world.setup
    car = two_car_world.car(VER)

    def incidents(intent: RacecraftIntent) -> int:
        c = car.model_copy(update={"lap_intent": intent})
        return sum(lap_incident(setup, s, c, 10) is not None for s in range(20_000))

    normal = incidents(RacecraftIntent.NORMAL)
    assert normal > 0
    assert incidents(RacecraftIntent.ATTACK) > incidents(RacecraftIntent.DEFEND) > normal


def test_v3_sees_points_at_stake_and_the_rival(two_car_world: WorldState) -> None:
    state = run_ticks(two_car_world, 600)
    context = ChampionshipContext({VER: 100.0, HAM: 90.0}, races_left=5)
    order = TeamOrder(Objective.CHAMPIONSHIP, rival=HAM, rival_weight=1.0)
    obs = rl_observation(state, VER, order, context)
    f = dict(zip(RL_V3_NAMES, obs, strict=True))
    assert f["rival.in_race"] == 1.0
    mine = engine.standings(state).index(VER) + 1
    theirs = engine.standings(state).index(HAM) + 1
    assert f["rival.places_ahead"] == pytest.approx((mine - theirs) / 20.0)
    assert f["race.points_now"] == pytest.approx((25, 18)[mine - 1] / 25)
    assert sum(f[f"car.intent.{i.value.lower()}"] for i in RacecraftIntent) == 1.0
    no_rival = dict(
        zip(RL_V3_NAMES, rl_observation(state, VER, TeamOrder(Objective.POINTS), None), strict=True)
    )
    assert no_rival["rival.in_race"] == 0.0


def test_randomise_character_redraws_only_the_ai_cars(isolated_setup: RaceSetup) -> None:
    out = randomise_character(isolated_setup, [VER], np.random.default_rng(0))
    a = {d.driver_id: d.profile.archetype.risk_appetite for d in isolated_setup.drivers}
    b = {d.driver_id: d.profile.archetype.risk_appetite for d in out.drivers}
    assert b[HAM] == a[HAM] and b[VER] != a[VER] and 0.05 <= b[VER] <= 0.95
    assert out.inputs["character_randomisation"].startswith("editorial_assumption")
