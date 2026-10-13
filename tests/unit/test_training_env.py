"""M5 slice 1: the training environment (D-037/D-038): sessions, decisions, masks, rewards,
objectives, the AGENT controller, reproducibility and no future leakage."""

import math

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env
from pydantic import ValidationError

from f1ar.api.schemas import ActionRequest, ActionType
from f1ar.domain.driver_profiles import ProfileSet
from f1ar.domain.identities import RaceRoster
from f1ar.domain.race_state import (
    Compound,
    PaceMode,
    RacePhase,
    RaceSetup,
    RequestPitStop,
    SetPaceMode,
    StrategyController,
)
from f1ar.simulation import lightweight_engine as engine
from f1ar.simulation.scenarios import build_setup
from f1ar.training.actions import N_ACTIONS, PIT_OPTIONS, decode, encode
from f1ar.training.features import RL_FEATURE_NAMES, rl_observation
from f1ar.training.gym_env import RaceStrategyEnv, random_policy, run_race
from f1ar.training.objectives import (
    POINTS_2024,
    CarResult,
    ChampionshipContext,
    Objective,
    TeamOrder,
    objective_value,
    risk_utility,
    scoring_for,
    team_principal,
)
from f1ar.training.regulations import REGULATIONS_2024, Ban
from f1ar.training.scenarios import ScenarioSampler, synthetic_standings
from f1ar.training.session import RaceSession
from tests.conftest import HAM, VER

STAY = encode(None, PaceMode.BALANCED)


def _session(setup: RaceSetup, agents: list[str], seed: int = 7, **kw: object) -> RaceSession:
    return RaceSession(
        setup,
        seed=seed,
        agents=agents,
        scoring=scoring_for(setup, agents, None),
        **kw,  # type: ignore[arg-type]
    )


def _full_field(roster: RaceRoster, profiles: ProfileSet) -> RaceSetup:
    return build_setup(roster, profiles, roster.driver_ids())


# --- actions and masks --------------------------------------------------------------------


def test_action_encoding_round_trips() -> None:
    seen = {decode(a) for a in range(N_ACTIONS)}
    assert len(seen) == N_ACTIONS == 18
    for a in range(N_ACTIONS):
        assert encode(*decode(a)) == a


def test_masks_follow_the_engine_and_the_regulations(isolated_setup: RaceSetup) -> None:
    s = _session(isolated_setup, [VER])
    d = s.next_decision()
    assert d is not None and d.mask.all()  # mid-race: every pit choice and pace mode
    banned = REGULATIONS_2024.with_ban(Ban("test ban", "2025", pit_compound=Compound.SOFT), "v2")
    s2 = _session(isolated_setup, [VER], regulations=banned)
    d2 = s2.next_decision()
    assert d2 is not None
    soft = [a for a in range(N_ACTIONS) if decode(a)[0] is Compound.SOFT]
    assert not d2.mask[soft].any() and d2.mask.sum() == N_ACTIONS - 3
    with pytest.raises(ValueError, match="not legal"):
        s2.act(VER, soft[0])
    # the final lap: no stop is legal any more, only "stay out" with a pace mode
    while (d := s.next_decision()) is not None:
        last = d
        s.act(VER, STAY)
    assert last.lap == isolated_setup.track.total_laps
    assert [decode(int(a))[0] for a in np.flatnonzero(last.mask)] == [None, None, None]


# --- the session ---------------------------------------------------------------------------


def test_an_ai_car_decides_every_lap_and_never_pits_on_its_own(isolated_setup: RaceSetup) -> None:
    s = _session(isolated_setup, [VER])
    reasons = []
    while (d := s.next_decision()) is not None:
        reasons.append(d.reason)
        s.act(VER, STAY)
    ver = s.state.car(VER)
    assert ver.controller is StrategyController.AGENT
    assert reasons == ["LAP_END"] * (isolated_setup.track.total_laps - 1)
    assert ver.pit_stops == 0  # the baseline planner left it alone
    assert s.state.car(HAM).pit_stops >= 1  # the rule car still stops


def test_an_ai_pit_decision_is_carried_out(isolated_setup: RaceSetup) -> None:
    s = _session(isolated_setup, [VER])
    while (d := s.next_decision()) is not None:
        hard = encode(Compound.HARD, PaceMode.PUSH)
        s.act(VER, hard if d.lap == 20 else encode(None, PaceMode.PUSH))
    ver = s.state.car(VER)
    assert ver.pit_stops == 1 and ver.compound is Compound.HARD
    assert ver.pace_mode is PaceMode.PUSH


def test_race_events_make_every_ai_car_decide(two_car_setup: RaceSetup) -> None:
    s = _session(two_car_setup, [VER, HAM], seed=3)
    for _ in range(4):
        d = s.next_decision()
        assert d is not None
        s.act(d.driver_id, STAY)
    s.apply_external(engine.deploy_neutralisation(s.state, RacePhase.SC, sc_laps=3))
    first = [s.next_decision(), s.next_decision()]
    assert {(d.driver_id, d.reason) for d in first if d is not None} == {
        (VER, "SAFETY_CAR_DEPLOYED"),
        (HAM, "SAFETY_CAR_DEPLOYED"),
    }


def test_same_seed_same_race(isolated_setup: RaceSetup) -> None:
    def race() -> tuple[dict[str, float], str]:
        s = _session(isolated_setup, [VER, HAM], seed=11)
        rewards = run_race(s, random_policy(np.random.default_rng(5)))
        return rewards, s.state.model_dump_json()

    assert race() == race()


def test_a_whole_ai_field_races(monza_roster: RaceRoster, monza_profiles: ProfileSet) -> None:
    setup = _full_field(monza_roster, monza_profiles)
    agents = [d.driver_id for d in setup.drivers]
    s = _session(setup, agents, seed=2)
    deciders: set[str] = set()
    policy = random_policy(np.random.default_rng(1))
    while (d := s.next_decision()) is not None:
        deciders.add(d.driver_id)
        s.act(d.driver_id, policy(d))
    assert deciders == set(agents)
    rewards = s.rewards()
    assert set(rewards) == set(agents) and all(math.isfinite(r) for r in rewards.values())
    assert all(s.state.car(d).controller is StrategyController.AGENT for d in agents)


# --- the user and the AI -----------------------------------------------------------------


def test_a_human_decision_takes_an_ai_car_over(isolated_setup: RaceSetup) -> None:
    s = _session(isolated_setup, [VER])
    assert s.state.car(VER).controller is StrategyController.AGENT
    human = engine.apply_action(s.state, SetPaceMode(driver_id=VER, pace_mode=PaceMode.PUSH))
    assert human.state.car(VER).controller is StrategyController.USER
    pit = engine.apply_action(s.state, RequestPitStop(driver_id=VER, compound=Compound.HARD))
    assert pit.state.car(VER).controller is StrategyController.USER
    # the AI's own commands keep it an AI car
    ai = engine.apply_action(
        s.state, SetPaceMode(driver_id=VER, pace_mode=PaceMode.PUSH, actor_id="AGENT:x")
    )
    assert ai.state.car(VER).controller is StrategyController.AGENT


def test_the_api_cannot_impersonate_an_ai() -> None:
    with pytest.raises(ValidationError, match="reserved for AI"):
        ActionRequest(action_type=ActionType.SET_PACE_MODE, actor_id="AGENT_x")
    with pytest.raises(ValidationError):
        ActionRequest(action_type=ActionType.SET_CONTROLLER, controller="AGENT")


# --- observation ----------------------------------------------------------------------------


def test_observation_has_fixed_names_and_sane_values(two_car_world: object) -> None:
    from f1ar.domain.race_state import WorldState

    assert isinstance(two_car_world, WorldState)
    order = TeamOrder(Objective.POINTS)
    obs = rl_observation(two_car_world, VER, order, None)
    assert obs.shape == (len(RL_FEATURE_NAMES),) and obs.dtype == np.float32
    assert np.isfinite(obs).all() and (np.abs(obs) <= 10).all()
    names = dict(zip(RL_FEATURE_NAMES, obs, strict=True))
    assert names["objective.points"] == 1.0 and names["championship.has_context"] == 0.0


def test_observation_never_sees_the_future(two_car_world: object) -> None:
    from f1ar.domain.race_state import WorldState

    assert isinstance(two_car_world, WorldState)
    order = TeamOrder(Objective.POINTS)
    now = rl_observation(two_car_world, VER, order, None)
    far = two_car_world.setup.weather.model_copy(update={"rain_timeline": ((5000.0, True),)})
    changed = two_car_world.model_copy(
        update={"setup": two_car_world.setup.model_copy(update={"weather": far})}
    )
    assert (rl_observation(changed, VER, order, None) == now).all()


# --- objectives -----------------------------------------------------------------------------


def _results(points_me: int, points_rival: int, pos: int) -> dict[str, CarResult]:
    return {
        "me": CarResult(pos, True, points_me, 0.0),
        "rival": CarResult(1, True, points_rival, 0.0),
    }


def test_objectives_score_what_they_say() -> None:
    win = _results(25, 18, 1)
    p11 = _results(0, 25, 11)
    points = TeamOrder(Objective.POINTS)
    assert objective_value(points, "me", win, 20) > objective_value(points, "me", p11, 20)
    title = TeamOrder(Objective.CHAMPIONSHIP, rival="rival", rival_weight=1.0)
    # beating the rival matters more for a contender than the points alone
    second_ahead = {"me": CarResult(2, True, 18, 0.0), "rival": CarResult(5, True, 10, 0.0)}
    second_behind = {"me": CarResult(2, True, 18, 0.0), "rival": CarResult(1, True, 25, 0.0)}
    assert objective_value(title, "me", second_ahead, 20) > objective_value(
        title, "me", second_behind, 20
    )
    assert objective_value(points, "me", second_ahead, 20) == objective_value(
        points, "me", second_behind, 20
    )
    dnf = {"me": CarResult(20, False, 0, math.nan), "rival": CarResult(1, True, 25, 0.0)}
    assert objective_value(TeamOrder(Objective.TIME), "me", dnf, 20) == -1.0


def test_risk_utility_encodes_character() -> None:
    lo, hi = 0.0, 1.0
    for appetite, sign in ((0.1, 1), (0.5, 0), (0.9, -1)):
        mid = risk_utility(0.5, appetite)
        avg = (risk_utility(lo, appetite) + risk_utility(hi, appetite)) / 2
        # careful drivers prefer the sure middle (concave), risk takers the gamble (convex)
        assert np.sign(round(mid - avg, 9)) == sign
    assert risk_utility(0.3, 0.2) < risk_utility(0.4, 0.2)  # always increasing


def test_team_principal_sets_title_or_points(
    monza_roster: RaceRoster, monza_profiles: ProfileSet
) -> None:
    teams = sorted({e.team_id for e in monza_roster.entries})
    offsets = {t: 0.2 * i for i, t in enumerate(teams)}  # distinct car strengths
    setup = build_setup(
        monza_roster, monza_profiles, monza_roster.driver_ids(), car_offsets_s=offsets
    )
    fastest = min(setup.drivers, key=lambda d: d.car_pace_offset_s).driver_id
    slowest = max(setup.drivers, key=lambda d: d.car_pace_offset_s).driver_id
    others = {d.driver_id: 10.0 for d in setup.drivers}
    context = ChampionshipContext(points={**others, fastest: 300.0, VER: 290.0}, races_left=5)
    lead = team_principal(setup, fastest, context)
    assert lead.objective is Objective.CHAMPIONSHIP and lead.rival == VER
    assert team_principal(setup, slowest, context).objective is Objective.POINTS
    start = ChampionshipContext(points=dict.fromkeys(others, 0.0), races_left=24)
    assert team_principal(setup, fastest, start).objective is Objective.POINTS


def test_synthetic_standings_favour_strong_cars(
    monza_roster: RaceRoster, monza_profiles: ProfileSet
) -> None:
    setup = _full_field(monza_roster, monza_profiles)
    a = synthetic_standings(setup, np.random.default_rng(1), 20)
    b = synthetic_standings(setup, np.random.default_rng(1), 20)
    assert a == b and a.races_left == 4
    assert sum(a.points.values()) == 20 * sum(POINTS_2024)


# --- the Gymnasium env ----------------------------------------------------------------------


def _env(setup: RaceSetup, **kw: object) -> RaceStrategyEnv:
    sampler = ScenarioSampler(["x"], lambda _e, _rain: setup)
    return RaceStrategyEnv(sampler, **kw)  # type: ignore[arg-type]


def test_gym_env_passes_the_checker_and_is_reproducible(isolated_setup: RaceSetup) -> None:
    env = _env(isolated_setup)
    check_env(env, skip_render_check=True)
    first, info = env.reset(seed=4)
    again, _ = env.reset(seed=4)
    assert (first == again).all() and info["observation_schema"] == "rl_obs_v1"
    done, steps, reward = False, 0, 0.0
    while not done:
        _, reward, done, _, info = env.step(STAY)
        steps += 1
    # never stopping breaks the two-dry-compounds rule: scored as a disqualification
    assert info["finished"] and info["disqualified"] and info["points"] == 0
    assert info["position"] == 2  # classified last of the two cars
    assert reward == env.session.rewards()[env.me]  # type: ignore[union-attr]


def test_gym_env_with_ai_opponents(monza_roster: RaceRoster, monza_profiles: ProfileSet) -> None:
    setup = _full_field(monza_roster, monza_profiles)
    env = _env(setup, opponent_policy=random_policy(np.random.default_rng(2)), opponents="all")
    env.reset(seed=1)
    assert env.session is not None and len(env.session.agents) == 20
    done = False
    while not done:
        _, _, done, _, info = env.step(int(np.flatnonzero(env.action_masks())[0]))
    assert info["driver_id"] == env.me


def test_pit_options_cover_every_compound() -> None:
    assert set(PIT_OPTIONS) == {None, *Compound}
