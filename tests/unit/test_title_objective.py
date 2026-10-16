"""D-050: the CHAMPIONSHIP objective is the chance of winning the title (S-shaped in the gap)."""

import math

import pytest

from f1ar.training.objectives import (
    CarResult,
    ChampionshipContext,
    Objective,
    TeamOrder,
    objective_value,
    team_principal,
    title_probability,
)


def _results(me: int, rival: int) -> dict[str, CarResult]:
    points = {1: 25, 2: 18, 3: 15, 4: 12}
    return {
        "me": CarResult(me, True, points.get(me, 0), 0.0),
        "rival": CarResult(rival, True, points.get(rival, 0), 0.0),
    }


def test_title_probability_shape() -> None:
    assert title_probability(0.0, 3) == pytest.approx(0.5)
    assert title_probability(1.0, 0) == 1.0 and title_probability(-1.0, 0) == 0.0
    assert title_probability(0.0, 0) == 0.5
    # behind: convex (each point is worth more the closer you get); ahead: concave
    d = [title_probability(g + 1, 2) - title_probability(g, 2) for g in (-30, -15, -5)]
    assert d[0] < d[1] < d[2]
    d = [title_probability(g + 1, 2) - title_probability(g, 2) for g in (5, 15, 30)]
    assert d[0] > d[1] > d[2]
    assert 0.0 < title_probability(-15, 2) < 0.5 < title_probability(15, 2) < 1.0
    assert math.isclose(title_probability(-15, 2), 1 - title_probability(15, 2))


def test_a_leader_and_a_chaser_want_different_things() -> None:
    """Winning ahead of the rival (+7 pts) vs a safe second behind them: the chaser values the
    swing more than the leader does (the leader is already close to safe)."""

    def value(gap: float, me: int, rival: int) -> float:
        order = TeamOrder(Objective.CHAMPIONSHIP, rival="rival", gap_before=gap, races_after=2)
        return objective_value(order, "me", _results(me, rival), 20)

    leader_gain = value(25.0, 1, 2) - value(25.0, 2, 1)
    chaser_gain = value(-10.0, 1, 2) - value(-10.0, 2, 1)
    assert chaser_gain > leader_gain > 0


def test_the_team_principal_passes_the_standings() -> None:
    class _Setup:  # the team principal only needs car ranks: a minimal stand-in
        drivers = ()

    context = ChampionshipContext({"me": 200.0, "rival": 210.0, "x": 50.0}, races_left=4)
    import f1ar.training.objectives as obj

    original = obj.car_ranks
    obj.car_ranks = lambda _s: {"me": 0, "rival": 1, "x": 5}  # type: ignore[assignment]
    try:
        order = team_principal(_Setup(), "me", context)  # type: ignore[arg-type]
    finally:
        obj.car_ranks = original
    assert order.objective is Objective.CHAMPIONSHIP and order.rival == "rival"
    assert order.gap_before == -10.0 and order.races_after == 3
