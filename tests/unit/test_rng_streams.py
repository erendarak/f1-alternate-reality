import statistics

from f1ar.simulation.rng_streams import normal, uniform


def test_same_key_same_draw() -> None:
    assert uniform(42, "lap_noise", "a", 3) == uniform(42, "lap_noise", "a", 3)
    assert normal(42, "lap_noise", "a", 3) == normal(42, "lap_noise", "a", 3)


def test_any_key_part_or_seed_changes_the_draw() -> None:
    base = uniform(42, "lap_noise", "a", 3)
    assert base != uniform(43, "lap_noise", "a", 3)
    assert base != uniform(42, "lap_noise", "b", 3)
    assert base != uniform(42, "lap_noise", "a", 4)
    assert base != uniform(42, "weather", "a", 3)


def test_uniform_is_in_open_unit_interval() -> None:
    draws = [uniform(1, i) for i in range(5000)]
    assert all(0.0 < u < 1.0 for u in draws)
    assert abs(statistics.fmean(draws) - 0.5) < 0.02


def test_normal_has_roughly_standard_moments() -> None:
    draws = [normal(7, "n", i) for i in range(20000)]
    assert abs(statistics.fmean(draws)) < 0.03
    assert abs(statistics.pstdev(draws) - 1.0) < 0.03
