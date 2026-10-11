import pytest

from f1ar.presentation.track_geometry import TrackGeometry

# 100 m x 50 m rectangle, start/finish at the origin, driven counter-clockwise: 300 m loop
RECT = [(0.0, 0.0), (100.0, 0.0), (100.0, 50.0), (0.0, 50.0)]


def test_loop_is_closed_and_measured() -> None:
    g = TrackGeometry.from_points(RECT)
    assert g.length_m == 300.0
    assert (g.xs[-1], g.ys[-1]) == (0.0, 0.0)


@pytest.mark.parametrize(
    ("fraction", "expected"),
    [
        (0.0, (0.0, 0.0)),
        (50 / 300, (50.0, 0.0)),
        (125 / 300, (100.0, 25.0)),
        (200 / 300, (50.0, 50.0)),
        (275 / 300, (0.0, 25.0)),
    ],
)
def test_point_at_follows_the_polyline(fraction: float, expected: tuple[float, float]) -> None:
    x, y = TrackGeometry.from_points(RECT).point_at(fraction)
    assert (x, y) == pytest.approx(expected)


def test_fraction_wraps_around_the_line() -> None:
    g = TrackGeometry.from_points(RECT)
    assert g.point_at(1.0) == pytest.approx(g.point_at(0.0))
    assert g.point_at(1.25) == pytest.approx(g.point_at(0.25))
    assert g.point_at(-0.25) == pytest.approx(g.point_at(0.75))


def test_already_closed_input_is_not_double_closed() -> None:
    closed = TrackGeometry.from_points([*RECT, RECT[0]])
    assert closed.length_m == 300.0 and len(closed.xs) == 5


def test_degenerate_outlines_are_rejected() -> None:
    with pytest.raises(ValueError):
        TrackGeometry.from_points([(0.0, 0.0), (1.0, 1.0)])
    with pytest.raises(ValueError):
        TrackGeometry.from_points([(0.0, 0.0)] * 5)


def test_bounds() -> None:
    assert TrackGeometry.from_points(RECT).bounds() == (0.0, 0.0, 100.0, 50.0)


@pytest.mark.parametrize(
    ("fraction", "heading"),
    [(0.1, 0.0), (125 / 300, 90.0), (200 / 300, 180.0), (275 / 300, -90.0)],
)
def test_heading_follows_direction_of_travel(fraction: float, heading: float) -> None:
    assert TrackGeometry.from_points(RECT).heading_at(fraction) == pytest.approx(heading)
