"""Arc-length mapping from race progress to 2D track coordinates (plan §7.4, §14.4).

The engine's canonical car position is ``distance_m`` into the lap. The viewer needs X/Y. We
treat the track outline as a closed polyline and place a car at the same *fraction of a lap*
along it:

    fraction = distance_m / engine_track_length_m      (0 = start/finish line)
    point    = point at fraction * outline_length along the polyline

Using a fraction, not raw metres, means a small mismatch between the outline's measured
length (sampled car path, ~5.76 km) and the official length used by the engine (5.793 km)
can't make cars drift off the end of the drawing. This is display only and never feeds
back into the race.
"""

import bisect
import math
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class TrackGeometry:
    xs: tuple[float, ...]
    ys: tuple[float, ...]
    cumulative_m: tuple[float, ...]
    """Arc length at each vertex; the last entry is the full closed-loop length."""

    @classmethod
    def from_points(cls, points: Sequence[tuple[float, float]]) -> "TrackGeometry":
        """Closed loop through ``points`` (the first point is the start/finish line)."""
        if len(points) < 3:
            raise ValueError("a track outline needs at least 3 points")
        xs = [float(x) for x, _ in points]
        ys = [float(y) for _, y in points]
        if (xs[0], ys[0]) != (xs[-1], ys[-1]):
            xs.append(xs[0])  # close the loop
            ys.append(ys[0])
        cumulative = [0.0]
        for i in range(1, len(xs)):
            cumulative.append(cumulative[-1] + math.hypot(xs[i] - xs[i - 1], ys[i] - ys[i - 1]))
        if cumulative[-1] <= 0:
            raise ValueError("degenerate outline (zero length)")
        return cls(xs=tuple(xs), ys=tuple(ys), cumulative_m=tuple(cumulative))

    @property
    def length_m(self) -> float:
        return self.cumulative_m[-1]

    def _segment(self, fraction: float) -> tuple[int, float]:
        target = (fraction % 1.0) * self.length_m
        i = bisect.bisect_right(self.cumulative_m, target)
        return min(max(i, 1), len(self.cumulative_m) - 1), target

    def heading_at(self, fraction: float) -> float:
        """Direction of travel at ``fraction`` in degrees, counter-clockwise from +X."""
        i, _ = self._segment(fraction)
        dx, dy = self.xs[i] - self.xs[i - 1], self.ys[i] - self.ys[i - 1]
        return math.degrees(math.atan2(dy, dx))

    def point_at(self, fraction: float) -> tuple[float, float]:
        """X/Y at ``fraction`` of a lap; wraps around, so 1.0 == 0.0 == the line."""
        i, target = self._segment(fraction)
        seg_start, seg_end = self.cumulative_m[i - 1], self.cumulative_m[i]
        t = 0.0 if seg_end == seg_start else (target - seg_start) / (seg_end - seg_start)
        x = self.xs[i - 1] + t * (self.xs[i] - self.xs[i - 1])
        y = self.ys[i - 1] + t * (self.ys[i] - self.ys[i - 1])
        return x, y

    def bounds(self) -> tuple[float, float, float, float]:
        return min(self.xs), min(self.ys), max(self.xs), max(self.ys)
