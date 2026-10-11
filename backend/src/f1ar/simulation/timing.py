"""Lap-by-lap timing sheet built from LAP_COMPLETED events (display helper, no race logic)."""

from collections import defaultdict
from collections.abc import Iterable, Mapping

from f1ar.domain.race_state import RaceEvent


class LapTimingSheet:
    """Collects line crossings and emits one text row per lap once every car has completed it."""

    def __init__(self, code_of: Mapping[str, str], total_laps: int, every: int = 1) -> None:
        self._code_of = dict(code_of)
        self._total = total_laps
        self._every = every
        self._crossings: dict[int, list[tuple[float, str, float]]] = defaultdict(list)

    def feed(self, events: Iterable[RaceEvent]) -> list[str]:
        rows = []
        for e in events:
            if e.type != "LAP_COMPLETED" or e.driver_id is None:
                continue
            lap = int(e.payload["lap"])
            row = self._crossings[lap]
            row.append(
                (float(e.payload["crossing_time_s"]), e.driver_id, float(e.payload["lap_time_s"]))
            )
            complete = len(row) == len(self._code_of)
            if complete and (lap % self._every == 0 or lap == self._total):
                rows.append(self._format(lap, sorted(row)))
        return rows

    def _format(self, lap: int, order: list[tuple[float, str, float]]) -> str:
        leader_t, leader, leader_lap = order[0]
        cells = [f"{self._code_of[leader]} {leader_lap:6.2f}"]
        cells += [f"{self._code_of[d]} +{t - leader_t:.3f}" for t, d, _ in order[1:]]
        return f"Lap {lap:>2}/{self._total}  " + "  ".join(cells)
