"""Lap-by-lap timing sheet and result gaps from events/state (display helpers, no race logic)."""

from collections import defaultdict
from collections.abc import Iterable, Mapping

from f1ar.domain.race_state import CarStatus, RaceEvent, WorldState


class LapTimingSheet:
    """Collects line crossings and emits one text row per lap once every car still racing on
    that lap has completed it. Cars flagged a lap (or more) down don't hold rows back. A ``P``
    after a car's gap marks a lap that ended in the pit lane."""

    def __init__(self, code_of: Mapping[str, str], total_laps: int, every: int = 1) -> None:
        self._code_of = dict(code_of)
        self._total = total_laps
        self._every = every
        self._crossings: dict[int, list[tuple[float, str, float]]] = defaultdict(list)
        self._pitted: dict[int, set[str]] = defaultdict(set)
        self._finished_after: dict[str, int] = {}
        self._next_lap = 1

    def feed(self, events: Iterable[RaceEvent]) -> list[str]:
        for e in events:
            if e.driver_id is None:
                continue
            if e.type == "LAP_COMPLETED":
                self._crossings[int(e.payload["lap"])].append(
                    (
                        float(e.payload["crossing_time_s"]),
                        e.driver_id,
                        float(e.payload["lap_time_s"]),
                    )
                )
            elif e.type == "PIT_STOP":
                self._pitted[int(e.payload["lap"])].add(e.driver_id)
            elif e.type == "CAR_FINISHED":
                self._finished_after[e.driver_id] = int(e.payload["laps"])
        rows = []
        while self._next_lap <= self._total and self._lap_complete(self._next_lap):
            lap = self._next_lap
            if lap % self._every == 0 or lap == self._total:
                rows.append(self._format(lap, sorted(self._crossings[lap])))
            self._next_lap += 1
        return rows

    def _lap_complete(self, lap: int) -> bool:
        expected = sum(1 for d in self._code_of if self._finished_after.get(d, lap) >= lap)
        return expected > 0 and len(self._crossings[lap]) == expected

    def _format(self, lap: int, order: list[tuple[float, str, float]]) -> str:
        pitted = self._pitted[lap]
        leader_t, leader, leader_lap = order[0]
        cells = [f"{self._code_of[leader]} {leader_lap:6.2f}{'P' if leader in pitted else ''}"]
        cells += [
            f"{self._code_of[d]} +{t - leader_t:.3f}{'P' if d in pitted else ''}"
            for t, d, _ in order[1:]
        ]
        return f"Lap {lap:>2}/{self._total}  " + "  ".join(cells)


def result_gap(state: WorldState, driver_id: str, winner_id: str) -> str:
    """Gap to the winner as on a results sheet: ``+12.345s``, or ``+1 lap`` for lapped cars."""
    car, winner = state.car(driver_id), state.car(winner_id)
    laps_down = winner.laps_completed - car.laps_completed
    if car.status is CarStatus.FINISHED and laps_down > 0:
        return f"+{laps_down} lap{'s' if laps_down > 1 else ''}"
    return f"+{(car.finish_time_s or 0.0) - (winner.finish_time_s or 0.0):.3f}s"
