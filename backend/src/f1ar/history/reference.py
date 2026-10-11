"""Historical reference race built from normalised FastF1 tables (``historical_verified``).

Provides what replay and later comparisons need:

- per-car lap records with session-time start/end (the race clock of the real event);
- the classification after any lap, computed from lap end times (cross-checked against the
  timing feed's own per-lap ``position`` in tests and in ``validate_positions``);
- a reference event list: pit stops (in-lap, compound change), retirements, lapped finishers;
- car progress at any session time, for drawing the real race on the map.

Mode is ``HISTORICAL_REPLAY``. These objects are not ``WorldState`` and can't be stepped,
forked or fed to the engine.
"""

from bisect import bisect_right
from collections.abc import Hashable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pandas as pd

MODE = "HISTORICAL_REPLAY"


@dataclass(frozen=True)
class LapRecord:
    lap: int
    start_s: float
    end_s: float
    lap_time_s: float | None
    compound: str | None
    tyre_life_laps: int | None
    stint: int | None
    feed_position: int | None
    """Position after this lap as published by the timing feed."""
    pit_in: bool
    pit_out: bool


@dataclass(frozen=True)
class CarHistory:
    driver_number: int
    code: str
    team: str
    team_color: str | None
    grid_position: int
    status: str
    classified_position: int | None
    laps: tuple[LapRecord, ...]

    @property
    def laps_completed(self) -> int:
        return len(self.laps)


@dataclass(frozen=True)
class HistoricalEvent:
    lap: int
    session_time_s: float
    type: Literal["PIT_STOP", "RETIREMENT", "FINISHED", "LAPPED_FINISH", "DISQUALIFIED"]
    code: str
    detail: str


def _none(v: Any) -> Any:
    return None if pd.isna(v) else v


@dataclass(frozen=True)
class HistoricalRace:
    event_id: str
    total_laps: int
    race_start_s: float
    """Session time of the start (lap 1 start); replay time 0."""
    cars: tuple[CarHistory, ...]
    mode: str = MODE

    # --- construction ------------------------------------------------------------------

    @classmethod
    def from_tables(
        cls, event_id: str, laps: pd.DataFrame, results: pd.DataFrame
    ) -> "HistoricalRace":
        result_rows: list[dict[Hashable, Any]] = results.to_dict("records")
        lap_rows: list[dict[Hashable, Any]] = laps.sort_values(
            ["driver_number", "lap_number"]
        ).to_dict("records")
        by_car: dict[int, list[LapRecord]] = {}
        for r in lap_rows:
            if pd.isna(r["lap_start_time_s"]) or pd.isna(r["lap_end_time_s"]):
                continue
            by_car.setdefault(int(r["driver_number"]), []).append(
                LapRecord(
                    lap=int(r["lap_number"]),
                    start_s=float(r["lap_start_time_s"]),
                    end_s=float(r["lap_end_time_s"]),
                    lap_time_s=_none(r["lap_time_s"]),
                    compound=_none(r["compound"]),
                    tyre_life_laps=None
                    if pd.isna(r["tyre_life_laps"])
                    else int(r["tyre_life_laps"]),
                    stint=None if pd.isna(r["stint"]) else int(r["stint"]),
                    feed_position=None if pd.isna(r["position"]) else int(r["position"]),
                    pit_in=not pd.isna(r["pit_in_time_s"]),
                    pit_out=not pd.isna(r["pit_out_time_s"]),
                )
            )
        cars = []
        for r in result_rows:
            number = int(r["driver_number"])
            cars.append(
                CarHistory(
                    driver_number=number,
                    code=str(r["driver_code"]),
                    team=str(r["team_name"]),
                    team_color=_none(r.get("team_color")),
                    grid_position=int(r["grid_position"]),
                    status=str(r["status"]),
                    classified_position=None if pd.isna(r["position"]) else int(r["position"]),
                    laps=tuple(by_car.get(number, [])),
                )
            )
        starts = [c.laps[0].start_s for c in cars if c.laps]
        if not starts:
            raise ValueError("no timed laps")
        total = max(c.laps_completed for c in cars)
        return cls(event_id=event_id, total_laps=total, race_start_s=min(starts), cars=tuple(cars))

    @classmethod
    def from_processed_dir(cls, event_id: str, directory: Path) -> "HistoricalRace":
        return cls.from_tables(
            event_id,
            pd.read_parquet(directory / "laps.parquet"),
            pd.read_parquet(directory / "results.parquet"),
        )

    def car(self, code: str) -> CarHistory:
        for c in self.cars:
            if c.code == code:
                return c
        raise KeyError(code)

    # --- classification -------------------------------------------------------------------

    def classification_after_lap(self, lap: int) -> list[tuple[str, float | None]]:
        """``(code, gap_s)`` in order after the leader completes ``lap``.

        Cars on the lead lap are ordered by when they completed ``lap``; gap = time behind the
        leader at that line. Cars that haven't completed ``lap`` (lapped or retired) follow,
        ordered by laps completed then time, with gap ``None``.
        """
        on_lap = [(c.laps[lap - 1].end_s, c.code) for c in self.cars if c.laps_completed >= lap]
        on_lap.sort()
        leader_t = on_lap[0][0] if on_lap else 0.0
        order: list[tuple[str, float | None]] = [(code, t - leader_t) for t, code in on_lap]
        behind = sorted(
            (c for c in self.cars if c.laps_completed < lap),
            key=lambda c: (-c.laps_completed, c.laps[-1].end_s if c.laps else float("inf")),
        )
        order += [(c.code, None) for c in behind]
        return order

    def validate_positions(self) -> list[tuple[str, int, int, int]]:
        """Laps where our end-time order disagrees with the feed: ``(code, lap, ours, feed)``."""
        mismatches = []
        for lap in range(1, self.total_laps + 1):
            ours = {
                code: i + 1
                for i, (code, gap) in enumerate(self.classification_after_lap(lap))
                if gap is not None
            }
            for c in self.cars:
                if c.laps_completed >= lap:
                    feed = c.laps[lap - 1].feed_position
                    if feed is not None and feed != ours[c.code]:
                        mismatches.append((c.code, lap, ours[c.code], feed))
        return mismatches

    # --- events ------------------------------------------------------------------------------

    def events(self) -> list[HistoricalEvent]:
        events: list[HistoricalEvent] = []
        for c in self.cars:
            for i, lap in enumerate(c.laps):
                if lap.pit_in:
                    nxt = c.laps[i + 1] if i + 1 < len(c.laps) else None
                    new = nxt.compound if nxt and nxt.pit_out else None
                    detail = (
                        f"{lap.compound} -> {new}" if new else f"{lap.compound} -> (did not rejoin)"
                    )
                    events.append(HistoricalEvent(lap.lap, lap.end_s, "PIT_STOP", c.code, detail))
            if not c.laps:
                continue
            last = c.laps[-1]
            if c.status == "Disqualified" and self.took_flag(c.code):
                events.append(
                    HistoricalEvent(
                        last.lap,
                        last.end_s,
                        "DISQUALIFIED",
                        c.code,
                        f"took the flag after {c.laps_completed} laps; disqualified afterwards",
                    )
                )
            elif self.took_flag(c.code) and c.laps_completed == self.total_laps:
                events.append(
                    HistoricalEvent(
                        last.lap, last.end_s, "FINISHED", c.code, f"P{c.classified_position}"
                    )
                )
            elif self.took_flag(c.code):
                events.append(
                    HistoricalEvent(
                        last.lap,
                        last.end_s,
                        "LAPPED_FINISH",
                        c.code,
                        f"P{c.classified_position}, {c.laps_completed} laps",
                    )
                )
            else:
                events.append(
                    HistoricalEvent(
                        last.lap,
                        last.end_s,
                        "RETIREMENT",
                        c.code,
                        f"{c.status} after {c.laps_completed} laps",
                    )
                )
        return sorted(events, key=lambda e: (e.session_time_s, e.code))

    # --- on-track result vs. official classification -------------------------------------------

    @property
    def flag_s(self) -> float:
        """Session time the chequered flag fell: the first car to complete the full distance."""
        return min(c.laps[-1].end_s for c in self.cars if c.laps_completed == self.total_laps)

    def took_flag(self, code: str) -> bool:
        """Crossed the line after the flag, i.e. finished *on track* (lapped or not).

        This is about what happened on the road. The official status may still differ, e.g. a
        post-race disqualification (2024 Belgian GP: RUS won on the road, then was DSQ'd).
        """
        car = self.car(code)
        return bool(car.laps) and car.laps[-1].end_s >= self.flag_s

    # --- replay ------------------------------------------------------------------------------

    def progress_at(self, replay_s: float) -> dict[str, tuple[int, float, str]]:
        """Per car at ``replay_s`` seconds after the start: ``(laps_done, lap_fraction, state)``.

        Within a lap the car is placed linearly in time (start -> end), which is how a timing
        screen would interpolate. ``state`` is RUNNING, FINISHED or RETIRED.
        """
        t = self.race_start_s + replay_s
        out: dict[str, tuple[int, float, str]] = {}
        for c in self.cars:
            if not c.laps:
                out[c.code] = (0, 0.0, "RETIRED")
                continue
            ends = [lap.end_s for lap in c.laps]
            done = bisect_right(ends, t)
            if done >= c.laps_completed:
                state = "FINISHED" if self.took_flag(c.code) else "RETIRED"
                out[c.code] = (c.laps_completed, 0.0, state)
                continue
            lap = c.laps[done]
            span = lap.end_s - lap.start_s
            frac = 0.0 if span <= 0 else min(max((t - lap.start_s) / span, 0.0), 1.0)
            out[c.code] = (done, frac, "RUNNING")
        return out

    @property
    def duration_s(self) -> float:
        return max(c.laps[-1].end_s for c in self.cars if c.laps) - self.race_start_s
