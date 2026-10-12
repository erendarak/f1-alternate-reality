"""Race-level metrics measured the same way on real and simulated races (EXP-002, M4).

Overtakes: positions = order of line crossings at the end of each lap (lap 0 = grid); a pass is
one car getting ahead of another between consecutive lap ends; pairs where either car pitted at
the end of either lap are excluded (stops, not passes). Safety cars / VSCs: deployments in the
track-status timeline (codes 4 and 6). Stops: pit entries, except entries during a red flag
(tyre changes then are free and not stops; the new stint still shows in the sequence). Values
are derived from F1 timing data via FastF1.
"""

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class RaceMetrics:
    order: list[str]
    """Road order at the end (finishers by laps then time, then retirements)."""
    finishers: set[str]
    winner: str
    winner_time_s: float
    stops: dict[str, int]
    sequences: dict[str, str]
    overtakes: int
    retirements: int
    safety_cars: int
    vscs: int
    lapped: int
    extra: dict[str, Any] = field(default_factory=dict)


def count_overtakes(
    crossings: dict[int, dict[str, float]],
    grid: Sequence[str],
    stopped_at_end_of: dict[str, set[int]],
) -> int:
    """Passes between consecutive lap ends (lap 0 = grid order), pit laps excluded."""
    positions: dict[int, dict[str, int]] = {0: {d: i for i, d in enumerate(grid)}}
    for lap, times in crossings.items():
        positions[lap] = {d: i for i, d in enumerate(sorted(times, key=lambda d: times[d]))}
    passes = 0
    for lap in sorted(positions):
        if lap == 0 or lap - 1 not in positions:
            continue
        before, after = positions[lap - 1], positions[lap]
        cars = [
            d
            for d in after
            if d in before
            and lap not in stopped_at_end_of.get(d, set())
            and lap - 1 not in stopped_at_end_of.get(d, set())
        ]
        for a in cars:
            for b in cars:
                if before[a] > before[b] and after[a] < after[b]:
                    passes += 1
    return passes


def sequence_of(stint_compounds: Iterable[str]) -> str:
    """One letter per stint, e.g. ``M-H`` (a repeated compound stays visible: ``M-M``)."""
    return "-".join(c[0] for c in stint_compounds)


# --- the real race -------------------------------------------------------------------------------


def red_flag_periods(status: pd.DataFrame) -> list[tuple[float, float]]:
    """``[start, end)`` session times of red flags (status 5) in a track-status timeline."""
    times = [float(t) for t in status["session_time_s"]]
    codes = [str(c) for c in status["status"]]
    periods = []
    for i, code in enumerate(codes):
        if code == "5" and (i == 0 or codes[i - 1] != "5"):
            end = next((times[j] for j in range(i + 1, len(codes)) if codes[j] != "5"), math.inf)
            periods.append((times[i], end))
    return periods


def real_metrics(data_dir: Path) -> RaceMetrics:
    laps = pd.read_parquet(data_dir / "laps.parquet")
    results = pd.read_parquet(data_dir / "results.parquet")
    status = pd.read_parquet(data_dir / "track_status.parquet")
    code_of = {
        int(n): str(c)
        for n, c in zip(results["driver_number"], results["driver_code"], strict=True)
    }
    official = {
        str(c): str(s) for c, s in zip(results["driver_code"], results["status"], strict=True)
    }
    rows = [r | {"code": code_of[int(r["driver_number"])]} for r in laps.to_dict("records")]
    timed = [r for r in rows if not pd.isna(r["lap_end_time_s"])]

    last: dict[str, tuple[int, float]] = {}  # code -> (laps completed, time of the last crossing)
    crossings: dict[int, dict[str, float]] = {}
    for r in timed:
        lap, t = int(r["lap_number"]), float(r["lap_end_time_s"])
        crossings.setdefault(lap, {})[r["code"]] = t
        if lap >= last.get(r["code"], (0, 0.0))[0]:
            last[r["code"]] = (lap, t)
    total = max(lap for lap, _ in last.values())
    finishers = {d for d in last if official.get(d) in ("Finished", "Lapped", "Disqualified")}
    order = sorted(last, key=lambda d: (d not in finishers, -last[d][0], last[d][1]))
    start = min(float(r["lap_start_time_s"]) for r in rows if int(r["lap_number"]) == 1)

    # Pit entries: all of them are excluded from pass counting; but entering the pit lane under a
    # red flag (everyone does; tyres may be changed freely) is not a pit stop (found in M4: 2024
    # Japan showed 3.0 "stops" per finisher, Monaco 1.44).
    red = red_flag_periods(status)
    stopped: dict[str, set[int]] = {}
    stops: dict[str, int] = {}
    for r in rows:
        if not pd.isna(r["pit_in_time_s"]):
            stopped.setdefault(r["code"], set()).add(int(r["lap_number"]))
            t = float(r["pit_in_time_s"])
            if not any(a <= t < b for a, b in red):
                stops[r["code"]] = stops.get(r["code"], 0) + 1
    grid_pairs = sorted(
        (int(g), str(c))
        for c, g in zip(results["driver_code"], results["grid_position"], strict=True)
    )
    grid = [c for g, c in grid_pairs if g > 0] + [c for g, c in grid_pairs if g <= 0]

    stints: dict[str, dict[int, str]] = {}
    for r in sorted(rows, key=lambda r: int(r["lap_number"])):
        if not pd.isna(r["compound"]):
            stints.setdefault(r["code"], {}).setdefault(int(r["stint"]), str(r["compound"]))
    sequences = {
        d: sequence_of(by_stint[k] for k in sorted(by_stint)) for d, by_stint in stints.items()
    }

    codes = [str(x) for x in status["status"]]

    def deployments(code: str) -> int:
        return sum(1 for i, c in enumerate(codes) if c == code and (i == 0 or codes[i - 1] != code))

    winner = order[0]
    return RaceMetrics(
        order=order,
        finishers=finishers,
        winner=winner,
        winner_time_s=last[winner][1] - start,
        stops={d: stops.get(d, 0) for d in finishers},
        sequences={d: sequences.get(d, "?") for d in finishers},
        overtakes=count_overtakes(crossings, grid, stopped),
        retirements=len(last) - len(finishers),
        safety_cars=deployments("4"),
        vscs=deployments("6"),
        lapped=sum(1 for d in finishers if last[d][0] < total),
    )
