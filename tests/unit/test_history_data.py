"""H0 (D-052): historical race data 2000-2025: cached Jolpica pages and what is inferred."""

import json
from pathlib import Path
from typing import Any

import pandas as pd

from f1ar.data.history_inference import inferred_pit_laps, match, neutralised_laps
from f1ar.data.jolpica_history import race_done, race_tables


def _page(kind: str, rows: list[Any], total: int) -> dict[str, Any]:
    key = {"laps": "Laps", "results": "Results", "pitstops": "PitStops"}.get(
        kind, "QualifyingResults"
    )
    return {"MRData": {"total": str(total), "RaceTable": {"Races": [{key: rows}]}}}


def _write(folder: Path, name: str, data: dict[str, Any]) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / name).write_text(json.dumps(data), encoding="utf-8")


def test_race_tables_parse_cached_pages(tmp_path: Path) -> None:
    folder = tmp_path / "2012" / "03"
    laps = [
        {"number": "1", "Timings": [{"driverId": "a", "position": "1", "time": "1:35.500"}]},
        {"number": "2", "Timings": [{"driverId": "a", "position": "1", "time": "1:31.250"}]},
    ]
    _write(folder, "laps_00000.json", _page("laps", laps, 2))
    result = {
        "Driver": {"driverId": "a", "code": "AAA"},
        "number": "7",
        "Constructor": {"constructorId": "t"},
        "grid": "2",
        "position": "1",
        "positionText": "1",
        "laps": "2",
        "status": "Finished",
        "points": "25",
        "Time": {"time": "1:30:01.100"},
    }
    _write(folder, "results_00000.json", _page("results", [result], 1))
    _write(folder, "qualifying_00000.json", _page("qualifying", [], 0))
    stop = {"driverId": "a", "lap": "1", "stop": "1", "duration": "22.4"}
    _write(folder, "pitstops_00000.json", _page("pitstops", [stop], 1))
    assert race_done(2012, 3, tmp_path)
    t = race_tables(2012, 3, tmp_path)
    assert t["laps"]["lap_time_s"].tolist() == [95.5, 91.25]
    assert t["results"].iloc[0]["time_s"] == 5401.1
    assert t["pitstops"].iloc[0]["duration_s"] == 22.4
    assert not race_done(2012, 4, tmp_path)


def _field(n_laps: int = 20, slow_lap: int | None = None) -> pd.DataFrame:
    rows = []
    for d in range(10):
        for lap in range(1, n_laps + 1):
            t = 90.0 + 0.1 * d
            if slow_lap is not None and lap == slow_lap:
                t *= 1.35  # safety car: the whole field slows
            if d == 3 and lap == 8:
                t += 21.0  # a stop under green
            if d == 5 and lap == 12:
                t += 6.0  # an in-lap, then the slower out-lap
            if d == 5 and lap == 13:
                t += 17.0
            rows.append({"driver": f"d{d}", "lap": lap, "lap_time_s": t})
    return pd.DataFrame(rows)


def test_neutralised_laps_and_pit_stops_are_inferred() -> None:
    laps = _field(slow_lap=15)
    assert neutralised_laps(laps) == {15}
    stops = inferred_pit_laps(laps)
    assert sorted(zip(stops["driver"], stops["lap"], strict=True)) == [("d3", 8), ("d5", 13)]


def test_match_counts_hits_within_one_lap() -> None:
    real = pd.DataFrame({"driver": ["d3", "d5", "d7"], "lap": [8, 12, 30]})
    inferred = pd.DataFrame({"driver": ["d3", "d5", "d9"], "lap": [8, 13, 4]})
    m = match(real, inferred)
    assert m["hits"] == 2 and m["precision"] == round(2 / 3, 3) and m["recall"] == round(2 / 3, 3)
