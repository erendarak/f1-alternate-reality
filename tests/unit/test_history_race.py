"""H2b (D-055): 2010-2017 races from the Jolpica history in the common race tables."""

from typing import Any

import pandas as pd

from f1ar.calibration.tyre_life_fit import DRY, WearFit, with_reference
from f1ar.data.history_race import WET_LAP_BAND, unique_grid, wet_laps


def _result(code: str, grid: int, number: int) -> dict[str, Any]:
    return {"grid": str(grid), "number": str(number), "Driver": {"code": code, "driverId": code}}


def test_a_shared_grid_slot_is_resolved_by_qualifying_and_moves_only_who_must() -> None:
    # 2015 Italian GP in Jolpica: SAI and NAS both on grid 17
    results = [
        _result("HAM", 1, 44),
        _result("SAI", 17, 55),
        _result("NAS", 17, 12),
        _result("KVY", 18, 26),
        _result("VER", 20, 33),
        _result("RIC", 0, 3),  # pit lane: untouched
    ]
    notes = unique_grid(results, {"NAS": 16, "SAI": 18})
    grid = {r["Driver"]["code"]: int(r["grid"]) for r in results}
    assert grid == {"HAM": 1, "NAS": 17, "SAI": 18, "KVY": 19, "VER": 20, "RIC": 0}
    assert notes[0].startswith("SAI grid 17 -> 18 (shares its slot")
    assert "behind a shared slot" in notes[1]
    assert unique_grid([_result("A", 1, 1), _result("B", 2, 2)], {}) == []


def _laps(medians: list[float]) -> pd.DataFrame:
    rows = []
    for lap, m in enumerate(medians, start=2):
        rows += [{"lap": lap, "lap_time_s": m + d} for d in (-0.4, 0.0, 0.4)]
    return pd.DataFrame(rows)


def test_wet_laps_are_moderately_slow_and_need_a_share_of_the_race() -> None:
    lo, hi = WET_LAP_BAND
    dry = [90.0] * 40
    assert wet_laps(_laps(dry)) == set()
    # 6 rain laps (+15 %) and 3 safety-car laps (+45 %) in a 40-lap race
    wet = dry[:10] + [103.5] * 6 + [130.0] * 3 + dry[19:]
    found = wet_laps(_laps(wet))
    assert found == set(range(12, 18))
    assert all(lo < 103.5 / 89.6 < hi for _ in found)
    # two slow laps only (under 10 % of the race): not a wet race
    assert wet_laps(_laps(dry[:20] + [103.5] * 2 + dry[22:])) == set()


def test_a_season_without_compounds_takes_the_reference_ratios() -> None:
    season = WearFit(
        lin={"MEDIUM": 0.05}, quad={"MEDIUM": -0.0004}, fresh_bonus_s=-0.3, offsets={}, laps=900
    )
    reference = {
        "season_wear": {
            "lin": {"SOFT": 0.06, "MEDIUM": 0.04, "HARD": 0.03},
            "quad": {"SOFT": -0.0006, "MEDIUM": -0.0004, "HARD": -0.0002},
        },
        "life_ratios_vs_medium": {"SOFT": 0.8, "MEDIUM": 1.0, "HARD": 1.3},
    }
    filled, ratios = with_reference(season, reference)
    assert set(filled.lin) == set(DRY)
    assert abs(filled.lin["SOFT"] - 0.075) < 1e-12 and abs(filled.lin["HARD"] - 0.0375) < 1e-12
    assert filled.lin["MEDIUM"] == 0.05 and abs(filled.quad["HARD"] + 0.0002) < 1e-12
    assert ratios == {"SOFT": 0.8, "MEDIUM": 1.0, "HARD": 1.3}
