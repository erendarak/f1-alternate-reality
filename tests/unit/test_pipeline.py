"""Pure parts of the generic fetch pipeline (no network)."""

from pathlib import Path

import pytest

from f1ar.data.fastf1_adapter import normalize_laps
from f1ar.data.pipeline import EventPaths, event_id_for, typical_lap
from tests.unit.test_fastf1_data import raw_laps


@pytest.mark.parametrize(
    ("season", "name", "expected"),
    [
        (2024, "Italian Grand Prix", "2024_italian_gp"),
        (2024, "Belgian Grand Prix", "2024_belgian_gp"),
        (2024, "São Paulo Grand Prix", "2024_sao_paulo_gp"),
        (2023, "Las Vegas Grand Prix", "2023_las_vegas_gp"),
    ],
)
def test_event_ids(season: int, name: str, expected: str) -> None:
    assert event_id_for(season, name) == expected


def test_event_paths(tmp_path: Path) -> None:
    p = EventPaths(tmp_path, "2024_belgian_gp")
    assert p.processed.name == "2024_belgian_gp"
    assert p.manifest.name == "2024_belgian_gp_fastf1.yaml"
    assert p.report.name == "2024_belgian_gp_quality.md"


def test_typical_lap_is_the_median_representative_lap() -> None:
    laps = normalize_laps(raw_laps(IsAccurate=True))
    laps.loc[laps["lap_number"] == 4, "lap_time_s"] = 85.4
    # car 16: lap 1 = start, 2 = in-lap, 3 = out-lap -> only lap 4 is representative
    assert typical_lap(laps, 16) == 4
    with pytest.raises(ValueError):
        typical_lap(laps, 99)
