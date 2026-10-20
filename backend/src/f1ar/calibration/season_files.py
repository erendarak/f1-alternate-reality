"""Which season a calibration file belongs to (H2a, D-054).

Fits were 2024-only (``<kind>_2024_v1.json``); H2a fits the same models per season
(``<kind>_<season>_v1.json``). A race of another season must not silently take 2024's numbers:

- values keyed by driver or team (race pace, racecraft, pit crews, personas): that season's fit
  or none (the editorial path, labelled as such); 2024's gap of a driver to his teammate says
  nothing about another season;
- tyre life / shape: that season's fit, else 2024's, labelled "borrowed";
- era-level rates (overtaking, race control, weather, wet races) stay 2024's until H2c, labelled.
"""

from pathlib import Path

REFERENCE_SEASON = 2024


def season_path(path: Path, season: int) -> Path:
    """``calibration/driver_pace_2024_v1.json`` -> ``calibration/driver_pace_<season>_v1.json``."""
    marker = f"_{REFERENCE_SEASON}_"
    if marker not in path.name:
        raise ValueError(f"{path} is not a per-season calibration file")
    return path.with_name(path.name.replace(marker, f"_{season}_", 1))


def own_season(configs: Path, path: Path, season: int) -> Path | None:
    """The file of ``season`` if it was fitted, else None (no borrowing)."""
    candidate = configs / season_path(path, season)
    return candidate if candidate.exists() else None


def with_fallback(configs: Path, path: Path, season: int) -> tuple[Path | None, int | None]:
    """The file of ``season``, else 2024's, with the season it belongs to; (None, None)."""
    for s in (season, REFERENCE_SEASON):
        candidate = configs / season_path(path, s)
        if candidate.exists():
            return candidate, s
    return None, None


def dataset_files(root: Path, season: int) -> tuple[Path, Path]:
    """The season's lap dataset and its manifest (``scripts/build_lap_dataset.py``)."""
    return (
        root / "data" / "processed" / "datasets" / f"laps_{season}_v1.parquet",
        root / "data" / "manifests" / f"dataset_laps_{season}_v1.yaml",
    )


def borrowed_note(season: int, used: int | None) -> str:
    if used is None or used == season:
        return ""
    return f" [borrowed from {used}: no {season} fit yet]"


def era_rates_note(season: int) -> str:
    if season == REFERENCE_SEASON:
        return ""
    return f" [{REFERENCE_SEASON} rates applied to {season}; per-era fit in H2c]"
