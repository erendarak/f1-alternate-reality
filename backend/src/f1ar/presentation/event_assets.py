"""Drawable assets of a fetched event: track outline, corners and team colours (display only).

Shared by the debug viewer and the M3 API. Everything here comes from the event's fetched
``data/processed/<event>/`` files (FastF1-derived); nothing feeds back into the race.
"""

from collections.abc import Hashable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from f1ar.domain.identities import RaceRoster
from f1ar.presentation.track_geometry import TrackGeometry
from f1ar.presentation.view_model import track_view

ASSET_FILES = ("track_outline.parquet", "corners.parquet", "results.parquet")


class MissingEventAssetsError(FileNotFoundError):
    """The event hasn't been fetched (``scripts/fetch_event.py``)."""


@dataclass(frozen=True)
class EventAssets:
    geometry: TrackGeometry
    track_payload: dict[str, Any]
    team_colors: dict[str, str]
    """driver_id -> hex colour (no '#')."""


def missing_assets(data_dir: Path) -> list[str]:
    return [name for name in ASSET_FILES if not (data_dir / name).exists()]


def load_event_assets(
    roster: RaceRoster, data_dir: Path, *, engine_length_m: float | None
) -> EventAssets:
    """Outline, corners and colours of ``roster``'s event from its processed data directory.

    ``engine_length_m`` is the official length the engine uses (None for replay-only events).
    """
    missing = missing_assets(data_dir)
    if missing:
        raise MissingEventAssetsError(
            f"missing {missing} in {data_dir}. Run first:\n"
            "    uv run python scripts/fetch_event.py --season <year> --round <n>"
        )
    outline = pd.read_parquet(data_dir / "track_outline.parquet")
    corners = pd.read_parquet(data_dir / "corners.parquet")
    results = pd.read_parquet(data_dir / "results.parquet")

    geometry = TrackGeometry.from_points(
        list(zip(outline["x_m"].astype(float), outline["y_m"].astype(float), strict=True))
    )
    rotation = float(corners["display_rotation_deg"].iloc[0]) if len(corners) else 0.0
    corner_records: list[dict[Hashable, Any]] = corners.to_dict("records")
    corner_rows = [
        {
            "number": int(r["number"]),
            "letter": None if pd.isna(r["letter"]) else str(r["letter"]),
            "x_m": float(r["x_m"]),
            "y_m": float(r["y_m"]),
            "angle_deg": float(r["angle_deg"]),
        }
        for r in corner_records
    ]
    payload = track_view(
        geometry,
        engine_length_m=engine_length_m,
        corners=corner_rows,
        rotation_deg=rotation,
        source=f"FastF1 {roster.event_id}: winner's typical-lap path (derived); circuit corners",
    )
    by_number = {e.car_number: e.driver_id for e in roster.entries}
    result_records: list[dict[Hashable, Any]] = results.to_dict("records")
    colors = {
        by_number[int(r["driver_number"])]: str(r["team_color"])
        for r in result_records
        if int(r["driver_number"]) in by_number and not pd.isna(r["team_color"])
    }
    return EventAssets(geometry, payload, colors)
