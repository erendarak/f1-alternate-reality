"""M0.5 debug viewer: watch the live simulation on the real Monza outline in your browser.

Usage:
    uv run python scripts/fetch_monza_2024.py        # once: builds the outline/corners/colours
    uv run python scripts/debug_viewer.py            # then open http://127.0.0.1:8765
    uv run python scripts/debug_viewer.py --drivers VER HAM LEC NOR --speed 5
    uv run python scripts/debug_viewer.py --all --seed 7
    uv run python scripts/debug_viewer.py --replay      # the REAL 2024 race, recorded history

The page is a read-only view of backend state; every button sends a command to the runner.
"""

import argparse
import sys
from collections.abc import Hashable
from pathlib import Path
from typing import Any

import pandas as pd
import uvicorn

from f1ar.api.debug_app import create_app
from f1ar.api.replay_app import create_replay_app
from f1ar.domain.identities import RaceRoster
from f1ar.domain.race_state import WorldState
from f1ar.history.reference import HistoricalRace
from f1ar.history.replay import REPLAY_SPEEDS, ReplayClock
from f1ar.persistence.sqlite_store import SnapshotStore
from f1ar.presentation.track_geometry import TrackGeometry
from f1ar.presentation.view_model import track_view
from f1ar.simulation import lightweight_engine as engine
from f1ar.simulation.scenarios import MONZA_2024, load_event, setup_from_codes
from f1ar.simulation.world_runner import ALLOWED_SPEEDS

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / MONZA_2024
CAR_SPRITE = ROOT / "assets" / "third_party" / "palm-tree-f1-pack" / "Ferrari" / "Ferrari.png"


def load_assets(roster: RaceRoster) -> tuple[TrackGeometry, dict[str, Any], dict[str, str]]:
    needed = [DATA / f for f in ("track_outline.parquet", "corners.parquet", "results.parquet")]
    missing = [p.name for p in needed if not p.exists()]
    if missing:
        sys.exit(
            f"missing {missing} in {DATA}. Run first:\n"
            "    uv run python scripts/fetch_monza_2024.py"
        )
    outline = pd.read_parquet(needed[0])
    corners = pd.read_parquet(needed[1])
    results = pd.read_parquet(needed[2])

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
        engine_length_m=5793.0,
        corners=corner_rows,
        rotation_deg=rotation,
        source="FastF1 2024 Italian GP: winner's lap-30 path (derived); corners from circuit info",
    )
    by_number = {e.car_number: e.driver_id for e in roster.entries}
    result_records: list[dict[Hashable, Any]] = results.to_dict("records")
    colors = {
        by_number[int(r["driver_number"])]: str(r["team_color"])
        for r in result_records
        if int(r["driver_number"]) in by_number and not pd.isna(r["team_color"])
    }
    return geometry, payload, colors


def main() -> None:
    parser = argparse.ArgumentParser(description="F1AR debug viewer")
    parser.add_argument("--drivers", nargs="+", default=["VER", "HAM", "LEC", "NOR", "PIA", "SAI"])
    parser.add_argument("--all", action="store_true", help="full 20-car grid")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--speed", type=int, default=5, choices=sorted({*ALLOWED_SPEEDS, *REPLAY_SPEEDS})
    )
    parser.add_argument("--replay", action="store_true", help="replay the real recorded race")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--db", type=Path, default=ROOT / "saves" / "viewer.sqlite")
    args = parser.parse_args()

    roster, profiles = load_event(ROOT / "configs")
    geometry, payload, colors = load_assets(roster)
    sprite = CAR_SPRITE if CAR_SPRITE.exists() else None
    if sprite is None:
        print(f"note: {CAR_SPRITE.relative_to(ROOT)} not found, drawing dots instead of cars")

    if args.replay:
        race = HistoricalRace.from_processed_dir(MONZA_2024, DATA)
        clock = ReplayClock(
            race.duration_s, speed=args.speed if args.speed in REPLAY_SPEEDS else 10
        )
        replay = create_replay_app(
            race, clock, geometry=geometry, track_payload=payload, car_sprite=sprite
        )
        print(f"F1AR REAL-RACE REPLAY -> http://127.0.0.1:{args.port}  (Ctrl+C to stop)")
        uvicorn.run(replay, host="127.0.0.1", port=args.port, log_level="warning")
        return
    if args.speed not in ALLOWED_SPEEDS:
        parser.error(f"simulation speed must be one of {ALLOWED_SPEEDS}")
    setup = setup_from_codes(roster, profiles, None if args.all else args.drivers)
    store = SnapshotStore(args.db)

    def new_world(seed: int, world_id: str) -> WorldState:
        return engine.create_world(setup, seed=seed, world_id=world_id)

    app = create_app(
        new_world,
        seed=args.seed,
        speed=args.speed,
        geometry=geometry,
        track_payload=payload,
        team_colors=colors,
        store=store,
        car_sprite=sprite,
    )

    print(f"F1AR debug viewer -> http://127.0.0.1:{args.port}  (Ctrl+C to stop)")
    try:
        uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
    finally:
        store.close()


if __name__ == "__main__":
    main()
