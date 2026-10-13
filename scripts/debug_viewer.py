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
from pathlib import Path

import uvicorn

from f1ar.api.debug_app import create_app
from f1ar.api.replay_app import create_replay_app
from f1ar.domain.identities import load_roster
from f1ar.domain.race_state import WorldState
from f1ar.history.reference import HistoricalRace
from f1ar.history.replay import REPLAY_SPEEDS, ReplayClock
from f1ar.persistence.sqlite_store import SnapshotStore
from f1ar.presentation.event_assets import MissingEventAssetsError, load_event_assets
from f1ar.simulation import lightweight_engine as engine
from f1ar.simulation.scenarios import (
    MONZA_2024,
    event_data_dir,
    has_personas,
    load_event,
    setup_from_codes,
    track_for,
)
from f1ar.simulation.world_runner import ALLOWED_SPEEDS

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
CAR_SPRITE = ROOT / "assets" / "third_party" / "palm-tree-f1-pack" / "Ferrari" / "Ferrari.png"


def main() -> None:
    parser = argparse.ArgumentParser(description="F1AR debug viewer")
    parser.add_argument("--drivers", nargs="+", default=["VER", "HAM", "LEC", "NOR", "PIA", "SAI"])
    parser.add_argument("--all", action="store_true", help="full 20-car grid")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--speed", type=int, default=5, choices=sorted({*ALLOWED_SPEEDS, *REPLAY_SPEEDS})
    )
    parser.add_argument("--replay", action="store_true", help="replay the real recorded race")
    parser.add_argument(
        "--synthetic", action="store_true", help="ignore fetched data: equal cars, all on MEDIUM"
    )
    parser.add_argument(
        "--event",
        default=MONZA_2024,
        help="event id, e.g. 2024_belgian_gp (simulation: events with personas; replay: any)",
    )
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--db", type=Path, default=ROOT / "saves" / "viewer.sqlite")
    args = parser.parse_args()

    if not args.replay and not has_personas(ROOT / "configs", args.event):
        parser.error(f"no driver personas for {args.event}; use --replay")
    roster_path = ROOT / "configs" / "rosters" / f"{args.event}.json"
    if not roster_path.exists():
        parser.error(f"unknown event {args.event!r}: fetch it first with scripts/fetch_event.py")
    roster = load_roster(roster_path, expected_event_id=args.event)
    data_dir = PROCESSED / args.event
    length = None if args.replay else track_for(args.event, data_dir).length_m
    try:
        assets = load_event_assets(roster, PROCESSED / args.event, engine_length_m=length)
    except MissingEventAssetsError as exc:
        sys.exit(str(exc))
    geometry, payload, colors = assets.geometry, assets.track_payload, assets.team_colors
    sprite = CAR_SPRITE if CAR_SPRITE.exists() else None
    if sprite is None:
        print(f"note: {CAR_SPRITE.relative_to(ROOT)} not found, drawing dots instead of cars")

    if args.replay:
        race = HistoricalRace.from_processed_dir(args.event, PROCESSED / args.event)
        clock = ReplayClock(
            race.duration_s, speed=args.speed if args.speed in REPLAY_SPEEDS else 10
        )
        replay = create_replay_app(
            race, clock, geometry=geometry, track_payload=payload, car_sprite=sprite
        )
        print(f"F1AR REAL-RACE REPLAY -> http://127.0.0.1:{args.port}  (Ctrl+C to stop)")
        uvicorn.run(replay, host="127.0.0.1", port=args.port, log_level="warning")
        return
    _, profiles = load_event(ROOT / "configs", args.event)
    if args.speed not in ALLOWED_SPEEDS:
        parser.error(f"simulation speed must be one of {ALLOWED_SPEEDS}")
    setup = setup_from_codes(
        roster,
        profiles,
        None if args.all else args.drivers,
        data_dir=None if args.synthetic else event_data_dir(ROOT, args.event),
    )
    data_note = ", ".join(setup.inputs) if setup.inputs else "synthetic defaults (no data)"
    print(f"inputs: {data_note}")
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
