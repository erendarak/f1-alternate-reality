"""Run the M3 backend: REST + WebSocket API over the save file (docs/API_CONTRACTS.md).

Usage:
    uv run python scripts/fetch_event.py --season 2024 --round 16   # once per race
    uv run python scripts/serve.py                                  # http://127.0.0.1:8000/docs
    uv run python scripts/serve.py --db saves/other.sqlite --port 8001

Worlds are kept in the save file; the backend starts with no live world (create or load one).
Stopping the server (Ctrl+C) pauses, autosaves and writes the logs of the live world.
"""

import argparse
from pathlib import Path

import uvicorn

from f1ar.api.catalog import load_catalog
from f1ar.api.main import create_api
from f1ar.persistence.sqlite_store import SnapshotStore

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--db", type=Path, default=ROOT / "saves" / "worlds.sqlite")
    args = parser.parse_args()

    catalog, skipped = load_catalog(ROOT)
    for race in catalog.all():
        inputs = "data inputs" if race.has_data else "synthetic"
        print(f"race {race.race_id}: {race.name} ({inputs})")
    for race_id, reason in skipped.items():
        print(f"race {race_id}: unavailable, {reason}")
    args.db.parent.mkdir(parents=True, exist_ok=True)
    store = SnapshotStore(args.db)
    app = create_api(store, catalog, skipped_races=skipped)
    print(f"F1AR API -> http://{args.host}:{args.port}/docs   save file: {args.db}")
    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    finally:
        store.close()


if __name__ == "__main__":
    main()
