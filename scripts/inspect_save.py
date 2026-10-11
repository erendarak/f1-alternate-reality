"""List the snapshots in a save database and verify each one.

Usage:
    uv run python scripts/inspect_save.py                  # saves/demo.sqlite
    uv run python scripts/inspect_save.py --db path/to/worlds.sqlite
"""

import argparse
from pathlib import Path

from f1ar.persistence.sqlite_store import SnapshotError, SnapshotStore

DEFAULT_DB = Path(__file__).resolve().parents[1] / "saves" / "demo.sqlite"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    args = parser.parse_args()
    if not args.db.exists():
        parser.error(f"{args.db} does not exist")

    with SnapshotStore(args.db) as store:
        snapshots = store.list_snapshots()
        print(f"{args.db}: {len(snapshots)} snapshot(s), newest first\n")
        for m in snapshots:
            try:
                store.load(m.snapshot_id)
                check = "OK"
            except SnapshotError as exc:
                check = f"INVALID ({type(exc).__name__})"
            clock = f"{m.sim_clock_s // 60:.0f}:{m.sim_clock_s % 60:04.1f}"
            print(
                f"{m.snapshot_id}  {m.world_id}/{m.timeline_id}  tick {m.tick:<6} sim {clock:>7}  "
                f"{m.world_status:<9} {m.kind:<8} {m.created_utc[:19]}  [{check}]  {m.label}"
            )


if __name__ == "__main__":
    main()
