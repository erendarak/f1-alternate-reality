"""Run a headless race and print a lap-by-lap timing sheet; optionally pause, save, reload, resume.

Usage:
    uv run python scripts/demo_headless.py                     # VER vs HAM, seed 42
    uv run python scripts/demo_headless.py --drivers VER HAM LEC NOR --seed 7
    uv run python scripts/demo_headless.py --all               # full 20-car grid

    # M0 acceptance gate: pause at lap 18, save to SQLite, reload in a FRESH process,
    # resume to the flag, and compare with an uninterrupted run of the same seed.
    uv run python scripts/demo_headless.py --seed 42 --pause --save --reload --resume

    # continue any saved snapshot (see scripts/inspect_save.py for ids)
    uv run python scripts/demo_headless.py --resume-from snap_... --db saves/demo.sqlite

The race model is a synthetic placeholder (no pit stops, no traffic, uncalibrated).
"""

import argparse
import subprocess
import sys
from pathlib import Path

from f1ar.domain.race_state import WorldState, WorldStatus
from f1ar.persistence.sqlite_store import SnapshotStore
from f1ar.simulation import lightweight_engine as engine
from f1ar.simulation.scenarios import MONZA_2024, load_event, setup_from_codes
from f1ar.simulation.timing import LapTimingSheet

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
DEFAULT_DB = ROOT / "saves" / "demo.sqlite"
DEFAULT_PAUSE_LAP = 18
FINAL_HASH_PREFIX = "FINAL_STATE_HASH "


def _leader_lap(state: WorldState) -> int:
    return max(c.laps_completed for c in state.cars)


def _run(
    state: WorldState, sheet: LapTimingSheet | None, *, stop_after_lap: int | None
) -> WorldState:
    """Step until the race ends or the leader has completed ``stop_after_lap`` laps."""
    while state.world_status is WorldStatus.RUNNING:
        if stop_after_lap is not None and _leader_lap(state) >= stop_after_lap:
            break
        result = engine.step(state)
        state = result.state
        if sheet is not None:
            for row in sheet.feed(result.events):
                print(row, flush=True)
    return state


def _print_classification(state: WorldState) -> None:
    setup = state.setup
    print("\nFinal classification")
    order = engine.standings(state)
    winner_t = state.car(order[0]).finish_time_s or 0.0
    for pos, driver_id in enumerate(order, start=1):
        t = state.car(driver_id).finish_time_s or 0.0
        gap = f"{t // 60:.0f}:{t % 60:06.3f}" if pos == 1 else f"+{t - winner_t:.3f}s"
        entry = setup.driver(driver_id)
        print(f"  P{pos:<2} {entry.code}  {gap:>12}   (started P{entry.grid_position})")
    print(f"\nticks {state.sim_tick} | events {state.event_seq}")
    print(f"{FINAL_HASH_PREFIX}{state.canonical_hash()}", flush=True)


def _sheet(state: WorldState, every: int) -> LapTimingSheet:
    code_of = {d.driver_id: d.code for d in state.setup.drivers}
    return LapTimingSheet(code_of, state.setup.track.total_laps, every=every)


def _resume_from(db: Path, snapshot_id: str, every: int) -> None:
    with SnapshotStore(db) as store:
        state = store.load(snapshot_id)  # verifies checksum + versions
    print(
        f"[LOADED {snapshot_id}] tick {state.sim_tick}, leader lap {_leader_lap(state)}, "
        f"status {state.world_status}, hash {state.canonical_hash()[:23]}...",
        flush=True,
    )
    if state.world_status is WorldStatus.PAUSED:
        state = engine.resume(state)
        print("[RESUMED]", flush=True)
    _print_classification(_run(state, _sheet(state, every), stop_after_lap=None))


def main() -> None:
    parser = argparse.ArgumentParser(description="Headless race demo")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--drivers", nargs="+", default=["VER", "HAM"], metavar="CODE")
    parser.add_argument("--all", action="store_true", help="race the full 20-car grid")
    parser.add_argument("--every", type=int, default=1, help="print every Nth lap")
    parser.add_argument("--pause", action="store_true", help=f"pause at lap {DEFAULT_PAUSE_LAP}")
    parser.add_argument("--pause-at-lap", type=int, help="pause once the leader completes lap N")
    parser.add_argument("--save", action="store_true", help="save the paused world to SQLite")
    parser.add_argument("--reload", action="store_true", help="reload the save in a new process")
    parser.add_argument("--resume", action="store_true", help="resume to the flag after pausing")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--resume-from", metavar="SNAPSHOT_ID", help="load a snapshot and finish")
    args = parser.parse_args()

    if args.resume_from:
        _resume_from(args.db, args.resume_from, args.every)
        return

    pause_lap = args.pause_at_lap or (DEFAULT_PAUSE_LAP if args.pause else None)
    if (args.save or args.reload or args.resume) and pause_lap is None:
        parser.error("--save/--reload/--resume need --pause or --pause-at-lap")
    if args.reload and not (args.save and args.resume):
        parser.error("--reload needs --save (something to reload) and --resume")

    roster, profiles = load_event(CONFIGS)
    try:
        setup = setup_from_codes(roster, profiles, None if args.all else args.drivers)
    except ValueError as exc:
        parser.error(str(exc))
    world = engine.start(engine.create_world(setup, seed=args.seed, world_id="demo"))

    print(
        f"{MONZA_2024} | {len(setup.drivers)} cars | {setup.track.total_laps} laps"
        f" | seed {args.seed}"
    )
    print("grid: " + "  ".join(f"P{d.grid_position} {d.code}" for d in setup.drivers))
    print(f"{engine.ENGINE_VERSION}: synthetic lap model, no pit stops, no traffic\n")

    sheet = _sheet(world, args.every)
    state = _run(world, sheet, stop_after_lap=pause_lap)
    if pause_lap is None or state.world_status is not WorldStatus.RUNNING:
        _print_classification(state)
        return

    state = engine.pause(state)
    order = " ".join(setup.driver(d).code for d in engine.standings(state))
    print(f"\n[PAUSED] leader completed lap {pause_lap} | tick {state.sim_tick} | order: {order}")

    snapshot_id = None
    if args.save:
        with SnapshotStore(args.db) as store:
            meta = store.save(state, label=f"demo seed {args.seed}, lap {pause_lap}")
        snapshot_id = meta.snapshot_id
        print(
            f"[SAVED] {snapshot_id} -> {args.db} "
            f"({meta.payload_bytes} bytes, {meta.payload_sha256[:23]}...)"
        )

    if not args.resume:
        print("Stopped while paused. Continue later with --resume-from <snapshot id>.")
        return

    if args.reload:
        assert snapshot_id is not None
        print("[RELOAD] starting a fresh Python process to load the snapshot...\n", flush=True)
        child = subprocess.run(
            [
                sys.executable,
                __file__,
                "--resume-from",
                snapshot_id,
                "--db",
                str(args.db),
                "--every",
                str(args.every),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
        )
        print(child.stdout, end="")
        resumed_hash = next(
            line.removeprefix(FINAL_HASH_PREFIX)
            for line in child.stdout.splitlines()
            if line.startswith(FINAL_HASH_PREFIX)
        )
    else:
        state = _run(engine.resume(state), sheet, stop_after_lap=None)
        _print_classification(state)
        resumed_hash = state.canonical_hash()

    # Reference: the same race with no pause, save or reload at all.
    reference = _run(
        engine.start(engine.create_world(setup, seed=args.seed, world_id="demo")),
        None,  # silent
        stop_after_lap=None,
    ).canonical_hash()
    verdict = "MATCH" if resumed_hash == reference else "MISMATCH"
    print(f"\nDeterminism check vs uninterrupted run: {verdict}")
    print(f"  uninterrupted: {reference}\n  interrupted:   {resumed_hash}")
    if verdict != "MATCH":
        sys.exit(1)


if __name__ == "__main__":
    main()
