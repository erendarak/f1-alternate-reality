"""Alternate-reality demo: save a race at lap N, fork it into branches with different decisions.

Usage:
    uv run python scripts/fork_demo.py                         # VER vs HAM, HAM decides at lap 18
    uv run python scripts/fork_demo.py --driver VER --at-lap 40
    uv run python scripts/fork_demo.py --drivers VER HAM LEC NOR --driver LEC --at-lap 30

Branches (all from the same saved snapshot, same luck = common random numbers):
    main      the driver keeps BALANCED (the original timeline)
    push      the driver switches to PUSH  (faster now, wears tyres faster)
    conserve  the driver switches to CONSERVE (slower now, saves tyres)
There are no pit stops yet, so tyres must last to the flag in every branch.
"""

import argparse
import uuid
from pathlib import Path

from f1ar.domain.race_state import PaceMode, SetPaceMode, WorldState, WorldStatus
from f1ar.persistence.sqlite_store import SnapshotStore
from f1ar.simulation import lightweight_engine as engine
from f1ar.simulation.scenarios import load_event, setup_from_codes

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
DEFAULT_DB = ROOT / "saves" / "fork_demo.sqlite"


def _run_to_end(state: WorldState) -> WorldState:
    while state.world_status is WorldStatus.RUNNING:
        state = engine.step(state).state
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description="Fork a saved race into alternate timelines")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--drivers", nargs="+", default=["VER", "HAM"], metavar="CODE")
    parser.add_argument("--driver", default="HAM", help="driver whose decision changes")
    parser.add_argument("--at-lap", type=int, default=18, help="branch after the leader's lap N")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    args = parser.parse_args()

    roster, profiles = load_event(CONFIGS)
    setup = setup_from_codes(roster, profiles, args.drivers)
    code_of = {d.driver_id: d.code for d in setup.drivers}
    by_code = {c: d for d, c in code_of.items()}
    if args.driver.upper() not in by_code:
        parser.error(f"--driver must be one of {sorted(by_code)}")
    driver_id = by_code[args.driver.upper()]
    code = code_of[driver_id]

    world_id = f"fork_demo_{uuid.uuid4().hex[:8]}"
    state = engine.start(engine.create_world(setup, seed=args.seed, world_id=world_id))
    while max(c.laps_completed for c in state.cars) < args.at_lap:
        state = engine.step(state).state
    state = engine.pause(state)

    with SnapshotStore(args.db) as store:
        source = store.save(state, label=f"lap {args.at_lap}, before the decision")
        print(f"[SAVED] main timeline at lap {args.at_lap}, tick {state.sim_tick}: ", end="")
        print(source.snapshot_id)

        branches: dict[str, tuple[str, WorldState]] = {}
        for name, mode in (("push", PaceMode.PUSH), ("conserve", PaceMode.CONSERVE)):
            branch, _ = store.fork(
                source.snapshot_id, label=f"{code} {mode} from lap {args.at_lap}"
            )
            branch = engine.resume(branch)
            branch = engine.apply_action(
                branch, SetPaceMode(driver_id=driver_id, pace_mode=mode)
            ).state
            branches[name] = (f"{code} -> {mode}", branch)
            print(f"[FORK]  {name:<8} -> {branch.timeline_id} (parent {branch.parent_timeline_id})")
        main_state = engine.resume(store.load(source.snapshot_id))
        branches = {"main": (f"{code} stays BALANCED", main_state), **branches}

        results: dict[str, WorldState] = {}
        for name, (_, start) in branches.items():
            final = _run_to_end(start)
            store.save(final, label=f"{name}: finished")
            results[name] = final

        unchanged = store.get_meta(source.snapshot_id).payload_sha256 == source.payload_sha256
        unchanged = unchanged and store.load(source.snapshot_id) == state
        timelines = store.list_timelines(world_id)

    print(f"\nSame snapshot, same luck, three decisions for {code} from lap {args.at_lap + 1}:\n")
    main_time = results["main"].car(driver_id).finish_time_s or 0.0
    for name, final in results.items():
        decision = branches[name][0]
        order = engine.standings(final)
        winner_t = final.car(order[0]).finish_time_s or 0.0
        cells = [code_of[order[0]]] + [
            f"{code_of[d]} +{(final.car(d).finish_time_s or 0.0) - winner_t:.3f}" for d in order[1:]
        ]
        t = final.car(driver_id).finish_time_s or 0.0
        delta = "" if name == "main" else f"   {code} {t - main_time:+.3f}s vs main"
        print(f"  {name:<9} {decision:<22} {'  '.join(cells)}{delta}")

    print("\nTimeline tree:")
    for tl in timelines:
        parent = (
            "" if tl.parent_timeline_id is None else f"  <- forked from {tl.parent_timeline_id}"
        )
        print(f"  {tl.timeline_id}  [{tl.label}]  {tl.snapshot_count} snapshot(s){parent}")
    print(f"\nSource snapshot unchanged after all branches ran: {'YES' if unchanged else 'NO'}")
    print(f"Saved to {args.db} (world {world_id}); inspect with scripts/inspect_save.py --db ...")


if __name__ == "__main__":
    main()
