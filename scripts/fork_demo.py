"""Alternate-reality demo: save a race at lap N, fork it into branches with different decisions.

Usage:
    uv run python scripts/fork_demo.py                         # VER vs HAM, HAM decides at lap 18
    uv run python scripts/fork_demo.py --driver VER --at-lap 40
    uv run python scripts/fork_demo.py --drivers VER HAM LEC NOR --driver LEC --at-lap 30

Branches (all from the same saved snapshot, same luck = common random numbers):
    main      the driver keeps BALANCED and the baseline strategy (the original timeline)
    push      the driver switches to PUSH (faster now, wears tyres faster; the plan adapts)
    conserve  the driver switches to CONSERVE (slower now, saves tyres; the plan adapts)
    box now   the team pits at the end of the current lap for the compound its plan intended,
              then decides stops itself (none, unless the tyre rule still requires one)
"""

import argparse
import uuid
from pathlib import Path

from f1ar.domain.race_state import (
    Compound,
    DriverAction,
    PaceMode,
    RequestPitStop,
    SetPaceMode,
    WorldState,
    WorldStatus,
)
from f1ar.persistence.sqlite_store import SnapshotStore
from f1ar.simulation import lightweight_engine as engine
from f1ar.simulation.scenarios import event_data_dir, load_event, setup_from_codes
from f1ar.simulation.timing import result_gap

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
DEFAULT_DB = ROOT / "saves" / "fork_demo.sqlite"


def _run_to_end(state: WorldState, driver_id: str) -> tuple[WorldState, list[str]]:
    """Finish the race; also returns ``driver_id``'s stops as ``L<lap> <compound>``."""
    stops: list[str] = []
    while state.world_status is WorldStatus.RUNNING:
        result = engine.step(state)
        state = result.state
        stops += [
            f"L{e.payload['lap']} {str(e.payload['to_compound'])[0]}"
            for e in result.events
            if e.type == "PIT_STOP" and e.driver_id == driver_id
        ]
    return state, stops


def main() -> None:
    parser = argparse.ArgumentParser(description="Fork a saved race into alternate timelines")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--drivers", nargs="+", default=["VER", "HAM"], metavar="CODE")
    parser.add_argument("--driver", default="HAM", help="driver whose decision changes")
    parser.add_argument("--at-lap", type=int, default=18, help="branch after the leader's lap N")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument(
        "--synthetic", action="store_true", help="ignore fetched data: equal cars, all on MEDIUM"
    )
    args = parser.parse_args()

    roster, profiles = load_event(CONFIGS)
    setup = setup_from_codes(
        roster, profiles, args.drivers, data_dir=None if args.synthetic else event_data_dir(ROOT)
    )
    data_note = ", ".join(setup.inputs) if setup.inputs else "synthetic defaults (no data)"
    print(f"inputs: {data_note}")
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

    car = state.car(driver_id)
    box_compound = car.planned_stops[0].compound if car.planned_stops else Compound.HARD
    decisions: list[tuple[str, str, DriverAction]] = [
        ("push", f"{code} -> PUSH", SetPaceMode(driver_id=driver_id, pace_mode=PaceMode.PUSH)),
        (
            "conserve",
            f"{code} -> CONSERVE",
            SetPaceMode(driver_id=driver_id, pace_mode=PaceMode.CONSERVE),
        ),
        (
            "box now",
            f"{code} boxes for {box_compound.value}",
            RequestPitStop(driver_id=driver_id, compound=box_compound),
        ),
    ]

    with SnapshotStore(args.db) as store:
        source = store.save(state, label=f"lap {args.at_lap}, before the decision")
        print(f"[SAVED] main timeline at lap {args.at_lap}, tick {state.sim_tick}: ", end="")
        print(source.snapshot_id)

        branches: dict[str, tuple[str, WorldState]] = {}
        for name, decision, action in decisions:
            branch, _ = store.fork(source.snapshot_id, label=f"{decision} after lap {args.at_lap}")
            branch = engine.apply_action(engine.resume(branch), action).state
            branches[name] = (decision, branch)
            print(f"[FORK]  {name:<8} -> {branch.timeline_id} (parent {branch.parent_timeline_id})")
        main_state = engine.resume(store.load(source.snapshot_id))
        branches = {"main": (f"{code} follows the plan", main_state), **branches}

        results: dict[str, tuple[WorldState, list[str]]] = {}
        for name, (_, start) in branches.items():
            final, stops = _run_to_end(start, driver_id)
            store.save(final, label=f"{name}: finished")
            results[name] = (final, stops)

        unchanged = store.get_meta(source.snapshot_id).payload_sha256 == source.payload_sha256
        unchanged = unchanged and store.load(source.snapshot_id) == state
        timelines = store.list_timelines(world_id)

    print(f"\n{code}'s baseline plan at the branch point: {engine.plan_text(car.planned_stops)}")
    print(f"Same snapshot, same luck, four decisions for {code} from lap {args.at_lap + 1}:\n")
    main_time = results["main"][0].car(driver_id).finish_time_s or 0.0
    for name, (final, stops) in results.items():
        decision = branches[name][0]
        order = engine.standings(final)
        cells = [code_of[order[0]]] + [
            f"{code_of[d]} {result_gap(final, d, order[0])}" for d in order[1:]
        ]
        t = final.car(driver_id).finish_time_s or 0.0
        delta = "" if name == "main" else f"   {code} {t - main_time:+.3f}s vs main"
        print(
            f"  {name:<9} {decision:<24} {'  '.join(cells)}{delta}"
            f"   [{code} stops: {', '.join(stops) or 'none'}]"
        )

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
