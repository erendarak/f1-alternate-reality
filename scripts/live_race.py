"""Watch a race in real time and pause/resume it from the keyboard.

Usage:
    uv run python scripts/live_race.py --drivers VER HAM LEC NOR --speed 25

Type a command and press Enter:
    (empty)   pause / resume
    s [N]     step N ticks while paused (default 1; 168 ticks = about one lap)
    1 2 5 10 25   set speed multiplier
    p CODE MODE   pace mode for a driver from their next lap, e.g. 'p HAM push'
    q         quit

The race runs on a background WorldRunner thread. Every keypress becomes a Command that is
applied at the next tick boundary; the printed ack shows the exact tick it took effect.
"""

import argparse
import itertools
from pathlib import Path

from f1ar.domain.race_state import PaceMode, RaceEvent, WorldState, WorldStatus
from f1ar.simulation import lightweight_engine as engine
from f1ar.simulation.scenarios import event_data_dir, load_event, setup_from_codes
from f1ar.simulation.timing import LapTimingSheet
from f1ar.simulation.world_runner import (
    ALLOWED_SPEEDS,
    Command,
    CommandAck,
    CommandType,
    WorldRunner,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
_keys = itertools.count(1)


def _send(runner: WorldRunner, type_: CommandType, **kw: object) -> CommandAck:
    command = Command.model_validate({"idempotency_key": f"cli-{next(_keys)}", "type": type_, **kw})
    return runner.submit(command).result(timeout=5)


def _describe(state: WorldState) -> str:
    leader = state.car(engine.standings(state)[0])
    clock = state.sim_clock_s
    return (
        f"tick {state.sim_tick} | sim {clock // 60:.0f}:{clock % 60:04.1f} | "
        f"leader on lap {min(leader.laps_completed + 1, state.setup.track.total_laps)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Live race with pause/resume")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--drivers", nargs="+", default=["VER", "HAM", "LEC", "NOR"])
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--speed", type=int, default=25, choices=ALLOWED_SPEEDS)
    parser.add_argument(
        "--synthetic", action="store_true", help="ignore fetched data: equal cars, all on MEDIUM"
    )
    args = parser.parse_args()

    roster, profiles = load_event(CONFIGS)
    setup = setup_from_codes(
        roster,
        profiles,
        None if args.all else args.drivers,
        data_dir=None if args.synthetic else event_data_dir(ROOT),
    )
    data_note = ", ".join(setup.inputs) if setup.inputs else "synthetic defaults (no data)"
    print(f"inputs: {data_note}")
    runner = WorldRunner(
        engine.create_world(setup, seed=args.seed, world_id="live"), speed=args.speed
    )
    by_code = {d.code: d.driver_id for d in setup.drivers}
    sheet = LapTimingSheet({d.driver_id: d.code for d in setup.drivers}, setup.track.total_laps)

    def on_tick(state: WorldState, events: tuple[RaceEvent, ...]) -> None:
        for row in sheet.feed(events):
            print(row, flush=True)
        if any(e.type == "RACE_COMPLETED" for e in events):
            order = " ".join(setup.driver(d).code for d in engine.standings(state))
            print(f"\nRACE COMPLETE: {order}\n{_describe(state)}\nPress Enter to exit.", flush=True)

    runner.add_listener(on_tick)
    runner.start_background()
    print(f"x{args.speed} | Enter = pause/resume, 's N' = step, 1/2/5/10/25 = speed,")
    print("      'p HAM push' = pace mode from next lap, q = quit\n")
    _send(runner, CommandType.START)
    try:
        while True:
            line = input().strip().lower()
            status = runner.state.world_status
            if line == "q" or status is WorldStatus.COMPLETED:
                break
            try:
                if line == "":
                    pause_or_resume = (
                        CommandType.RESUME if status is WorldStatus.PAUSED else CommandType.PAUSE
                    )
                    ack = _send(runner, pause_or_resume)
                elif line.startswith("s"):
                    ack = _send(runner, CommandType.STEP, ticks=int(line[1:].strip() or 1))
                elif line.startswith("p "):
                    _, code, mode = line.split()
                    ack = _send(
                        runner,
                        CommandType.SET_PACE_MODE,
                        driver_id=by_code[code.upper()],
                        pace_mode=PaceMode(mode.upper()),
                    )
                elif line.isdigit():
                    ack = _send(runner, CommandType.SET_SPEED, speed=int(line))
                else:
                    print("? unknown command")
                    continue
            except (ValueError, KeyError) as exc:  # bad numbers, modes, driver codes
                print(f"? invalid command: {str(exc).splitlines()[0]}")
                continue
            verdict = "ok" if ack.accepted else f"REJECTED ({ack.reason})"
            status_line = f"{ack.world_status} x{ack.speed} | {_describe(runner.state)}"
            print(f"[{ack.type} {verdict}] {status_line}", flush=True)
    except (EOFError, KeyboardInterrupt):
        pass
    finally:
        runner.stop_background()


if __name__ == "__main__":
    main()
