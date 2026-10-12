"""A scripted client driving a race through a ``WorldRunner``: commands at fixed ticks with
stable idempotency keys, steps between them while paused. Used by the crash-recovery test in
this process and in a fresh child process (so it must not depend on pytest fixtures)."""

import hashlib
import json
from pathlib import Path
from typing import Any

from f1ar.domain.race_state import RaceSetup, WorldState, WorldStatus
from f1ar.persistence.snapshot_codec import state_hash
from f1ar.persistence.sqlite_store import SnapshotStore
from f1ar.simulation import lightweight_engine as engine
from f1ar.simulation.scenarios import build_setup, load_event
from f1ar.simulation.world_runner import (
    AutosavePolicy,
    Command,
    CommandAck,
    WorldRunner,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CODES = ("NOR", "PIA", "LEC", "SAI", "HAM", "VER")
HAM, VER = "lewis_hamilton_2024", "max_verstappen_2024"
POLICY = AutosavePolicy(every_laps=5, keep=10)

# (tick, idempotency key, command fields)
SCRIPT: list[tuple[int, str, dict[str, Any]]] = [
    (1500, "user-pit-ham", {"type": "PIT_REQUEST", "driver_id": HAM, "compound": "HARD"}),
    (2600, "user-sc", {"type": "DEPLOY_SC", "laps": 3}),
    (4000, "user-push-ver", {"type": "SET_PACE_MODE", "driver_id": VER, "pace_mode": "PUSH"}),
]


def make_setup() -> RaceSetup:
    roster, profiles = load_event(REPO_ROOT / "configs")
    ids = [e.driver_id for e in roster.entries if e.code in CODES]
    return build_setup(roster, profiles, ids)


def new_world() -> WorldState:
    return engine.create_world(make_setup(), seed=7, world_id="world_scripted")


def _send(runner: WorldRunner, key: str, **fields: Any) -> CommandAck:
    future = runner.submit(Command.model_validate({"idempotency_key": key, **fields}))
    runner.pump(0.0)
    return future.result(timeout=5)


def _step_to(runner: WorldRunner, target: int | None) -> None:
    """Step (paused) to ``target`` tick, or to the end of the race when ``target`` is None."""
    while runner.state.world_status is WorldStatus.PAUSED:
        tick = runner.state.sim_tick
        if target is not None and tick >= target:
            return
        ticks = 10_000 if target is None else min(10_000, target - tick)
        _send(runner, f"step-{tick}-{target}", type="STEP", ticks=ticks)


def drive(runner: WorldRunner, *, crash_at: int | None = None) -> None:
    """Play the client script from wherever the world is now. Re-sent commands that the
    world already applied come back as duplicates and change nothing."""
    _send(runner, "user-start", type="START")
    _send(runner, "user-pause", type="PAUSE")
    for tick, key, fields in SCRIPT:
        if crash_at is not None and tick > crash_at:
            break
        _step_to(runner, tick)
        _send(runner, key, **fields)
    if crash_at is not None:
        _step_to(runner, crash_at)
        return  # the "crash": no shutdown, no flush
    _step_to(runner, None)


def runner_for(state: WorldState, store: SnapshotStore, flush_every_ticks: int) -> WorldRunner:
    return WorldRunner(
        state,
        clock=lambda: 0.0,
        store=store,
        autosave=POLICY,
        flush_every_ticks=flush_every_ticks,
    )


LINEAGE_FIELDS = {"timeline_id", "parent_timeline_id", "branch_event_seq"}


def race_hash(state: WorldState) -> str:
    """Hash of everything about the race except which timeline it is (a recovered branch is
    the same race under a new timeline id)."""
    return state_hash(state.model_dump(mode="json", exclude=LINEAGE_FIELDS))


def digest(events: list[Any]) -> str:
    rows = [[e.seq, e.tick, e.type, e.driver_id, e.payload] for e in events]
    return hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()


def recover(db: Path, world_id: str, flush_every_ticks: int) -> dict[str, Any]:
    """What a restarted app does: restore the latest valid save, replay the client script."""
    with SnapshotStore(db) as store:
        meta = store.latest_valid(world_id)
        assert meta is not None
        state = store.resume(meta.snapshot_id)
        runner = runner_for(state, store, flush_every_ticks)
        drive(runner)
        runner.shutdown()
        final = runner.state
        return {
            "race_hash": race_hash(final),
            "events_digest": digest(store.events(final.timeline_id)),
            "event_count": len(store.events(final.timeline_id)),
            "restored_tick": state.sim_tick,
            "restored_label": meta.label,
            "branched": state.timeline_id != meta.timeline_id,
            "ham_stops": final.car(HAM).pit_stops,
        }


if __name__ == "__main__":
    import sys

    print(json.dumps(recover(Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]))))
