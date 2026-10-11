"""Save -> brand-new process -> load -> continue (plan §4.7 tests 1-2, §18.4).

The child process shares nothing with the test process except the SQLite file: fresh
interpreter, fresh imports, fresh hash randomisation.
"""

import subprocess
import sys
from pathlib import Path

from f1ar.domain.race_state import WorldState
from f1ar.persistence.sqlite_store import SnapshotStore
from tests.conftest import REPO_ROOT, run_ticks, run_to_end

_CHILD = """
import sys
from pathlib import Path
from f1ar.domain.race_state import WorldStatus
from f1ar.persistence.sqlite_store import SnapshotStore
from tests.conftest import run_ticks, run_to_end
db, snapshot_id, ticks = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
with SnapshotStore(db) as store:
    state = store.load(snapshot_id)
state = run_to_end(state) if ticks == "end" else run_ticks(state, int(ticks))
print(state.canonical_hash())
"""


def _continue_in_fresh_process(db: Path, snapshot_id: str, ticks: int | str) -> str:
    return subprocess.run(
        [sys.executable, "-c", _CHILD, str(db), snapshot_id, str(ticks)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
        encoding="utf-8",
    ).stdout.strip()


def test_loaded_state_in_fresh_process_equals_saved(
    tmp_path: Path, two_car_world: WorldState
) -> None:
    """Plan §4.7 test 1: save, drop every in-memory object, load elsewhere -> equal."""
    state = run_ticks(two_car_world, 777)
    db = tmp_path / "w.sqlite"
    with SnapshotStore(db) as store:
        snapshot_id = store.save(state, label="A").snapshot_id
    expected = state.canonical_hash()
    del state
    assert _continue_in_fresh_process(db, snapshot_id, 0) == expected


def test_1000_ticks_equal_400_save_load_600(tmp_path: Path, two_car_world: WorldState) -> None:
    """Plan §4.7 test 2 / §18.4: interrupting with save + fresh-process load changes nothing."""
    straight = run_ticks(two_car_world, 1000).canonical_hash()

    db = tmp_path / "w.sqlite"
    with SnapshotStore(db) as store:
        snapshot_id = store.save(run_ticks(two_car_world, 400), label="tick 400").snapshot_id
    assert _continue_in_fresh_process(db, snapshot_id, 600) == straight


def test_resume_mid_race_to_the_finish(tmp_path: Path, two_car_world: WorldState) -> None:
    """Save around lap 18, finish the race in another process: same final classification."""
    final = run_to_end(two_car_world).canonical_hash()
    mid = run_ticks(two_car_world, 18 * 168)
    assert 17 <= max(c.laps_completed for c in mid.cars) <= 18

    db = tmp_path / "w.sqlite"
    with SnapshotStore(db) as store:
        snapshot_id = store.save(mid, label="lap 18").snapshot_id
    assert _continue_in_fresh_process(db, snapshot_id, "end") == final
