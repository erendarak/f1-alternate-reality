"""Crash recovery (plan §4.7 tests 1, 7, 8, 9; M2 DoD): run -> crash -> fresh process restores
the latest valid save, the client re-sends its script -> same race, same event history."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from f1ar.persistence.sqlite_store import SnapshotStore
from tests.integration.scripted_race import (
    REPO_ROOT,
    digest,
    drive,
    new_world,
    race_hash,
    runner_for,
)


@pytest.fixture(scope="module")
def reference(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    """The same script, never interrupted."""
    db = tmp_path_factory.mktemp("ref") / "ref.sqlite"
    with SnapshotStore(db) as store:
        runner = runner_for(new_world(), store, flush_every_ticks=240)
        drive(runner)
        runner.shutdown()
        final = runner.state
        events = store.events(final.timeline_id)
        return {
            "race_hash": race_hash(final),
            "events_digest": digest(events),
            "event_count": len(events),
            "in_memory_digest": digest(list(runner.events)),
            "ham_stops": final.car("lewis_hamilton_2024").pit_stops,
        }


def test_stored_history_equals_what_happened(reference: dict[str, object]) -> None:
    assert reference["events_digest"] == reference["in_memory_digest"]


@pytest.mark.parametrize(
    ("flush_every", "expect_branch"),
    [
        # Logs flushed every 240 ticks: events after the last autosave reached the disk, so the
        # restored save isn't the end of its timeline -> recovery continues in a new branch.
        (240, True),
        # Logs only written with saves: the last autosave *is* the end of the timeline -> the
        # same timeline continues.
        (1_000_000, False),
    ],
)
def test_crash_then_restore_in_a_fresh_process(
    tmp_path: Path, reference: dict[str, object], flush_every: int, expect_branch: bool
) -> None:
    db = tmp_path / "crash.sqlite"
    with SnapshotStore(db) as store:
        runner = runner_for(new_world(), store, flush_every_ticks=flush_every)
        drive(runner, crash_at=3000)  # after the pit request and the safety car, mid-SC
        world_id = runner.state.world_id
        crashed_tick = runner.state.sim_tick
    # Abandoned without shutdown: whatever wasn't saved or flushed is lost, like a real crash.

    out = subprocess.run(
        [
            sys.executable,
            "-m",
            "tests.integration.scripted_race",
            str(db),
            world_id,
            str(flush_every),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    recovered = json.loads(out.stdout.strip().splitlines()[-1])

    assert recovered["restored_tick"] < crashed_tick  # some progress was really lost
    assert recovered["restored_label"].startswith("autosave")
    assert recovered["branched"] is expect_branch
    # Same race and the same full history (parent prefix + branch), not just the same end.
    assert recovered["race_hash"] == reference["race_hash"]
    assert recovered["events_digest"] == reference["events_digest"]
    assert recovered["event_count"] == reference["event_count"]
    # The pit request was re-sent after the restart but applied only once (plan §4.7 test 9).
    assert recovered["ham_stops"] == reference["ham_stops"]
