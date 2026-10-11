"""The M0 acceptance-gate command from the master plan, run exactly as a user would."""

import subprocess
import sys
from pathlib import Path

from tests.conftest import REPO_ROOT


def test_demo_pause_save_reload_resume_matches(tmp_path: Path) -> None:
    db = tmp_path / "gate.sqlite"
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "demo_headless.py"),
            "--seed",
            "42",
            "--pause",
            "--save",
            "--reload",
            "--resume",
            "--db",
            str(db),
            "--every",
            "53",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    out = result.stdout
    assert "[PAUSED]" in out and "[SAVED]" in out and "[LOADED" in out
    assert "Determinism check vs uninterrupted run: MATCH" in out
    assert db.exists()
