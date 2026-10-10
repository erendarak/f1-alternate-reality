import json
import math
import subprocess
import sys

import pytest

from f1ar.persistence.snapshot_codec import canonical_json, state_hash

SAMPLE = {
    "tick": 920,
    "sim_clock_s": 1263.5,
    "cars": [{"driver_id": "max_verstappen_2024", "tyre_age": 12, "compound": "HARD"}],
    "world_status": "PAUSED",
    "note": "Monza — Parabolica",
}


def test_key_order_does_not_change_encoding() -> None:
    reordered = dict(reversed(list(SAMPLE.items())))
    assert canonical_json(SAMPLE) == canonical_json(reordered)
    assert state_hash(SAMPLE) == state_hash(reordered)


def test_any_value_change_changes_hash() -> None:
    changed = {**SAMPLE, "tick": 921}
    assert state_hash(SAMPLE) != state_hash(changed)


def test_float_round_trip_is_exact() -> None:
    value = {"x": 0.1 + 0.2}
    assert json.loads(canonical_json(value))["x"] == value["x"]


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_non_finite_floats_are_rejected(bad: float) -> None:
    with pytest.raises(ValueError):
        canonical_json({"x": bad})


def test_non_json_types_are_rejected() -> None:
    with pytest.raises(TypeError):
        canonical_json({"x": {1, 2, 3}})


def test_hash_is_stable_across_processes() -> None:
    """Hash randomization (PYTHONHASHSEED) must not leak into the canonical hash."""
    code = f"from f1ar.persistence.snapshot_codec import state_hash;print(state_hash({SAMPLE!r}))"
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        encoding="utf-8",
    ).stdout.strip()
    assert out == state_hash(SAMPLE)
