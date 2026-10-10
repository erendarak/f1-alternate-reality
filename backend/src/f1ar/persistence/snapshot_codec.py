"""Canonical encoding and hashing of world state.

Every snapshot, determinism test and save checksum goes through this module, so the
encoding must be byte-for-byte stable across runs, processes and machines:

- keys are sorted, separators are fixed, output is UTF-8;
- NaN / +-Infinity are rejected (they have no portable JSON form and usually mean a bug);
- only JSON-native types are accepted (dict with str keys, list/tuple, str, int, float,
  bool, None). Anything else must be converted to one of these by its owner first.

Python's float repr is the shortest round-tripping representation, so floats encode
deterministically without extra rounding. Rounding policy, if ever needed, belongs to the
state model, not to the codec.
"""

import hashlib
import json
from typing import Any

CODEC_VERSION = 1

HASH_PREFIX = "sha256:"


def canonical_json(obj: Any) -> bytes:
    """Encode ``obj`` as canonical UTF-8 JSON bytes."""
    try:
        text = json.dumps(
            obj,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except ValueError as exc:  # NaN / Infinity
        raise ValueError(f"state is not canonically encodable: {exc}") from exc
    return text.encode("utf-8")


def state_hash(obj: Any) -> str:
    """Return ``sha256:<hex>`` of the canonical encoding of ``obj``."""
    return HASH_PREFIX + hashlib.sha256(canonical_json(obj)).hexdigest()
