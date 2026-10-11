"""Keyed (counter-based) random numbers: ``draw = f(seed, key...)``.

Instead of one shared generator whose state advances with every call, each random draw is
derived from the world seed plus a semantic key such as ``("lap_noise", driver_id, lap)``.

- **No hidden RNG state.** The only thing to save is the seed and scheme, so save/load
  can't drift.
- **Common random numbers across forks** (plan §4.5). A fork that changes one driver's
  strategy still sees identical exogenous luck for every other (driver, lap). Adding or
  reordering draws elsewhere can't shift anyone's sequence.

Uniforms come straight from SHA-256, so they are platform independent. Normals use
Box-Muller via ``math.log/cos``. Those are deterministic on one machine/libm, which is the
plan's "deterministic CPU mode" scope; cross-platform last-ulp equality is not promised.
"""

import hashlib
import math
from typing import Literal

from pydantic import BaseModel, ConfigDict

SCHEME: Literal["sha256-keyed-v1"] = "sha256-keyed-v1"
_TWO_POW_53 = float(1 << 53)


class RngSpec(BaseModel):
    """Everything needed to reproduce every random draw of a world."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    scheme: Literal["sha256-keyed-v1"] = SCHEME
    seed: int


def uniform(seed: int, *key: str | int) -> float:
    """Uniform draw in the open interval (0, 1), fully determined by ``seed`` and ``key``."""
    material = "|".join([SCHEME, str(seed), *(str(k) for k in key)]).encode("utf-8")
    top53 = int.from_bytes(hashlib.sha256(material).digest()[:8], "big") >> 11
    return (top53 + 0.5) / _TWO_POW_53


def normal(seed: int, *key: str | int) -> float:
    """Standard normal draw (Box-Muller), fully determined by ``seed`` and ``key``."""
    u1 = uniform(seed, *key, "bm1")
    u2 = uniform(seed, *key, "bm2")
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
