"""Timeline forks: alternative futures from one canonical state (plan §4.5, §5 divergence contract).

A fork is a copy of a world state under a NEW ``timeline_id`` that remembers its parent.
Because ``WorldState`` is immutable and contains everything that drives the future, the
source can't be affected by anything that happens on the branch.

Randomness on the branch (plan §4.5, §13.3 ``seed_mode``):

- ``COMMON_EXOGENOUS_RANDOMNESS`` (default): same seed as the parent. Every (driver, lap)
  noise draw is identical, so outcome differences come from the decision being tested,
  not from different dice rolls. Use this for "what if" comparisons.
- ``INDEPENDENT``: the branch gets its own seed derived from (parent seed, timeline id),
  so it's a fresh sample of luck from the branch point onward. The lap already in progress
  keeps its pre-sampled duration.
"""

import hashlib
import uuid
from enum import StrEnum

from f1ar.domain.race_state import WorldState, WorldStatus
from f1ar.simulation.rng_streams import RngSpec


class SeedMode(StrEnum):
    COMMON_EXOGENOUS_RANDOMNESS = "COMMON_EXOGENOUS_RANDOMNESS"
    INDEPENDENT = "INDEPENDENT"


def new_timeline_id() -> str:
    return f"timeline_{uuid.uuid4().hex[:12]}"


def new_world_id(prefix: str = "world") -> str:
    """A world id that won't collide with another world in the same save file."""
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def main_timeline_id(world_id: str) -> str:
    """The id of a world's original timeline (timeline ids are unique across worlds)."""
    return f"{world_id}.main"


def branch_seed(parent_seed: int, timeline_id: str) -> int:
    digest = hashlib.sha256(f"fork|{parent_seed}|{timeline_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") >> 1  # non-negative 63-bit


def fork_world(
    state: WorldState,
    *,
    new_timeline_id: str,
    seed_mode: SeedMode = SeedMode.COMMON_EXOGENOUS_RANDOMNESS,
) -> WorldState:
    """Branch ``state`` into a new timeline. A running source is forked as PAUSED."""
    if new_timeline_id == state.timeline_id:
        raise ValueError("a fork needs a new timeline id")
    status = WorldStatus.PAUSED if state.world_status is WorldStatus.RUNNING else state.world_status
    rng = state.rng
    if seed_mode is SeedMode.INDEPENDENT:
        rng = RngSpec(seed=branch_seed(state.rng.seed, new_timeline_id))
    return state.model_copy(
        update={
            "timeline_id": new_timeline_id,
            "parent_timeline_id": state.timeline_id,
            "branch_event_seq": state.event_seq,
            "world_status": status,
            "rng": rng,
        }
    )
