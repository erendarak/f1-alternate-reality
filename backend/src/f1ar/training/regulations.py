"""The FIA's rule set for AI decisions (M5, D-037, ``docs/M5_DESIGN.md`` §5).

The engine already enforces the sporting rules it models (no stop on the final lap or after the
chequered flag, the two-dry-compounds rule unless a wet tyre was used). This module adds what
the FIA can **ban** on top: a versioned list of bans that removes actions from the AI's action
mask. A world pins its regulation version; new bans apply between seasons, each with a reason
(the owner's example: a strategy that is real and legal can still be banned later, as real
innovations have been).

``fia_2024_v1`` has no bans.
"""

from dataclasses import dataclass, field

from f1ar.domain.race_state import Compound, PaceMode


@dataclass(frozen=True)
class Ban:
    reason: str
    since: str
    """Season or regulation version that introduced it."""
    pit_compound: Compound | None = None
    """Stopping for this compound is illegal."""
    pace_mode: PaceMode | None = None
    """Choosing this pace mode is illegal."""


@dataclass(frozen=True)
class RegulationSet:
    version: str
    description: str
    bans: tuple[Ban, ...] = field(default=())

    def allows_pit(self, compound: Compound) -> bool:
        return all(b.pit_compound is not compound for b in self.bans)

    def allows_pace(self, mode: PaceMode) -> bool:
        return all(b.pace_mode is not mode for b in self.bans)

    def with_ban(self, ban: Ban, version: str) -> "RegulationSet":
        """A new version with one more ban (the old version stays valid for worlds pinning it)."""
        return RegulationSet(version, f"{self.description}; + ban: {ban.reason}", (*self.bans, ban))


REGULATIONS_2024 = RegulationSet(
    version="fia_2024_v1",
    description=(
        "2024 sporting rules as the engine models them (two dry compounds unless a wet tyre is "
        "used, no stop on the final lap); no bans"
    ),
)
