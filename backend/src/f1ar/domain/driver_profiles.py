"""Driver profiles: the persona vector every Stage 1 policy is conditioned on (plan §3).

Three layers (plan §3.1). This module holds layer 1; the others come later:

1. ``DriverArchetype``: seeded persona/skill priors (here).
2. ``DriverCondition``: dynamic in-world state such as form or fatigue (world state, later).
3. ``DriverPolicy``: learned decision making, referenced by ``PolicyRef`` (training, later).

Every number here is an ``editorial_assumption``: an illustrative simulation prior, NOT
measured psychometrics or telemetry. Profiles are resolved from a template plus explicit
overrides, so it is always visible which values were chosen per driver and which were
inherited from a generic template.

Behaviour must be driven by the numeric vector, never by the driver's name or id.
"""

from pathlib import Path
from typing import Annotated, Any, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from f1ar.domain.identities import RaceRoster
from f1ar.domain.provenance import SourceType
from f1ar.persistence.snapshot_codec import state_hash

PROFILE_SCHEMA_VERSION = 1

Unit = Annotated[float, Field(ge=0.0, le=1.0)]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class DriverArchetype(_Frozen):
    """Persona/skill priors in [0, 1]. Field order defines the feature vector order.

    Note the separate causal channels (plan §3.2): ``risk_appetite`` is a *preference*,
    ``consistency`` is a *skill*. High ``tire_management`` does not imply slow pace.
    """

    # Stage 1: affect transitions and/or strategy preferences
    race_pace_skill: Unit
    consistency: Unit
    risk_appetite: Unit
    tire_management: Unit
    wet_skill: Unit
    adaptability: Unit
    # Future simulators / Stage 2
    qualifying_skill: Unit
    overtake_skill: Unit
    defense_skill: Unit
    team_orientation: Unit
    composure: Unit
    # Stage 3
    ambition: Unit
    sociability: Unit
    negotiation_skill: Unit

    def as_vector(self) -> tuple[float, ...]:
        return tuple(getattr(self, name) for name in ARCHETYPE_FIELDS)


ARCHETYPE_FIELDS: tuple[str, ...] = tuple(DriverArchetype.model_fields)

STAGE1_FIELDS: tuple[str, ...] = ARCHETYPE_FIELDS[:6]
"""Fields with a direct Stage 1 effect (plan §3.2 table)."""

PROFILE_FEATURE_NAMES: tuple[str, ...] = tuple(f"profile.{name}" for name in ARCHETYPE_FIELDS)
"""Names of the profile block in policy observations; stable order, versioned by schema."""


class ArchetypeTemplate(_Frozen):
    description: str
    archetype: DriverArchetype


class ProfileMetadata(_Frozen):
    numeric_values_source_type: Literal[
        SourceType.EDITORIAL_ASSUMPTION,
        SourceType.ESTIMATED_FROM_DATA,
        SourceType.LEARNED_PARAMETER,
    ]
    """A persona can be assumed, estimated or learned, but never ``historical_verified``."""
    estimated: bool
    confidence: Unit
    explanation: str
    persona_label: str | None
    origin: str
    base_template: str
    overridden_fields: tuple[str, ...]


class PolicyRef(_Frozen):
    type: str
    version: str


class DriverProfile(_Frozen):
    schema_version: Literal[1]
    driver_id: str
    person_id: str
    season: int
    display_name: str
    identity_source_type: Literal[SourceType.HISTORICAL_VERIFIED]
    profile_revision: str
    archetype: DriverArchetype
    metadata: ProfileMetadata
    policy: PolicyRef

    @property
    def content_hash(self) -> str:
        return state_hash(self.model_dump(mode="json"))

    def features(self) -> dict[str, float]:
        """Profile block of a policy observation, keyed by ``PROFILE_FEATURE_NAMES``."""
        return dict(zip(PROFILE_FEATURE_NAMES, self.archetype.as_vector(), strict=True))


class ProfileSet(_Frozen):
    """All profiles for one event, pinned together by ``manifest_hash``."""

    schema_version: Literal[1]
    event_id: str
    season: int
    profiles: tuple[DriverProfile, ...]

    @model_validator(mode="after")
    def _unique(self) -> Self:
        ids = [p.driver_id for p in self.profiles]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate driver_id in profile set")
        return self

    @property
    def manifest_hash(self) -> str:
        ordered = sorted(self.profiles, key=lambda p: p.driver_id)
        return state_hash(
            {
                "schema_version": self.schema_version,
                "event_id": self.event_id,
                "profiles": [p.model_dump(mode="json") for p in ordered],
            }
        )

    def get(self, driver_id: str) -> DriverProfile:
        for p in self.profiles:
            if p.driver_id == driver_id:
                return p
        raise KeyError(driver_id)


# --- seed resolution ---------------------------------------------------------------------


class _SeedDriver(_Frozen):
    template: str
    confidence: Unit
    origin: str
    persona_label: str | None = None
    overrides: dict[str, float] = Field(default_factory=dict)


class _SeedFile(_Frozen):
    schema_version: Literal[1]
    season: int
    event_id: str
    explanation: str
    profile_revision: str
    policy: PolicyRef
    drivers: dict[str, _SeedDriver]


def load_templates(path: Path) -> dict[str, ArchetypeTemplate]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if raw.get("schema_version") != PROFILE_SCHEMA_VERSION:
        raise ValueError(f"unsupported template schema_version in {path.name}")
    return {name: ArchetypeTemplate.model_validate(t) for name, t in raw["templates"].items()}


def build_profiles(
    seed: dict[str, Any], templates: dict[str, ArchetypeTemplate], roster: RaceRoster
) -> ProfileSet:
    """Resolve a seed (template + overrides per driver) against a race roster.

    Every starter in the roster must have exactly one profile, and no extra profiles are
    allowed, so a roster change can't silently leave a driver unprofiled.
    """
    spec = _SeedFile.model_validate(seed)
    if spec.event_id != roster.event_id or spec.season != roster.season:
        raise ValueError(
            f"seed is for {spec.event_id}/{spec.season}, roster is "
            f"{roster.event_id}/{roster.season}"
        )
    missing = set(roster.driver_ids()) - set(spec.drivers)
    extra = set(spec.drivers) - set(roster.driver_ids())
    if missing or extra:
        raise ValueError(f"seed/roster mismatch: missing={sorted(missing)} extra={sorted(extra)}")

    profiles = []
    for entry in roster.entries:
        d = spec.drivers[entry.driver_id]
        if d.template not in templates:
            raise ValueError(f"{entry.driver_id}: unknown template {d.template!r}")
        unknown = set(d.overrides) - set(ARCHETYPE_FIELDS)
        if unknown:
            raise ValueError(f"{entry.driver_id}: unknown profile fields {sorted(unknown)}")
        values = templates[d.template].archetype.model_dump() | d.overrides
        profiles.append(
            DriverProfile(
                schema_version=1,
                driver_id=entry.driver_id,
                person_id=entry.person_id,
                season=roster.season,
                display_name=entry.display_name,
                identity_source_type=SourceType.HISTORICAL_VERIFIED,
                profile_revision=spec.profile_revision,
                archetype=DriverArchetype.model_validate(values),
                metadata=ProfileMetadata(
                    numeric_values_source_type=SourceType.EDITORIAL_ASSUMPTION,
                    estimated=True,
                    confidence=d.confidence,
                    explanation=spec.explanation,
                    persona_label=d.persona_label,
                    origin=d.origin,
                    base_template=d.template,
                    overridden_fields=tuple(f for f in ARCHETYPE_FIELDS if f in d.overrides),
                ),
                policy=spec.policy,
            )
        )
    return ProfileSet(
        schema_version=1, event_id=roster.event_id, season=roster.season, profiles=tuple(profiles)
    )


def load_profiles(seed_path: Path, templates_path: Path, roster: RaceRoster) -> ProfileSet:
    seed = yaml.safe_load(seed_path.read_text(encoding="utf-8"))
    return build_profiles(seed, load_templates(templates_path), roster)


def load_season_profiles(seed_path: Path, templates_path: Path, roster: RaceRoster) -> ProfileSet:
    """Profiles of one race from a season seed (M4): the seed covers every driver of the season,
    the race takes those of its roster. A roster driver missing from the seed is still an error.
    """
    seed = yaml.safe_load(seed_path.read_text(encoding="utf-8"))
    if seed.get("season") != roster.season:
        raise ValueError(f"season seed is for {seed.get('season')}, roster is {roster.season}")
    wanted = set(roster.driver_ids())
    race_seed = seed | {
        "event_id": roster.event_id,
        "drivers": {d: v for d, v in seed["drivers"].items() if d in wanted},
    }
    return build_profiles(race_seed, load_templates(templates_path), roster)
