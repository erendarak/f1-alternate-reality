import copy
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from f1ar.domain.driver_profiles import (
    ARCHETYPE_FIELDS,
    PROFILE_FEATURE_NAMES,
    STAGE1_FIELDS,
    DriverProfile,
    ProfileSet,
    build_profiles,
    load_templates,
)
from f1ar.domain.identities import RaceRoster
from f1ar.domain.provenance import SourceType
from tests.conftest import CONFIGS

SEED_PATH = CONFIGS / "drivers" / "seed_2024_monza_drivers.yaml"
TEMPLATES_PATH = CONFIGS / "drivers" / "archetypes.yaml"


def _seed() -> dict[str, Any]:
    data: dict[str, Any] = yaml.safe_load(SEED_PATH.read_text(encoding="utf-8"))
    return data


def _build(seed: dict[str, Any], roster: RaceRoster) -> ProfileSet:
    return build_profiles(seed, load_templates(TEMPLATES_PATH), roster)


# --- seed coverage and provenance --------------------------------------------------------


def test_every_starter_has_exactly_one_profile(
    monza_profiles: ProfileSet, monza_roster: RaceRoster
) -> None:
    assert len(monza_profiles.profiles) == 20
    assert {p.driver_id for p in monza_profiles.profiles} == set(monza_roster.driver_ids())


def test_all_persona_values_are_labelled_editorial(monza_profiles: ProfileSet) -> None:
    for p in monza_profiles.profiles:
        assert p.metadata.numeric_values_source_type is SourceType.EDITORIAL_ASSUMPTION
        assert p.metadata.estimated is True
        assert p.identity_source_type is SourceType.HISTORICAL_VERIFIED


def test_persona_can_never_claim_historical_verified(monza_profiles: ProfileSet) -> None:
    data = monza_profiles.profiles[0].model_dump(mode="json")
    data["metadata"]["numeric_values_source_type"] = "historical_verified"
    with pytest.raises(ValidationError):
        DriverProfile.model_validate(data)


def test_missing_driver_in_seed_is_rejected(monza_roster: RaceRoster) -> None:
    seed = _seed()
    del seed["drivers"]["franco_colapinto_2024"]
    with pytest.raises(ValueError, match="missing=.*franco_colapinto_2024"):
        _build(seed, monza_roster)


def test_seed_for_another_event_is_rejected(monza_roster: RaceRoster) -> None:
    seed = _seed()
    seed["event_id"] = "2026_italian_gp"
    with pytest.raises(ValueError, match="seed is for"):
        _build(seed, monza_roster)


# --- value validation -------------------------------------------------------------------


@pytest.mark.parametrize("bad", [-0.01, 1.01])
def test_out_of_range_values_rejected(monza_roster: RaceRoster, bad: float) -> None:
    seed = _seed()
    seed["drivers"]["lewis_hamilton_2024"]["overrides"]["risk_appetite"] = bad
    with pytest.raises(ValidationError):
        _build(seed, monza_roster)


def test_unknown_profile_field_rejected(monza_roster: RaceRoster) -> None:
    seed = _seed()
    seed["drivers"]["lewis_hamilton_2024"]["overrides"]["aggression"] = 0.9
    with pytest.raises(ValueError, match="unknown profile fields"):
        _build(seed, monza_roster)


# --- resolution: template + overrides -----------------------------------------------------


def test_template_only_driver_equals_template(monza_profiles: ProfileSet) -> None:
    templates = load_templates(TEMPLATES_PATH)
    colapinto = monza_profiles.get("franco_colapinto_2024")
    assert colapinto.metadata.base_template == "generic_rookie"
    assert colapinto.metadata.overridden_fields == ()
    assert colapinto.archetype == templates["generic_rookie"].archetype


def test_overrides_are_recorded_and_rest_inherited(monza_profiles: ProfileSet) -> None:
    templates = load_templates(TEMPLATES_PATH)
    ham = monza_profiles.get("lewis_hamilton_2024")
    assert ham.metadata.overridden_fields == STAGE1_FIELDS
    assert ham.archetype.tire_management == 0.95
    veteran = templates["generic_veteran"].archetype
    for name in ARCHETYPE_FIELDS[6:]:
        assert getattr(ham.archetype, name) == getattr(veteran, name)


def test_plan_personas_differ_where_the_plan_says(monza_profiles: ProfileSet) -> None:
    max_ = monza_profiles.get("max_verstappen_2024").archetype
    ham = monza_profiles.get("lewis_hamilton_2024").archetype
    assert max_.risk_appetite > ham.risk_appetite
    assert ham.tire_management > max_.tire_management
    assert ham.consistency > max_.consistency


# --- policy-facing vector -----------------------------------------------------------------


def test_feature_vector_shape_and_order(monza_profiles: ProfileSet) -> None:
    assert len(ARCHETYPE_FIELDS) == 14
    p = monza_profiles.get("max_verstappen_2024")
    features = p.features()
    assert tuple(features) == PROFILE_FEATURE_NAMES
    assert features["profile.risk_appetite"] == 0.82
    assert all(isinstance(v, float) for v in features.values())


def test_features_depend_on_values_not_identity(monza_profiles: ProfileSet) -> None:
    """Two drivers with the same persona values get identical policy features.

    Policies see the profile only through the numeric vector, so behaviour can't be
    keyed on a name (plan: test_driver_policy_not_hardcoded_from_name, data side).
    """
    perez = monza_profiles.get("sergio_perez_2024")
    hulkenberg = monza_profiles.get("nico_hulkenberg_2024")
    assert perez.features() == hulkenberg.features()


# --- pinning / hashing --------------------------------------------------------------------


def test_manifest_hash_is_stable_and_sensitive(
    monza_profiles: ProfileSet, monza_roster: RaceRoster
) -> None:
    again = _build(_seed(), monza_roster)
    assert again.manifest_hash == monza_profiles.manifest_hash

    seed = copy.deepcopy(_seed())
    seed["drivers"]["oscar_piastri_2024"]["overrides"]["risk_appetite"] = 0.63
    assert _build(seed, monza_roster).manifest_hash != monza_profiles.manifest_hash


def test_profiles_are_immutable(monza_profiles: ProfileSet) -> None:
    with pytest.raises(ValidationError):
        monza_profiles.profiles[0].archetype.risk_appetite = 0.1  # type: ignore[misc]
