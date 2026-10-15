"""W5 (D-045): persona fields estimated from data where 2024 can measure them."""

import shutil
from pathlib import Path
from typing import Any

from f1ar.calibration.persona_fit import (
    MIN_RELIABILITY,
    PERSONA_FILE,
    load_personas,
    rank_map,
    refit_profiles,
)
from f1ar.domain.driver_profiles import ProfileSet
from f1ar.domain.identities import RaceRoster
from f1ar.simulation.scenarios import setup_from_codes
from tests.conftest import CONFIGS, HAM, VER
from tests.unit.test_race_inputs import synthetic_laps


def test_rank_map_reorders_the_persona_values_by_the_measure() -> None:
    persona = {"A": 0.9, "B": 0.5, "C": 0.7}
    out = rank_map({"A": 3.0, "B": 1.0, "C": 2.0}, persona, higher_is_better=False)
    assert out == {"B": 0.9, "C": 0.7, "A": 0.5}  # lowest measure gets the best value
    assert sorted(out.values()) == sorted(persona.values())  # the spread stays the persona's
    assert rank_map({"A": 1.0}, persona, higher_is_better=True) == {"A": 0.9}


def _inputs(roster: RaceRoster, reliability: float) -> tuple[Any, Any, Any]:
    codes = [e.code for e in roster.entries]
    personas = {
        "wear": {
            "reliability_drivers": reliability,
            "drivers": {c: {"wear_multiplier": 1.25 if c == "VER" else 1.0} for c in codes},
        }
    }
    # VER: noisiest laps, best attacker; HAM: steadiest, worst defender
    noise = {c: 0.5 for c in codes} | {"VER": 0.9, "HAM": 0.1}
    pace = {"drivers": {c: {"lap_noise_s": v} for c, v in noise.items()}}
    over = {c: {"attack_s": 0.0, "defend_s": 0.0} for c in codes}
    over |= {"VER": {"attack_s": 0.5, "defend_s": 0.0}, "HAM": {"attack_s": 0.0, "defend_s": -0.5}}
    return personas, pace, {"overtaking": {"drivers": over}}


def test_refit_uses_the_measured_order_and_skips_unreliable_wear(
    monza_roster: RaceRoster, monza_profiles: ProfileSet
) -> None:
    personas, pace, racecraft = _inputs(monza_roster, MIN_RELIABILITY - 0.01)
    out, notes = refit_profiles(monza_profiles, monza_roster, personas, pace, racecraft, 1.3, 0.7)
    before = {p.driver_id: p.archetype for p in monza_profiles.profiles}
    after = {p.driver_id: p for p in out.profiles}
    field = [a.consistency for a in before.values()]
    assert after[VER].archetype.consistency == min(field)
    assert after[HAM].archetype.consistency == max(field)
    assert after[VER].archetype.overtake_skill == max(a.overtake_skill for a in before.values())
    assert after[HAM].archetype.defense_skill == min(a.defense_skill for a in before.values())
    # wear reliability below the bar: tire_management stays editorial
    assert after[VER].archetype.tire_management == before[VER].tire_management
    assert "tire_management" not in notes[VER]
    assert set(notes[VER]) == {"consistency", "overtake_skill", "defense_skill"}
    assert after[VER].profile_revision.endswith("+w5")
    assert "consistency" in after[VER].metadata.overridden_fields
    assert "W5" in after[VER].metadata.explanation
    # reliable enough: VER's high wear maps to the low end of tire management
    personas, pace, racecraft = _inputs(monza_roster, MIN_RELIABILITY)
    out, notes = refit_profiles(monza_profiles, monza_roster, personas, pace, racecraft, 1.25, 0.7)
    assert next(p for p in out.profiles if p.driver_id == VER).archetype.tire_management == 0.0
    assert "tire_management" in notes[VER]


def test_nothing_changes_without_data(monza_roster: RaceRoster, monza_profiles: ProfileSet) -> None:
    out, notes = refit_profiles(monza_profiles, monza_roster, None, None, None, 1.3, 0.7)
    assert out == monza_profiles
    assert all(not n for n in notes.values())


def test_setup_records_the_estimated_fields(
    monza_roster: RaceRoster, monza_profiles: ProfileSet, tmp_path: Path
) -> None:
    synthetic_laps().to_parquet(tmp_path / "laps.parquet")
    with_file = setup_from_codes(
        monza_roster, monza_profiles, ["VER", "HAM"], data_dir=tmp_path, configs_dir=CONFIGS
    )
    assert load_personas(CONFIGS) is not None
    assert with_file.inputs["personas"].startswith("estimated_from_data")
    ver = next(d for d in with_file.drivers if d.driver_id == VER)
    assert "consistency" in ver.persona_estimates
    assert "tire_management" not in ver.persona_estimates  # 2024 wear reliability is too low
    configs = tmp_path / "configs"
    shutil.copytree(CONFIGS, configs)
    (configs / PERSONA_FILE).unlink()
    without = setup_from_codes(
        monza_roster, monza_profiles, ["VER", "HAM"], data_dir=tmp_path, configs_dir=configs
    )
    assert without.inputs["personas"].startswith("editorial")
    assert all(not d.persona_estimates for d in without.drivers)
