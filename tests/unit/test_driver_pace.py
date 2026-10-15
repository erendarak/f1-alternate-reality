"""W2 (engine v0.12.0, D-043): driver race pace vs teammate, race form and lap noise per driver."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from f1ar.calibration.driver_pace_fit import PACE_FILE, fit, load_pace
from f1ar.domain.driver_profiles import ProfileSet
from f1ar.domain.identities import RaceRoster
from f1ar.domain.race_state import Compound, PaceMode, RaceSetup
from f1ar.simulation import lightweight_engine as engine
from f1ar.simulation.lap_model import sample_lap_time_s
from f1ar.simulation.scenarios import event_data_dir, load_event, setup_from_codes
from tests.conftest import CONFIGS, HAM, REPO_ROOT, VER


def _synthetic_season(races: int = 12, form_sd: float = 0.2) -> pd.DataFrame:
    """A 20-car field (10 teams): A1 is 0.3 s faster than A2, every other pair equal; lap noise
    0.3 s in team a, 0.6 s in team b, 0.4 s elsewhere (the per-lap term needs a full field)."""
    rng = np.random.default_rng(1)
    pace = {"A1": -0.15, "A2": 0.15}
    noise = {"A1": 0.3, "A2": 0.3, "B1": 0.6, "B2": 0.6}
    team = {}
    for t in "abcdefghij":
        for k in (1, 2):
            team[f"{t.upper()}{k}"] = t
    rows = []
    for r in range(races):
        form = {d: rng.normal(0, form_sd) for d in team}
        for lap in range(2, 52):
            lap_term = 90.0 - 0.05 * lap + rng.normal(0, 0.1)
            for d, t in team.items():
                mean = lap_term + pace.get(d, 0.0) + form[d] + 0.1 * "abcdefghij".index(t)
                rows.append(
                    {
                        "event_id": f"r{r}",
                        "driver_code": d,
                        "team_id": t,
                        "lap_number": lap,
                        "lap_time_s": mean + 0.03 * (lap - 1) + rng.normal(0, noise.get(d, 0.4)),
                        "compound": "MEDIUM",
                        "tyre_age_start": 0,
                        "stint": 1,
                        "stint_lap": lap,
                        "pit_in_lap": False,
                        "pit_out_lap": False,
                        "track_status_start": "1",
                        "gap_ahead_s": 5.0,
                    }
                )
    return pd.DataFrame(rows)


def test_fit_recovers_teammate_gap_noise_and_form() -> None:
    laps = _synthetic_season(races=24, form_sd=0.2)
    out = fit(laps, sorted(laps["event_id"].unique()))
    d = out["drivers"]
    gap = d["A2"]["pace_vs_teammate_s"] - d["A1"]["pace_vs_teammate_s"]
    assert gap == pytest.approx(0.3 * 24 / 27, abs=0.08)  # shrunk with 3 prior races
    assert abs(d["B1"]["pace_vs_teammate_s"] - d["B2"]["pace_vs_teammate_s"]) < 0.12
    assert d["A1"]["raw_noise_s"] == pytest.approx(0.3, abs=0.04)
    assert d["B1"]["raw_noise_s"] == pytest.approx(0.6, abs=0.06)
    assert out["form_sd_s"] == pytest.approx(0.2, abs=0.07)


def test_noise_offset_and_form_in_the_lap_model(two_car_setup: RaceSetup) -> None:
    lm = two_car_setup.lap_model
    driver = two_car_setup.driver(VER).profile.archetype
    kw: dict[str, Any] = {
        "driver_id": VER,
        "compound": Compound.MEDIUM,
        "tyre_wear_s": 0.0,
        "pace_mode": PaceMode.BALANCED,
        "laps_remaining": 20,
    }
    wide = [
        sample_lap_time_s(lm, driver, seed=1, lap_number=k, noise_sd=0.5, **kw) for k in range(400)
    ]
    narrow = [
        sample_lap_time_s(lm, driver, seed=1, lap_number=k, noise_sd=0.1, **kw) for k in range(400)
    ]
    assert np.std(wide) == pytest.approx(5 * np.std(narrow), rel=0.05)
    # the engine adds the driver's offset and today's form to the car's
    entry = two_car_setup.driver(VER).model_copy(update={"driver_pace_offset_s": 0.4})
    base = engine._sample_next_lap(
        two_car_setup,
        3,
        VER,
        compound=Compound.MEDIUM,
        pace_mode=PaceMode.BALANCED,
        laps_completed=5,
        tyre_wear_s=0.0,
    )
    moved = two_car_setup.model_copy(update={"drivers": (entry, two_car_setup.driver(HAM))})
    slower = engine._sample_next_lap(
        moved,
        3,
        VER,
        compound=Compound.MEDIUM,
        pace_mode=PaceMode.BALANCED,
        laps_completed=5,
        tyre_wear_s=0.0,
    )
    assert slower - base == pytest.approx(0.4)
    formed = moved.model_copy(
        update={"lap_model": moved.lap_model.model_copy(update={"form_sd_s": 0.3})}
    )
    a = [
        engine._sample_next_lap(
            formed,
            s,
            VER,
            compound=Compound.MEDIUM,
            pace_mode=PaceMode.BALANCED,
            laps_completed=5,
            tyre_wear_s=0.0,
        )
        for s in (1, 1)
    ]
    assert a[0] == a[1]  # keyed: same seed, same form
    laps = [
        engine._sample_next_lap(
            formed,
            7,
            VER,
            compound=Compound.MEDIUM,
            pace_mode=PaceMode.BALANCED,
            laps_completed=k,
            tyre_wear_s=0.0,
        )
        - engine._sample_next_lap(
            moved,
            7,
            VER,
            compound=Compound.MEDIUM,
            pace_mode=PaceMode.BALANCED,
            laps_completed=k,
            tyre_wear_s=0.0,
        )
        for k in (4, 9, 20)
    ]
    assert np.ptp(laps) == pytest.approx(0.0, abs=1e-9)  # one form for the whole race


def test_data_backed_race_uses_the_fitted_pace(
    monza_roster: RaceRoster, monza_profiles: ProfileSet
) -> None:
    cfg = load_pace(CONFIGS)
    assert cfg is not None, f"{PACE_FILE} missing"
    roster, profiles = load_event(CONFIGS, "2024_italian_gp")
    data = event_data_dir(REPO_ROOT, "2024_italian_gp")
    if data is None:
        pytest.skip("Monza data not fetched")
    setup = setup_from_codes(roster, profiles, None, data_dir=data)
    assert setup.lap_model.skill_gap_s == 0.0
    assert setup.lap_model.form_sd_s == pytest.approx(cfg["form_sd_s"])
    ver = setup.driver(VER)
    assert ver.driver_pace_offset_s == pytest.approx(cfg["drivers"]["VER"]["pace_vs_teammate_s"])
    assert ver.lap_noise_s == pytest.approx(cfg["drivers"]["VER"]["lap_noise_s"])
    assert "estimated_from_data" in setup.inputs["driver_pace"]
    # W3 refits overtaking with this noise (W2's global shift isn't applied then)
    assert "refit with the W2 lap noise" in setup.inputs["traffic"]
    plain = setup_from_codes(monza_roster, monza_profiles, ["VER", "HAM"])
    assert plain.lap_model.skill_gap_s > 0 and plain.driver(VER).lap_noise_s is None
    assert Path(CONFIGS / PACE_FILE).exists()
