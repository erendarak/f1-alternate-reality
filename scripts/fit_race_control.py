"""Fit a season's race-control rates (H2c, D-056): configs/calibration/race_control_<season>_v1.json
and its registry manifest.

Same method of moments as 2024 (``calibration/race_control_fit``, EXP-006) on the season's
dry_train races (2024's own file used all 21 dry races; here the dry_test races stay out for
EXP-025). Track status is real from 2018 and inferred from lap times before (D-055), where SC
and VSC aren't told apart: for 2015-2017 (the VSC exists) the inferred neutralisations are split
with 2018's VSC share (borrowed, labelled); 2010-2014 had no VSC.

Usage:
    uv run python scripts/fit_race_control.py --season 2019
"""

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import yaml

from f1ar.calibration.race_control_fit import RACE_CONTROL_FILE, fit_race_control, race_counts
from f1ar.calibration.season_files import dataset_files, season_path
from f1ar.domain.season_rules import rules_for

ROOT = Path(__file__).resolve().parents[1]
FIRST_TRACK_STATUS_SEASON = 2018


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--season", type=int, required=True)
    args = parser.parse_args()
    season = args.season
    _, splits_file = dataset_files(ROOT, season)
    manifest = yaml.safe_load(splits_file.read_text(encoding="utf-8"))
    train = list(manifest["splits"]["dry_train"])
    counts = [race_counts(ROOT / "data" / "processed" / e) for e in train]
    inferred = season < FIRST_TRACK_STATUS_SEASON
    note = f"H2c: {len(train)} dry_train {season} races"
    fit = fit_race_control(counts, note=note)
    if inferred and rules_for(ROOT / "configs", season).vsc:
        ref_file = ROOT / "configs" / season_path(RACE_CONTROL_FILE, FIRST_TRACK_STATUS_SEASON)
        ref = json.loads(ref_file.read_text(encoding="utf-8"))
        vsc_share = ref["vscs"] / max(1, ref["vscs"] + ref["safety_cars"])
        neutral = fit.safety_cars  # inferred: SC and VSC together
        sc = neutral * (1 - vsc_share)
        vsc = neutral * vsc_share
        fit = replace(
            fit,
            sc_per_retirement=round(sc / fit.retirements, 4) if fit.retirements else 0.0,
            vsc_per_retirement_without_sc=round(vsc / max(1.0, fit.retirements - sc), 4),
            safety_cars=round(sc),
            vscs=round(vsc),
            note=f"{note}; neutralisations inferred from lap times, split SC/VSC with "
            f"{FIRST_TRACK_STATUS_SEASON}'s VSC share {vsc_share:.2f} (borrowed)",
        )
    elif inferred:
        fit = replace(fit, note=f"{note}; neutralisations inferred from lap times (no VSC yet)")
    out = ROOT / "configs" / season_path(RACE_CONTROL_FILE, season)
    out.write_text(fit.to_json() + "\n", encoding="utf-8", newline="\n")
    body = {
        "model_name": "race_control",
        "model_version": "v1",
        "kind": "method of moments: retirement hazard, SC/VSC per retirement, SC length",
        "source_type": "estimated_from_data",
        "artifact": out.relative_to(ROOT).as_posix(),
        "artifact_sha256": "sha256:" + hashlib.sha256(out.read_bytes()).hexdigest(),
        "feature_schema_version": 1,
        "training_dataset_manifest": splits_file.relative_to(ROOT).as_posix(),
        "training_dataset_content_sha256": manifest["file"]["content_sha256"],
        "training_races": sorted(train),
        "metrics": {
            "retirements": fit.retirements,
            "safety_cars": fit.safety_cars,
            "vscs": fit.vscs,
            "held_out_check": "EXP-025",
        },
        "status": f"fitted for {season} (H2c, D-056)",
    }
    (ROOT / "configs" / "models" / f"race_control_{season}_v1.yaml").write_text(
        yaml.safe_dump(body, sort_keys=False, width=100), encoding="utf-8", newline="\n"
    )
    print(
        f"{season}: {fit.races} races, retirements {fit.retirement_per_car_lap:.5f}/car-lap, "
        f"lap 1 {fit.lap1_retirement_per_starter:.4f}/starter, SC {fit.sc_per_retirement:.2f} / "
        f"VSC {fit.vsc_per_retirement_without_sc:.2f} per retirement, SC "
        f"{fit.sc_laps_p25}-{fit.sc_laps_p75} laps -> {out.relative_to(ROOT)}"
    )


if __name__ == "__main__":
    main()
