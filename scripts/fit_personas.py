"""Fit the W5 persona estimates (D-045): writes configs/calibration/personas_<season>_v1.json and
its registry manifest.

Tyre wear vs teammate and team, from the 16 dry_train races (``calibration/persona_fit``); the
reference slope is the W1 season medium wear rate. Consistency and overtake/defense skill come
from the W2 / W3 configs at setup time (rank-mapped), so they need no fit here.

Usage:
    uv run python scripts/fit_personas.py
    uv run python scripts/fit_personas.py --season 2019   # H2a: needs that season's W1 fit
"""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import yaml

from f1ar.calibration.lap_model_fit import LIFE_FILE
from f1ar.calibration.persona_fit import PERSONA_FILE, fit_wear
from f1ar.calibration.season_files import dataset_files, own_season, season_path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--season", type=int, default=2024)
    season = parser.parse_args().season
    dataset, splits_file = dataset_files(ROOT, season)
    manifest = yaml.safe_load(splits_file.read_text(encoding="utf-8"))
    train = list(manifest["splits"]["dry_train"])
    life_file = own_season(ROOT / "configs", LIFE_FILE, season)
    assert life_file is not None, f"fit the {season} W1 tyre life first (fit_tyre_life.py)"
    life = json.loads(life_file.read_text(encoding="utf-8"))
    reference = float(life["season_wear"]["lin"]["MEDIUM"])
    wear = fit_wear(pd.read_parquet(dataset), train, reference)
    cfg = {
        "label": f"{season} persona estimates (W5, D-045)",
        "fitted_on": train,
        "wear": wear,
        "derived_at_setup": {
            "consistency": f"rank of W2 lap noise (driver_pace_{season}_v1), persona spread",
            "overtake_skill": f"rank of W3 attack term (racecraft_{season}_v1), persona spread",
            "defense_skill": f"rank of W3 defend term (racecraft_{season}_v1), persona spread",
        },
        "not_measurable": {
            "mistake proneness (consistency x composure in incidents)": "blue flags dominate big "
            "losses (2024: 24 events left; odd vs even races r = -0.33)",
            "composure": "no measurement (penalties too few per driver)",
            "wet_skill": "3 wet races",
            "risk_appetite": "a preference, not a measurement",
            "qualifying_skill": "the grid is real",
        },
    }
    out = ROOT / "configs" / season_path(PERSONA_FILE, season)
    out.write_text(json.dumps(cfg, indent=1) + "\n", encoding="utf-8", newline="\n")
    body = {
        "model_name": "personas",
        "model_version": "v1",
        "kind": "tyre wear slopes vs teammate/team (shrunk by split-half reliability)",
        "source_type": "estimated_from_data",
        "artifact": out.relative_to(ROOT).as_posix(),
        "artifact_sha256": "sha256:" + hashlib.sha256(out.read_bytes()).hexdigest(),
        "feature_schema_version": 1,
        "training_dataset_manifest": splits_file.relative_to(ROOT).as_posix(),
        "training_dataset_content_sha256": manifest["file"]["content_sha256"],
        "training_races": sorted(train),
        "metrics": {
            "split_half_drivers": wear["split_half_drivers"],
            "split_half_teams": wear["split_half_teams"],
            "held_out_check": "EXP-014",
        },
        "status": "validated_for_engine_use (W5, D-045)"
        if season == 2024
        else f"fitted for {season} (H2a, D-054); held-out check EXP-023",
    }
    (ROOT / "configs" / "models" / f"personas_{season}_v1.yaml").write_text(
        yaml.safe_dump(body, sort_keys=False, width=100), encoding="utf-8", newline="\n"
    )
    for d, v in sorted(wear["drivers"].items(), key=lambda kv: kv[1]["wear_multiplier"]):
        print(f"{d} {v['team_id']:<20} wear x{v['wear_multiplier']:.3f}")
    print(
        f"split-half drivers {wear['split_half_drivers']}, teams {wear['split_half_teams']}; "
        f"wrote {out.relative_to(ROOT)}"
    )


if __name__ == "__main__":
    main()
