"""Fit the race-day tyre priors and the surprise size (H3, D-058):
configs/calibration/race_day_prior_v1.json and its registry manifest.

For every race 2000-2025 with a tyre-life fit: the prior from the races strictly before it
(``calibration/race_day_prior``), the truth (the race's own fit) and the residual. Hyper-parameters
are chosen on targets in the tuning seasons (2000-2012) from a small grid and reported on
2013-2025; simpler predictors are reported next to it.

Usage:
    uv run python scripts/fit_race_day_prior.py
"""

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path

import yaml

from f1ar.calibration.lap_model_fit import LIFE_FILE
from f1ar.calibration.race_day_prior import (
    COMPOUNDS,
    ERAS,
    PRIOR_FILE,
    TUNING_SEASONS,
    PriorParams,
    Residual,
    era_of,
    load_records,
    residuals,
    score,
    surprise_size,
)
from f1ar.calibration.season_files import season_path

ROOT = Path(__file__).resolve().parents[1]
SEASONS = range(2000, 2026)
GRID = list(itertools.product((1.0, 3.0, 6.0, 12.0), (0.5, 1.0, 2.0, 4.0), (0.5, 0.7, 0.85, 1.0)))


def split_scores(res: list[Residual]) -> dict[str, float]:
    tune = [x for x in res if x.record.season in TUNING_SEASONS]
    rest = [x for x in res if x.record.season not in TUNING_SEASONS]
    return {"tuning": round(score(tune), 4), "reported": round(score(rest), 4)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=ROOT / "configs" / PRIOR_FILE)
    args = parser.parse_args()
    configs = ROOT / "configs"
    records = load_records(configs, SEASONS)
    print(f"{len(records)} races with a tyre-life fit")
    trials = []
    for ss, ts, rc in GRID:
        p = PriorParams(ss, ts, rc)
        trials.append((split_scores(residuals(records, p)), p))
    trials.sort(key=lambda t: t[0]["tuning"])
    chosen_scores, chosen = trials[0]
    print(f"chosen {chosen}: {chosen_scores}")
    res = residuals(records, chosen)
    baselines = {
        "no_track_effect": split_scores(
            residuals(records, PriorParams(chosen.season_shrink, 1e9, chosen.recency))
        ),
        "track_only_no_recency": split_scores(
            residuals(records, PriorParams(chosen.season_shrink, chosen.track_shrink, 1.0))
        ),
    }
    overall = surprise_size(res)
    eras = {
        name: surprise_size([x for x in res if x.record.season in years])
        for name, years in ERAS.items()
    }
    by_history = {
        "new_track": surprise_size([x for x in res if x.prior.track_years == 0]),
        "track_with_history": surprise_size([x for x in res if x.prior.track_years > 0]),
        "season_opener": surprise_size([x for x in res if x.prior.season_races == 0]),
    }
    events = {}
    for x in res:
        r = x.record
        events[r.event_id] = {
            "season": r.season,
            "round": r.round,
            "circuit_id": r.circuit_id,
            "split": r.split,
            "era": era_of(r.season),
            "track_years": x.prior.track_years,
            "season_races_before": x.prior.season_races,
            "prior": {
                "life_raw": {c: round(math.exp(x.prior.values[c]), 3) for c in COMPOUNDS},
                "wear_multiplier": round(math.exp(x.prior.values["WEAR"]), 4),
            },
            "truth": {
                "life_raw": {c: r.life_raw[c] for c in COMPOUNDS},
                "wear_multiplier": r.wear,
                "measured": [c for c in COMPOUNDS if r.measured[c]],
            },
            "surprise": {
                "life_shared_log": round(x.shared, 4),
                "wear_log": round(x.wear, 4),
            },
        }
    sources = {}
    for season in SEASONS:
        path = configs / season_path(LIFE_FILE, season)
        sources[str(season)] = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    out = {
        "label": "race-day tyre priors from history before each race, and the surprise (H3, D-058)",
        "source_type": "estimated_from_data",
        "method": (
            "log life_raw per compound and log wear_multiplier = season level (random walk, "
            "shrunk toward the previous season) + track effect (recency-weighted, shrunk to 0), "
            "backfitted on the races strictly before each target (earlier seasons and earlier "
            "rounds); hyper-parameters chosen on 2000-2012 targets"
        ),
        "params": {
            "season_shrink": chosen.season_shrink,
            "track_shrink": chosen.track_shrink,
            "recency": chosen.recency,
        },
        "scores": {"chosen": chosen_scores, **baselines},
        "grid": [{"params": [p.season_shrink, p.track_shrink, p.recency], **s} for s, p in trials],
        "surprise": {"all": overall, "eras": eras, **by_history},
        "caveats": [
            "truth = the race's own tyre-life fit (estimated_from_data, with its own estimation "
            "noise: the wear multiplier especially)",
            "season-level calibration (kappa, wear curve shape, compound ratios) is fitted on "
            "the whole season and treated as known calibration, not race-day information",
            "before 2018 the compounds are one measured stint length times 2018's ratios: no "
            "per-compound surprise",
            "before 2010 life_raw (stint length) also carries the refuelling strategy",
        ],
        "tyre_life_files": sources,
        "events": events,
    }
    text = json.dumps(out, indent=1) + "\n"
    args.out.write_text(text, encoding="utf-8", newline="\n")
    sha = "sha256:" + hashlib.sha256(text.encode()).hexdigest()
    print(json.dumps(out["surprise"], indent=1))
    manifest = {
        "model_name": "race_day_prior",
        "model_version": "v1",
        "kind": "track + season random-walk prior on log tyre life and wear, history before a race",
        "source_type": "estimated_from_data",
        "artifact": (Path("configs") / PRIOR_FILE).as_posix(),
        "artifact_sha256": sha,
        "feature_schema_version": 1,
        "training_dataset_manifest": "configs/calibration/tyre_life_<season>_v1.json (2000-2025)",
        "training_races": sorted(events),
        "metrics": {
            "life_shared_sd": overall["life_shared_sd"],
            "life_compound_sd": overall["life_compound_sd"],
            "wear_sd": overall["wear_sd"],
            "held_out_check": "EXP-027",
        },
        "status": "fitted (H3, D-058); value of information EXP-027",
    }
    manifest_path = configs / "models" / "race_day_prior_v1.yaml"
    manifest_path.write_text(
        yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
        newline="\n",
    )
    print(f"wrote {args.out} and {manifest_path}")


if __name__ == "__main__":
    main()
