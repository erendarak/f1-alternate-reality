"""EXP-003: pace models on the 2024 season (M4 slice 2).

Two tasks (docs/EXPERIMENTS.md, EXP-003):

1. **Engine shape**: how well do tyre/fuel coefficients learned on other races explain a race?
   Shapes: the engine's placeholder (editorial), fitted linear wear, fitted quadratic wear, and
   quadratic + a per-race adjustment estimated on that race (in-sample, as the engine would).
   Per-driver race pace is fitted on the evaluated race for every shape (in-sample nuisance).
2. **Online lap prediction**: predict each lap from what was known when it started. Rolling
   median (A0), rolling median + shape correction (A), LightGBM on the residual, with ablations
   (no weather, + driver/team identity, + editorial persona vector).

Evaluation: leave-one-race-out over the 16 dry_train races, then a final fit on all of them
evaluated on the 5 dry_test races. Wet races are excluded (weather is slice 4).

Usage:
    uv run python scripts/experiments/exp003_pace_models.py
    uv run python scripts/experiments/exp003_pace_models.py --json docs/experiments/exp003.json
"""

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml

from f1ar.calibration.lap_dataset import FEATURE_SCHEMA_VERSION
from f1ar.calibration.pace_features import add_recent_features
from f1ar.calibration.pace_models import (
    DRY,
    TyreShape,
    fit_race_adjustment,
    fit_shape,
    online_rows,
    predict_rolling,
    predict_rolling_corrected,
    shape_residuals,
)
from f1ar.domain.driver_profiles import ARCHETYPE_FIELDS
from f1ar.simulation.scenarios import BELGIUM_2024, MONZA_2024, default_lap_model, load_event

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "data" / "processed" / "datasets" / "laps_2024_v1.parquet"
DATASET_MANIFEST = ROOT / "data" / "manifests" / "dataset_laps_2024_v1.yaml"
SEED = 42
BOOTSTRAP = 1000

LGB_PARAMS: dict[str, Any] = {
    "objective": "regression_l1",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "seed": SEED,
    "deterministic": True,
    "force_row_wise": True,
    "num_threads": 1,
    "verbosity": -1,
}
ROUNDS = 400

BASE = [
    "age",
    "fuel",
    "compound",
    "drv_recent_compound",
    "compound_changed",
    "d_age",
    "d_fuel",
    "stint_lap",
    "fresh_tyre",
    "gap_ahead_s",
    "gap_behind_prev_s",
    "dirty",
    "position_start",
    "lap_number",
    "scheduled_laps",
    "drv_recent_n",
    "field_vs_driver",
]
WEATHER = ["air_temp_c", "track_temp_c", "humidity_pct", "wind_speed_ms"]
IDENTITY = ["driver_id", "team_id"]
PERSONA = [f"persona_{f}" for f in ARCHETYPE_FIELDS]
CATEGORICAL = ["compound", "drv_recent_compound", "driver_id", "team_id"]

VARIANTS: dict[str, list[str]] = {
    "gbm": BASE + WEATHER,
    "gbm_no_weather": BASE,
    "gbm_identity": BASE + WEATHER + IDENTITY,
    "gbm_persona": BASE + WEATHER + PERSONA,
}


# --- data ----------------------------------------------------------------------------------------


def personas() -> pd.DataFrame:
    """Editorial persona values per person (Monza + Spa 2024 seed files; others missing)."""
    rows: dict[str, dict[str, float]] = {}
    for event in (MONZA_2024, BELGIUM_2024):
        _, profiles = load_event(ROOT / "configs", event)
        for p in profiles.profiles:
            rows[p.person_id] = {f"persona_{k}": v for k, v in p.archetype.model_dump().items()}
    return pd.DataFrame.from_dict(rows, orient="index")


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    data = add_recent_features(pd.read_parquet(DATASET))
    dry = data[data["group"].isin(["dry_train", "dry_test"])].copy()
    rows = online_rows(dry).copy()
    rows["compound_changed"] = (rows["compound"] != rows["drv_recent_compound"]).astype(int)
    rows["d_age"] = rows["age"] - rows["drv_recent_age"].astype(float)
    rows["d_fuel"] = rows["fuel"] - rows["drv_recent_fuel"].astype(float)
    rows["field_vs_driver"] = rows["field_recent_s"].astype(float) - rows["drv_recent_s"].astype(
        float
    )
    rows["person_id"] = rows["driver_id"].astype(str).str.rsplit("_", n=1).str[0]
    rows = rows.join(personas(), on="person_id")
    for col in CATEGORICAL:
        rows[col] = pd.Categorical(rows[col].astype(str))
    for col in BASE + WEATHER + PERSONA:
        if col not in CATEGORICAL:
            rows[col] = pd.to_numeric(rows[col], errors="coerce").astype(float)
    return dry, rows


def races(df: pd.DataFrame, group: str) -> list[str]:
    g = df[df["group"] == group]
    return list(g.sort_values("round")["event_id"].astype(str).unique())


def bootstrap_ci(per_race: dict[str, list[float]], rng: np.random.Generator) -> list[float]:
    """95 % interval of the pooled MAE when races are resampled (|error| lists per race)."""
    names = list(per_race)
    stats = []
    for _ in range(BOOTSTRAP):
        pick = rng.choice(len(names), size=len(names), replace=True)
        pooled = np.concatenate([per_race[names[i]] for i in pick])
        stats.append(float(np.mean(pooled)))
    return [round(float(np.percentile(stats, 2.5)), 3), round(float(np.percentile(stats, 97.5)), 3)]


# --- task 1: engine shape ---------------------------------------------------------------------


def shape_task(dry: pd.DataFrame, rng: np.random.Generator) -> dict[str, Any]:
    train = races(dry, "dry_train")
    test = races(dry, "dry_test")
    placeholder = TyreShape.placeholder(default_lap_model())
    names = ["placeholder", "linear", "quadratic", "quadratic+race"]
    per_race: dict[str, dict[str, list[float]]] = {n: {} for n in names}

    def evaluate(event: str, fit_on: list[str]) -> None:
        held = dry[dry["event_id"] == event]
        fitted_on = dry[dry["event_id"].isin(fit_on)]
        shapes = {
            "placeholder": placeholder,
            "linear": fit_shape(fitted_on, quadratic=False),
            "quadratic": fit_shape(fitted_on),
        }
        shapes["quadratic+race"] = fit_race_adjustment(held, shapes["quadratic"]).apply(
            shapes["quadratic"]
        )
        for n, s in shapes.items():
            per_race[n][event] = list(np.abs(shape_residuals(held, s)))

    for e in train:
        evaluate(e, [x for x in train if x != e])
    for e in test:
        evaluate(e, train)

    def summary(events: list[str]) -> dict[str, Any]:
        return {
            n: {
                "mae": round(float(np.mean(np.concatenate([per_race[n][e] for e in events]))), 3),
                "ci95": bootstrap_ci({e: per_race[n][e] for e in events}, rng),
            }
            for n in names
        }

    final = fit_shape(dry[dry["event_id"].isin(train)], label="season 2024 dry_train, quadratic")
    # coefficient uncertainty: refit on races resampled with replacement
    coef_samples = []
    for _ in range(200):
        pick = rng.choice(train, size=len(train), replace=True)
        sample = pd.concat(
            [dry[dry["event_id"] == e].assign(event_id=f"{e}#{i}") for i, e in enumerate(pick)]
        )
        coef_samples.append(json.loads(fit_shape(sample).to_json()))

    def ci(path: list[str]) -> list[float]:
        vals = []
        for s in coef_samples:
            v: Any = s
            for k in path:
                v = v[k]
            vals.append(float(v))
        return [
            round(float(np.percentile(vals, 2.5)), 4),
            round(float(np.percentile(vals, 97.5)), 4),
        ]

    coefficients = {
        "fuel_s_per_lap": [round(final.fuel_s_per_lap, 4), ci(["fuel_s_per_lap"])],
        "dirty_air_s": [round(final.dirty_air_s, 4), ci(["dirty_air_s"])],
        **{f"offset_{c}": [round(final.offset[c], 4), ci(["offset", c])] for c in DRY},
        **{f"wear_lin_{c}": [round(final.wear_lin[c], 4), ci(["wear_lin", c])] for c in DRY},
        **{f"wear_quad_{c}": [round(final.wear_quad[c], 5), ci(["wear_quad", c])] for c in DRY},
    }
    adjustments = {
        e: {
            "wear_multiplier": round(a.wear_multiplier, 3),
            "offset_delta": {k: round(v, 3) for k, v in a.offset_delta.items()},
        }
        for e in train + test
        for a in [fit_race_adjustment(dry[dry["event_id"] == e], final)]
    }
    per_track = {
        e: {n: round(float(np.mean(per_race[n][e])), 3) for n in names} for e in train + test
    }
    return {
        "loro_dry_train": summary(train),
        "dry_test": summary(test),
        "per_race_mae": per_track,
        "final_shape": json.loads(final.to_json()),
        "coefficients_with_race_bootstrap_ci95": coefficients,
        "race_adjustments_in_sample": adjustments,
        "_final": final,
    }


# --- task 2: online prediction ----------------------------------------------------------------


def train_gbm(train: pd.DataFrame, features: list[str], **extra: Any) -> lgb.Booster:
    target = train["y"] - train["drv_recent_s"].astype(float)
    ds = lgb.Dataset(
        train[features],
        label=target,
        categorical_feature=[c for c in features if c in CATEGORICAL],
        free_raw_data=False,
    )
    return lgb.train({**LGB_PARAMS, **extra}, ds, num_boost_round=ROUNDS)


def gbm_predict(model: lgb.Booster, rows: pd.DataFrame, features: list[str]) -> np.ndarray:
    out: np.ndarray = rows["drv_recent_s"].astype(float).to_numpy() + np.asarray(
        model.predict(rows[features])
    )
    return out


def online_task(dry: pd.DataFrame, rows: pd.DataFrame, rng: np.random.Generator) -> dict[str, Any]:
    train = races(dry, "dry_train")
    test = races(dry, "dry_test")
    names = ["rolling_median", "rolling+shape", *VARIANTS]
    per_race: dict[str, dict[str, list[float]]] = {n: {} for n in names}
    signed: dict[str, np.ndarray] = {}  # gbm residuals (truth - prediction) per race

    def evaluate(held: list[str], fit_on: list[str]) -> dict[str, lgb.Booster]:
        shape = fit_shape(dry[dry["event_id"].isin(fit_on)])
        fit_rows = rows[rows["event_id"].isin(fit_on)]
        models = {v: train_gbm(fit_rows, f) for v, f in VARIANTS.items()}
        for e in held:
            r = rows[rows["event_id"] == e]
            y = r["y"].to_numpy()
            preds = {
                "rolling_median": predict_rolling(r),
                "rolling+shape": predict_rolling_corrected(r, shape),
                **{v: gbm_predict(models[v], r, f) for v, f in VARIANTS.items()},
            }
            for n, p in preds.items():
                per_race[n][e] = list(np.abs(p - y))
            signed[e] = y - preds["gbm"]
        return models

    for e in train:
        evaluate([e], [x for x in train if x != e])
    final_models = evaluate(test, train)

    def summary(events: list[str]) -> dict[str, Any]:
        out = {}
        for n in names:
            pooled = np.concatenate([per_race[n][e] for e in events])
            out[n] = {
                "n": int(len(pooled)),
                "mae": round(float(np.mean(pooled)), 3),
                "median_ae": round(float(np.median(pooled)), 3),
                "rmse": round(float(np.sqrt(np.mean(pooled**2))), 3),
                "ci95": bootstrap_ci({e: per_race[n][e] for e in events}, rng),
            }
        return out

    # p10 / p90 on the final fit, coverage on dry_test
    fit_rows = rows[rows["event_id"].isin(train)]
    test_rows = rows[rows["event_id"].isin(test)]
    feats = VARIANTS["gbm"]
    q10 = train_gbm(fit_rows, feats, objective="quantile", alpha=0.1)
    q90 = train_gbm(fit_rows, feats, objective="quantile", alpha=0.9)
    lo, hi = gbm_predict(q10, test_rows, feats), gbm_predict(q90, test_rows, feats)
    y = test_rows["y"].to_numpy()
    coverage: dict[str, Any] = {
        "p10_p90_interval_coverage": round(float(np.mean((y >= lo) & (y <= hi))), 3),
        "below_p10": round(float(np.mean(y < lo)), 3),
        "above_p90": round(float(np.mean(y > hi)), 3),
        "median_width_s": round(float(np.median(hi - lo)), 3),
    }
    # split-conformal interval: the 10th/90th percentile of the leave-one-race-out residuals of
    # the same model, added to its prediction on the test races
    loro = np.concatenate([signed[e] for e in train])
    r10, r90 = float(np.percentile(loro, 10)), float(np.percentile(loro, 90))
    pred = gbm_predict(final_models["gbm"], test_rows, feats)
    c_lo, c_hi = pred + r10, pred + r90
    coverage["conformal_p10_p90_coverage"] = round(float(np.mean((y >= c_lo) & (y <= c_hi))), 3)
    coverage["conformal_offsets_s"] = [round(r10, 3), round(r90, 3)]
    coverage["conformal_width_s"] = round(r90 - r10, 3)
    gain = final_models["gbm"].feature_importance(importance_type="gain")
    importance = sorted(
        ((f, round(float(g) / float(sum(gain)), 3)) for f, g in zip(feats, gain, strict=True)),
        key=lambda x: -x[1],
    )
    persona_missing = float(rows[PERSONA[0]].isna().mean())
    per_track = {
        e: {n: round(float(np.mean(per_race[n][e])), 3) for n in names} for e in train + test
    }
    per_driver: dict[str, dict[str, float]] = {}
    test_with_err = test_rows.assign(
        err_gbm=np.abs(gbm_predict(final_models["gbm"], test_rows, feats) - y)
    )
    for code, g in test_with_err.groupby("driver_code", observed=True):
        per_driver[str(code)] = {"n": int(len(g)), "mae_gbm": round(float(g["err_gbm"].mean()), 3)}
    return {
        "loro_dry_train": summary(train),
        "dry_test": summary(test),
        "per_race_mae": per_track,
        "per_driver_dry_test_gbm": dict(
            sorted(per_driver.items(), key=lambda kv: kv[1]["mae_gbm"])
        ),
        "quantiles_dry_test": coverage,
        "feature_importance_gain_share": importance,
        "persona_missing_share": round(persona_missing, 3),
        "_final": final_models["gbm"],
        "_q": (q10, q90),
    }


# --- manifests ---------------------------------------------------------------------------------


def sha(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def write_artifacts(shape_res: dict[str, Any], online_res: dict[str, Any]) -> list[str]:
    dataset = yaml.safe_load(DATASET_MANIFEST.read_text(encoding="utf-8"))
    content = dataset["file"]["content_sha256"]
    written = []

    shape: TyreShape = shape_res["_final"]
    shape_path = ROOT / "configs" / "calibration" / "tyre_shape_2024_v1.json"
    shape_path.parent.mkdir(parents=True, exist_ok=True)
    shape_text = shape.to_json() + "\n"
    shape_path.write_text(shape_text, encoding="utf-8", newline="\n")
    written.append(str(shape_path.relative_to(ROOT)))

    model: lgb.Booster = online_res["_final"]
    art_dir = ROOT / "artifacts" / "models"
    art_dir.mkdir(parents=True, exist_ok=True)
    model_path = art_dir / "pace_lgbm_2024_v1.txt"
    model_text = model.model_to_string()
    model_path.write_text(model_text, encoding="utf-8", newline="\n")

    config = {
        "lgb_params": LGB_PARAMS,
        "rounds": ROUNDS,
        "features": VARIANTS["gbm"],
        "target": "lap_time_s - drv_recent_s",
    }
    manifests = {
        "tyre_shape_2024_v1": {
            "model_name": "tyre_shape",
            "model_version": "v1",
            "kind": "structural least squares (driver-race pace absorbed)",
            "source_type": "estimated_from_data",
            "artifact": str(shape_path.relative_to(ROOT).as_posix()),
            "artifact_sha256": sha(shape_text.encode()),
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "training_dataset_manifest": DATASET_MANIFEST.relative_to(ROOT).as_posix(),
            "training_dataset_content_sha256": content,
            "training_races": list(shape.fitted_on),
            "metrics": {"dry_test": shape_res["dry_test"]},
            "status": "validated_for_engine_use (slice 3)",
        },
        "pace_lgbm_2024_v1": {
            "model_name": "pace_predictor_online",
            "model_version": "v1",
            "kind": "LightGBM residual on the driver's recent median",
            "source_type": "learned_parameter",
            "artifact": "artifacts/models/pace_lgbm_2024_v1.txt",
            "artifact_tracked": False,  # gitignored: rebuilt locally by this script
            "artifact_sha256": sha(model_text.encode()),
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "training_dataset_manifest": DATASET_MANIFEST.relative_to(ROOT).as_posix(),
            "training_dataset_content_sha256": content,
            "training_config_sha256": sha(json.dumps(config, sort_keys=True).encode()),
            "seed": SEED,
            "config": config,
            "metrics": {
                "dry_test": online_res["dry_test"]["gbm"],
                "quantiles": online_res["quantiles_dry_test"],
            },
            "status": "experimental",
        },
    }
    reg = ROOT / "configs" / "models"
    reg.mkdir(parents=True, exist_ok=True)
    for name, m in manifests.items():
        path = reg / f"{name}.yaml"
        path.write_text(
            yaml.safe_dump(m, sort_keys=False, width=100), encoding="utf-8", newline="\n"
        )
        written.append(str(path.relative_to(ROOT)))
    return written


def public(d: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in d.items() if not k.startswith("_")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if not DATASET.exists():
        parser.error("build the dataset first: scripts/build_lap_dataset.py")
    started = time.monotonic()
    rng = np.random.default_rng(SEED)
    dry, rows = load()
    shape_res = shape_task(dry, rng)
    online_res = online_task(dry, rows, rng)
    written = write_artifacts(shape_res, online_res)
    report = {
        "experiment": "EXP-003",
        "dataset_content_sha256": yaml.safe_load(DATASET_MANIFEST.read_text(encoding="utf-8"))[
            "file"
        ]["content_sha256"],
        "races": {"dry_train": races(dry, "dry_train"), "dry_test": races(dry, "dry_test")},
        "shape": public(shape_res),
        "online": public(online_res),
    }
    print(
        json.dumps(
            {
                "shape": {k: report["shape"][k] for k in ("loro_dry_train", "dry_test")},
                "online": {
                    k: report["online"][k]
                    for k in (
                        "loro_dry_train",
                        "dry_test",
                        "quantiles_dry_test",
                        "persona_missing_share",
                    )
                },
            },
            indent=1,
        )
    )
    print("coefficients:", json.dumps(report["shape"]["coefficients_with_race_bootstrap_ci95"]))
    print("importance:", report["online"]["feature_importance_gain_share"][:8])
    print("wrote:", written, f"({time.monotonic() - started:.0f} s)")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
