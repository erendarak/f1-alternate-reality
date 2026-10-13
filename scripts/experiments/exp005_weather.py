"""EXP-005: the weather model on the 2024 wet races (M4 slice 4, D-034).

1. Fit wetting/drying rates and slick/intermediate penalty lines on all three wet races (lap
   times + the teams' tyre choices) and write configs/calibration/weather_2024_v1.json with its
   manifest.
2. Leave one wet race out: fit on two, measure lap-time residuals (pace per driver in-sample)
   and how well the model predicts the held-out race's tyre choices.
3. Simulate the three races with engine v0.8.0 (30 seeds): when does the field switch tyres
   compared with the real race, how much of the race is on wet tyres, order, race time.

Usage:
    uv run python scripts/experiments/exp005_weather.py --json docs/experiments/exp005.json
"""

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from f1ar.calibration.pace_models import TyreShape
from f1ar.calibration.weather_fit import (
    WEATHER_FILE,
    choice_residuals,
    fit_weather,
    no_weather_residuals,
    residuals,
    wet_laps,
)
from f1ar.domain.race_state import WorldStatus
from f1ar.simulation import lightweight_engine as engine
from f1ar.simulation.scenarios import event_data_dir, load_event, setup_from_codes
from f1ar.simulation.weather import crossovers

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "data" / "processed" / "datasets" / "laps_2024_v1.parquet"
WET_TYRES = ("INTERMEDIATE", "WET")


def load() -> tuple[pd.DataFrame, dict[str, pd.DataFrame], TyreShape]:
    d = pd.read_parquet(DATASET)
    wet = d[d["group"] == "wet"]
    weather = {
        e: pd.read_parquet(ROOT / "data" / "processed" / e / "weather.parquet")
        for e in wet["event_id"].unique()
    }
    shape = TyreShape.from_json(
        (ROOT / "configs" / "calibration" / "tyre_shape_2024_v1.json").read_text(encoding="utf-8")
    )
    return wet, weather, shape


def choice_accuracy(data: Any, params: Any) -> float:
    """Mean probability the model gives to the tyre type each team actually chose."""
    r = choice_residuals(data, params)
    return float(np.mean(np.exp(-(r**2) / 2))) if len(r) else float("nan")


def loro(wet: pd.DataFrame, weather: dict[str, pd.DataFrame], shape: TyreShape) -> dict[str, Any]:
    races = sorted(wet["event_id"].unique())
    out = {}
    for held in races:
        train = wet[wet["event_id"] != held]
        test = wet[wet["event_id"] == held]
        params, _ = fit_weather(wet_laps(train, weather, shape))
        data = wet_laps(test, weather, shape)
        out[held] = {
            "lap_mae_weather_s": round(float(np.mean(np.abs(residuals(data, params)))), 3),
            "lap_mae_no_weather_s": round(float(np.mean(np.abs(no_weather_residuals(data)))), 3),
            "choice_p_chosen": round(choice_accuracy(data, params), 3),
            "crossovers": [round(x, 3) for x in crossovers(params)],
        }
    return out


def wet_share_by_lap(compounds: dict[str, dict[int, str]], laps: int) -> list[float]:
    """Share of running cars on wet-weather tyres on each lap (1..laps)."""
    out = []
    for lap in range(1, laps + 1):
        on = [c[lap] for c in compounds.values() if lap in c]
        out.append(sum(x in WET_TYRES for x in on) / len(on) if on else float("nan"))
    return out


def full_wet_share(compounds: dict[str, dict[int, str]]) -> float:
    """Share of all car-laps on full wets (inters and wets are counted together above)."""
    laps = [c for car in compounds.values() for c in car.values()]
    return sum(c == "WET" for c in laps) / len(laps) if laps else float("nan")


def switches(share: list[float]) -> dict[str, int | None]:
    """First lap the majority goes onto wet tyres, and first lap after that it's back on slicks."""
    to_wet = next((i + 1 for i, s in enumerate(share) if s > 0.5), None)
    start = (to_wet or 1) - 1
    back = next((i + 1 for i, s in enumerate(share) if i >= start and s < 0.5), None)
    if share and share[0] > 0.5:
        to_wet = 1
        back = next((i + 1 for i, s in enumerate(share) if s < 0.5), None)
    return {"majority_on_wet_from_lap": to_wet, "majority_back_on_slicks_lap": back}


def real_compounds(event: str) -> tuple[dict[str, dict[int, str]], int]:
    laps = pd.read_parquet(ROOT / "data" / "processed" / event / "laps.parquet")
    out: dict[str, dict[int, str]] = {}
    for r in laps.to_dict("records"):
        if not pd.isna(r["compound"]):
            out.setdefault(str(r["driver_code"]), {})[int(r["lap_number"])] = str(r["compound"])
    return out, int(laps["lap_number"].max())


def sim_race(event: str, seed: int) -> dict[str, Any]:
    roster, profiles = load_event(ROOT / "configs", event)
    setup = setup_from_codes(roster, profiles, None, data_dir=event_data_dir(ROOT, event))
    state = engine.start(engine.create_world(setup, seed=seed, world_id="w"))
    code = {d.driver_id: d.code for d in setup.drivers}
    tyres: dict[str, dict[int, str]] = {
        code[c.driver_id]: {1: c.compound.value} for c in state.cars
    }
    current = {code[c.driver_id]: c.compound.value for c in state.cars}
    stops = {k: 0 for k in current}
    rain_events = 0
    while state.world_status is WorldStatus.RUNNING:
        result = engine.step(state)
        state = result.state
        for e in result.events:
            if e.type == "PIT_STOP" and e.driver_id:
                current[code[e.driver_id]] = str(e.payload["to_compound"])
                stops[code[e.driver_id]] += 1
            elif e.type == "LAP_COMPLETED" and e.driver_id:
                lap = int(e.payload["lap"]) + 1
                tyres[code[e.driver_id]][lap] = current[code[e.driver_id]]
            elif e.type in ("RAIN_STARTED", "RAIN_STOPPED"):
                rain_events += 1
    finishers = [c for c in state.cars if c.finish_time_s is not None]
    winner = min(finishers, key=lambda c: (-c.laps_completed, c.finish_time_s or 0.0))
    return {
        "tyres": tyres,
        "stops": statistics.fmean(stops[code[c.driver_id]] for c in finishers),
        "winner": code[winner.driver_id],
        "winner_time_min": (winner.finish_time_s or 0.0) / 60,
        "rain_events": rain_events,
        "weather_note": setup.inputs.get("weather", ""),
        "total_laps": setup.track.total_laps,
    }


def simulate(event: str, seeds: int) -> dict[str, Any]:
    real, total = real_compounds(event)
    real_share = wet_share_by_lap(real, total)
    sims = [sim_race(event, s) for s in range(1, seeds + 1)]
    shares = [wet_share_by_lap(s["tyres"], total) for s in sims]
    sw = [switches(sh) for sh in shares]

    def describe(vals: list[float]) -> dict[str, float]:
        v = sorted(x for x in vals if x is not None and not np.isnan(x))
        if not v:
            return {"n": 0}
        return {"n": len(v), "mean": round(statistics.fmean(v), 2), "min": v[0], "max": v[-1]}

    return {
        "laps": total,
        "real": {
            **switches(real_share),
            "wet_tyre_lap_share": round(float(np.nanmean(real_share)), 3),
            "full_wet_lap_share": round(full_wet_share(real), 3),
        },
        "sim": {
            "majority_on_wet_from_lap": describe(
                [float(s["majority_on_wet_from_lap"]) for s in sw if s["majority_on_wet_from_lap"]]
            ),
            "majority_back_on_slicks_lap": describe(
                [
                    float(s["majority_back_on_slicks_lap"])
                    for s in sw
                    if s["majority_back_on_slicks_lap"]
                ]
            ),
            "wet_tyre_lap_share": describe([float(np.nanmean(sh)) for sh in shares]),
            "full_wet_lap_share": describe([full_wet_share(s["tyres"]) for s in sims]),
            "stops_per_finisher": describe([s["stops"] for s in sims]),
            "winner_time_min": describe([s["winner_time_min"] for s in sims]),
            "winners": dict(pd.Series([s["winner"] for s in sims]).value_counts()),
            "rain_events": sims[0]["rain_events"],
        },
        "real_share_by_lap": [round(x, 2) for x in real_share],
        "sim_mean_share_by_lap": [
            round(float(np.nanmean([sh[i] for sh in shares])), 2) for i in range(total)
        ],
        "weather_note": sims[0]["weather_note"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    wet, weather, shape = load()
    params, info = fit_weather(wet_laps(wet, weather, shape))
    text = params.model_dump_json(indent=2) + "\n"
    path = ROOT / "configs" / WEATHER_FILE
    path.write_text(text, encoding="utf-8", newline="\n")
    dataset = yaml.safe_load(
        (ROOT / "data" / "manifests" / "dataset_laps_2024_v1.yaml").read_text(encoding="utf-8")
    )
    manifest = {
        "model_name": "weather",
        "model_version": "v1",
        "kind": "non-linear least squares: lap times + tyre choices (wetness from the rain flag)",
        "source_type": "estimated_from_data (wet-tyre line and factors editorial)",
        "artifact": (Path("configs") / WEATHER_FILE).as_posix(),
        "artifact_sha256": "sha256:" + hashlib.sha256(text.encode()).hexdigest(),
        "training_dataset_content_sha256": dataset["file"]["content_sha256"],
        "training_races": sorted(wet["event_id"].unique()),
        "fit": info,
        "crossovers": [round(x, 3) for x in crossovers(params)],
        "status": "experimental (3 races)",
    }
    (ROOT / "configs" / "models" / "weather_2024_v1.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False, width=100), encoding="utf-8", newline="\n"
    )
    print("fit:", info, "crossovers", manifest["crossovers"])
    print(
        "params:", {k: round(v, 4) for k, v in params.model_dump().items() if isinstance(v, float)}
    )

    held_out = loro(wet, weather, shape)
    print("leave-one-race-out:", json.dumps(held_out, indent=1))
    sims = {e: simulate(e, args.seeds) for e in sorted(wet["event_id"].unique())}
    for e, r in sims.items():
        print(f"\n== {e}  real {r['real']}")
        print("   sim ", {k: v for k, v in r["sim"].items()})
        print("   real share", r["real_share_by_lap"][::3])
        print("   sim  share", r["sim_mean_share_by_lap"][::3])
    if args.json:
        args.json.write_text(
            json.dumps(
                {
                    "fit": manifest,
                    "params": json.loads(text),
                    "leave_one_race_out": held_out,
                    "simulation": sims,
                },
                indent=2,
                default=str,
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )


if __name__ == "__main__":
    main()
