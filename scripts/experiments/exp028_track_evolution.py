"""EXP-028: track evolution in real races 2000-2025, and what our world is missing (H3b).

Per dry race (wet races left out), the common trend of the same-lap regression
(``calibration/track_evolution``), its SE from a driver bootstrap, and:

- **evolution**: 2000-2009 the trend with the fuel on board taken off (three variants: fuel
  estimated freely, fixed at the editorial 0.0825 s per lap of fuel, fixed at the season's fitted
  fuel effect that the engine uses); 2010-2025 the trend minus the fuel physics (0.03 s/kg x race
  fuel / laps; sensitivity 0.025 / 0.035 s/kg);
- **missing from the engine**: the measured trend minus the engine's common trend (2010+: its fuel
  term, ``-fuel_s_per_lap`` of the tyre shape; 2000-2009: none, the fuel term follows the fuel on
  board, so the trend with the season's fitted fuel taken off);
- **wear**: the race's own tyre-age slope (what an in-race estimator reads), to compare sizes.

Per era: mean, spread, spread net of the bootstrap noise, and how much a track's earlier years
predict a race's evolution (RMSE of "track's earlier mean" vs "all earlier races' mean").

Usage:
    uv run python scripts/experiments/exp028_track_evolution.py --bootstrap 30 \\
        --json docs/experiments/exp028.json
"""

import argparse
import functools
import json
import math
import statistics
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd

print = functools.partial(print, flush=True)  # noqa: A001 (progress through pipes)
ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "configs"
ERAS = {
    "2000-2009": range(2000, 2010),
    "2010-2017": range(2010, 2018),
    "2018-2025": range(2018, 2026),
}
EDITORIAL_FUEL = 0.0825
MAX_SE = 0.03
"""Races whose bootstrap SE exceeds this (few green laps: SC-heavy races, the bootstrap's lap
terms become ill-conditioned) stay in the per-race output but out of the era statistics."""


@functools.cache
def season_laps(season: int) -> pd.DataFrame:
    from f1ar.calibration.track_evolution import evolution_laps

    d = pd.read_parquet(ROOT / "data" / "processed" / "datasets" / f"laps_{season}_v1.parquet")
    return evolution_laps(d[d["group"] != "wet"])


@functools.cache
def season_fuel(season: int) -> float:
    """The fuel effect the engine runs on: per lap of fuel on board (< 2010) or per lap left."""
    from f1ar.calibration.lap_model_fit import load_life, load_shape

    if season <= 2009:
        life = load_life(CONFIGS, season)
        assert life is not None
        return float(life[0]["globals"]["fuel_s_per_lap"])
    return float(load_shape(CONFIGS, season)[0].fuel_s_per_lap)


def job(args: tuple[str, int, int]) -> dict[str, Any] | None:
    from f1ar.calibration.track_evolution import physics_fuel_slope, race_trend

    event, season, bootstrap = args
    g = season_laps(season)
    g = g[g["event_id"] == event]
    fit = race_trend(g, bootstrap=bootstrap, seed=season)
    if fit is None:
        return None
    row: dict[str, Any] = {**fit.__dict__, "engine_fuel": season_fuel(season)}
    if season <= 2009:
        fixed = {
            "editorial": race_trend(g, fuel_fixed=EDITORIAL_FUEL),
            "season_fit": race_trend(g, fuel_fixed=season_fuel(season), bootstrap=bootstrap),
        }
        row["evolution_variants"] = {
            "free_fuel": fit.evolution_s_per_lap,
            **{k: v.evolution_s_per_lap if v else None for k, v in fixed.items()},
        }
        sf = fixed["season_fit"]
        row["evolution_s_per_lap"] = sf.evolution_s_per_lap if sf else None
        row["trend_se_used"] = sf.trend_se if sf else None
        if sf:  # the wear slope with the engine's fuel effect off (free fuel confounds it)
            row["wear_s_per_lap"] = sf.wear_s_per_lap
        row["missing_from_engine"] = row["evolution_s_per_lap"]
    else:
        laps = fit.race_laps
        row["evolution_variants"] = {
            f"{kg_s}": round(
                fit.trend_s_per_lap - physics_fuel_slope(season, laps) * kg_s / 0.03, 5
            )
            for kg_s in (0.025, 0.03, 0.035)
        }
        row["trend_se_used"] = fit.trend_se
        row["missing_from_engine"] = round(fit.trend_s_per_lap + season_fuel(season), 5)
    return row


def spread(values: list[float], ses: list[float]) -> dict[str, float]:
    sd = statistics.stdev(values)
    noise = statistics.fmean(s * s for s in ses)
    return {
        "mean": round(statistics.fmean(values), 4),
        "sd": round(sd, 4),
        "noise_sd": round(math.sqrt(noise), 4),
        "true_sd": round(math.sqrt(max(0.0, sd * sd - noise)), 4),
    }


def predictability(rows: list[dict[str, Any]], key: str) -> dict[str, float | int]:
    """Races with earlier races at the same circuit: track's earlier mean vs all earlier mean."""
    ordered = sorted(rows, key=lambda r: (r["season"], r["event_id"]))
    err_track, err_all = [], []
    for i, r in enumerate(ordered):
        before = [x for x in ordered[:i] if x["season"] < r["season"]]
        same = [x[key] for x in before if x["circuit_id"] == r["circuit_id"]]
        if not same or len(before) < 10:
            continue
        err_track.append(r[key] - statistics.fmean(same))
        err_all.append(r[key] - statistics.fmean(x[key] for x in before))

    def rms(e: list[float]) -> float:
        return round(math.sqrt(statistics.fmean(x * x for x in e)), 4)

    return {
        "races": len(err_track),
        "rmse_track_mean": rms(err_track),
        "rmse_all_mean": rms(err_all),
    }


def summarise(all_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Era statistics on the races with a stable bootstrap (SE <= ``MAX_SE``)."""
    rows = [r for r in all_rows if r["trend_se_used"] <= MAX_SE]
    ses = [r["trend_se_used"] for r in rows]
    evo = [r["evolution_s_per_lap"] for r in rows]
    out: dict[str, Any] = {
        "races": len(rows),
        "unstable_races_left_out": sorted(
            r["event_id"] for r in all_rows if r["trend_se_used"] > MAX_SE
        ),
        "trend": spread([r["trend_s_per_lap"] for r in rows], [r["trend_se"] for r in rows]),
        "evolution": spread(evo, ses),
        "missing_from_engine": spread([r["missing_from_engine"] for r in rows], ses),
        "missing_over_race_s": round(
            statistics.fmean(r["missing_from_engine"] * r["race_laps"] for r in rows), 2
        ),
        "wear_s_per_lap": spread([r["wear_s_per_lap"] for r in rows], [0.0] * len(rows)),
        "evolution_predictability": predictability(rows, "evolution_s_per_lap"),
    }
    out["evolution_true_sd_vs_wear_mean"] = round(
        out["evolution"]["true_sd"] / abs(out["wear_s_per_lap"]["mean"]), 2
    )
    variants = rows[0]["evolution_variants"]
    out["evolution_variants_mean"] = {
        k: round(
            statistics.fmean(
                r["evolution_variants"][k] for r in rows if r["evolution_variants"][k] is not None
            ),
            4,
        )
        for k in variants
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--from", dest="first", type=int, default=2000)
    parser.add_argument("--to", dest="last", type=int, default=2025)
    parser.add_argument("--bootstrap", type=int, default=30)
    parser.add_argument("--workers", type=int, default=14)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    jobs = []
    for season in range(args.first, args.last + 1):
        events = sorted(season_laps(season)["event_id"].unique())
        jobs += [(e, season, args.bootstrap) for e in events]
    season_laps.cache_clear()
    print(f"EXP-028: {len(jobs)} dry races {args.first}-{args.last}, bootstrap {args.bootstrap}")
    t0 = time.perf_counter()
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(job, j) for j in jobs]
        for i, f in enumerate(as_completed(futures), 1):
            r = f.result()
            if r is not None and r["evolution_s_per_lap"] is not None:
                rows.append(r)
            if i % 25 == 0 or i == len(jobs):
                el = time.perf_counter() - t0
                eta = el / i * (len(jobs) - i) / 60
                print(f"  {i}/{len(jobs)} races, {el / 60:.1f} min, ETA {eta:.1f} min")
    out: dict[str, Any] = {"bootstrap": args.bootstrap, "eras": {}}
    for name, years in ERAS.items():
        mine = [r for r in rows if r["season"] in years]
        if len(mine) > 2:
            out["eras"][name] = summarise(mine)
    out["seasons"] = {}
    for s in sorted({r["season"] for r in rows}):
        mine = [r for r in rows if r["season"] == s]
        out["seasons"][str(s)] = {
            "races": len(mine),
            "trend": round(statistics.fmean(r["trend_s_per_lap"] for r in mine), 4),
            "evolution": round(statistics.fmean(r["evolution_s_per_lap"] for r in mine), 4),
            "missing_from_engine": round(
                statistics.fmean(r["missing_from_engine"] for r in mine), 4
            ),
            "engine_fuel": mine[0]["engine_fuel"],
        }
    out["races"] = {r["event_id"]: r for r in sorted(rows, key=lambda r: r["event_id"])}
    print(json.dumps(out["eras"], indent=1))
    print(json.dumps(out["seasons"], indent=1))
    print(f"EXP-028 done in {(time.perf_counter() - t0) / 60:.1f} min")
    if args.json:
        args.json.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
