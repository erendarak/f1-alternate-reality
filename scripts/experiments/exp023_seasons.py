"""EXP-023: races of 2018-2025 simulated with their own season's fits vs reality (H2a, D-054).

Per season, its dry_test races (never used by the simulation-fitted values: tyre life kappa/delta,
risk scale, pass shift and launch spread; the W3 encounter terms and W4 rates use them, as in
2024), ``--seeds`` per-race seeds, two variants:

- ``season``: the season's own fits (H2a);
- ``borrowed``: the same race without them (a configs copy without the season's files): 2024's
  tyre model, editorial driver pace / racecraft / crews / personas, i.e. what H2a replaces.

Real vs simulated per race: stops per finisher, passes, retirements, safety cars, VSCs, and the
Spearman correlation of the finishing order with the real one.

Usage:
    uv run python scripts/experiments/exp023_seasons.py --seasons 2019 2020 2021 \\
        --seeds 6 --json docs/experiments/exp023.json
"""

import argparse
import functools
import hashlib
import json
import shutil
import statistics
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import yaml

from f1ar.calibration.reference_metrics import real_metrics
from f1ar.calibration.season_files import dataset_files
from f1ar.calibration.simulated_metrics import sim_metrics, spearman

print = functools.partial(print, flush=True)  # noqa: A001 (progress through pipes)
ROOT = Path(__file__).resolve().parents[2]
METRICS = ("stops", "passes", "retirements", "safety_cars", "vscs")


def event_seed(event: str, k: int) -> int:
    return int(hashlib.sha256(event.encode()).hexdigest()[:6], 16) * 100 + k


@functools.cache
def _setup(event: str, configs: str) -> Any:
    from f1ar.simulation.scenarios import event_data_dir, load_event, setup_from_codes

    roster, profiles = load_event(Path(configs), event)
    return setup_from_codes(
        roster, profiles, None, data_dir=event_data_dir(ROOT, event), configs_dir=Path(configs)
    )


def simulate(job: tuple[str, str, str, int]) -> dict[str, Any]:
    event, variant, configs, seed = job
    m = sim_metrics(_setup(event, configs), seed)
    stops = list(m.stops.values())
    return {
        "event": event,
        "variant": variant,
        "stops": statistics.fmean(stops) if stops else float("nan"),
        "passes": m.overtakes,
        "retirements": m.retirements,
        "safety_cars": m.safety_cars,
        "vscs": m.vscs,
        "order": m.order,
    }


def borrowed_configs(season: int, tmp: Path) -> Path:
    """A copy of configs/ without the season's own calibration files (H2a undone)."""
    out = tmp / f"configs_{season}"
    shutil.copytree(ROOT / "configs", out)
    for f in (out / "calibration").glob(f"*_{season}_v1.json"):
        f.unlink()
    return out


def real_row(event: str) -> dict[str, Any]:
    m = real_metrics(ROOT / "data" / "processed" / event)
    stops = [m.stops.get(d, 0) for d in m.finishers]
    return {
        "stops": statistics.fmean(stops) if stops else float("nan"),
        "passes": m.overtakes,
        "retirements": m.retirements,
        "safety_cars": m.safety_cars,
        "vscs": m.vscs,
        "order": m.order,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seasons", type=int, nargs="+", required=True)
    parser.add_argument("--seeds", type=int, default=6)
    parser.add_argument("--workers", type=int, default=14)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    t0 = time.perf_counter()
    out: dict[str, Any] = {"seeds": args.seeds, "seasons": {}}
    with tempfile.TemporaryDirectory() as tmp:
        for season in args.seasons:
            _, splits_file = dataset_files(ROOT, season)
            splits = yaml.safe_load(splits_file.read_text(encoding="utf-8"))["splits"]
            test = list(splits["dry_test"])
            variants = {
                "season": str(ROOT / "configs"),
                "borrowed": str(borrowed_configs(season, Path(tmp))),
            }
            jobs = [
                (e, v, c, event_seed(e, k))
                for e in test
                for v, c in variants.items()
                for k in range(1, args.seeds + 1)
            ]
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                rows = list(pool.map(simulate, jobs, chunksize=2))
            real = {e: real_row(e) for e in test}
            per_race: dict[str, dict[str, Any]] = {}
            summary: dict[str, dict[str, float]] = {}
            for v in variants:
                errs: dict[str, list[float]] = {k: [] for k in METRICS}
                rhos = []
                for e in test:
                    mine = [r for r in rows if r["event"] == e and r["variant"] == v]
                    sim = {k: statistics.fmean(r[k] for r in mine) for k in METRICS}
                    rho = statistics.fmean(spearman(r["order"], real[e]["order"]) for r in mine)
                    rhos.append(rho)
                    for k in METRICS:
                        errs[k].append(sim[k] - real[e][k])
                    per_race.setdefault(e, {"real": {k: real[e][k] for k in METRICS}})[v] = {
                        **{k: round(x, 2) for k, x in sim.items()},
                        "order_rho": round(rho, 3),
                    }
                summary[v] = {
                    **{f"{k}_bias": round(statistics.fmean(errs[k]), 3) for k in METRICS},
                    **{
                        f"{k}_mae": round(statistics.fmean(abs(x) for x in errs[k]), 3)
                        for k in METRICS
                    },
                    "order_rho": round(statistics.fmean(rhos), 3),
                }
            out["seasons"][season] = {"dry_test": test, "summary": summary, "races": per_race}
            s, b = summary["season"], summary["borrowed"]
            print(
                f"{season} ({len(test)} races, {(time.perf_counter() - t0) / 60:.1f} min): "
                f"stops MAE {s['stops_mae']} (borrowed {b['stops_mae']}), passes MAE "
                f"{s['passes_mae']} ({b['passes_mae']}), retirements bias "
                f"{s['retirements_bias']} ({b['retirements_bias']}), order rho "
                f"{s['order_rho']} ({b['order_rho']})"
            )
    if args.json:
        args.json.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
