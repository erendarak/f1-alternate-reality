"""EXP-007: what does each calibrated part of the engine add? (M4 slice 6, D-036)

Ablation by removal: a race setup is built with one fitted artifact missing (the setup then falls
back to the editorial model for that part, as it would on a machine without the file), and with
all of them missing (the deterministic editorial baseline). Races: the 5 dry test races (Monza
among them: the plan's historical reference) and the 3 wet races; the weather ablation only
matters on the wet races.

Per race and setting, over N seeds, against the real race: winner race time error, Spearman of the
finishing order, share of the real winner's wins, stops per finisher error, share of simulated
compound sequences that a real finisher also used, overtakes error, retirements error.

Usage:
    uv run python scripts/experiments/exp007_ablation.py --seeds 10 \
        --json docs/experiments/exp007.json
"""

import argparse
import json
import shutil
import statistics
import tempfile
import time
from pathlib import Path
from typing import Any

from f1ar.calibration.lap_model_fit import SHAPE_FILE
from f1ar.calibration.race_control_fit import RACE_CONTROL_FILE
from f1ar.calibration.reference_metrics import RaceMetrics, real_metrics
from f1ar.calibration.simulated_metrics import sim_metrics, spearman
from f1ar.calibration.traffic_fit import TRAFFIC_FILE
from f1ar.calibration.weather_fit import WEATHER_FILE
from f1ar.simulation.scenarios import event_data_dir, load_event, setup_from_codes

ROOT = Path(__file__).resolve().parents[2]
DRY_TEST = (
    "2024_italian_gp",
    "2024_japanese_gp",
    "2024_monaco_gp",
    "2024_mexico_city_gp",
    "2024_abu_dhabi_gp",
)
WET = ("2024_canadian_gp", "2024_british_gp", "2024_sao_paulo_gp")
ALL = (SHAPE_FILE, WEATHER_FILE, TRAFFIC_FILE, RACE_CONTROL_FILE)
SETTINGS: dict[str, tuple[Path, ...]] = {
    "full": (),
    "no_tyre_shape": (SHAPE_FILE,),
    "no_weather": (WEATHER_FILE,),
    "no_traffic": (TRAFFIC_FILE,),
    "no_race_control": (RACE_CONTROL_FILE,),
    "all_editorial": ALL,
}


def configs_without(base: Path, missing: tuple[Path, ...]) -> Path:
    """A copy of ``configs/`` without the given calibration files."""
    target = Path(tempfile.mkdtemp(prefix="exp007_")) / "configs"
    shutil.copytree(base, target)
    for f in missing:
        (target / f).unlink()
    return target


def score(sims: list[RaceMetrics], real: RaceMetrics) -> dict[str, float]:
    real_finish = [d for d in real.order if d in real.finishers]
    real_sequences = set(real.sequences.values())
    real_stops = statistics.fmean(real.stops.values())

    def mean(values: list[float]) -> float:
        return round(statistics.fmean(values), 3)

    return {
        "winner_time_error_min": mean(
            [abs(s.winner_time_s - real.winner_time_s) / 60 for s in sims]
        ),
        "spearman": mean(
            [spearman([d for d in s.order if d in s.finishers], real_finish) for s in sims]
        ),
        "real_winner_wins": mean([float(s.winner == real.winner) for s in sims]),
        "stops_error": mean([abs(statistics.fmean(s.stops.values()) - real_stops) for s in sims]),
        "sequence_overlap": mean(
            [
                sum(seq in real_sequences for seq in s.sequences.values()) / len(s.sequences)
                for s in sims
            ]
        ),
        "passes_error": mean([abs(s.overtakes - real.overtakes) for s in sims]),
        "retirements_error": mean([abs(s.retirements - real.retirements) for s in sims]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    t0 = time.perf_counter()
    configs = {name: configs_without(ROOT / "configs", files) for name, files in SETTINGS.items()}
    results: dict[str, Any] = {}
    for event in DRY_TEST + WET:
        data_dir = event_data_dir(ROOT, event)
        if data_dir is None:
            raise SystemExit(f"fetch {event} first")
        real = real_metrics(data_dir)
        row: dict[str, Any] = {"group": "wet" if event in WET else "dry_test"}
        for name, cfg in configs.items():
            if name == "no_weather" and event not in WET:
                continue  # dry races: no weather model to remove
            setup = setup_from_codes(
                *load_event(cfg, event), None, data_dir=data_dir, configs_dir=cfg
            )
            sims = [sim_metrics(setup, seed) for seed in range(1, args.seeds + 1)]
            row[name] = score(sims, real)
            row[name]["labels"] = {
                k: setup.inputs[k].split(":")[0]
                for k in ("tyre_model", "weather", "traffic", "race_control")
            }
        results[event] = row
        print(
            f"{event:26} "
            + " | ".join(
                f"{n} sp {row[n]['spearman']:.2f} pass {row[n]['passes_error']:.0f} "
                f"seq {row[n]['sequence_overlap']:.2f}"
                for n in SETTINGS
                if n in row
            ),
            flush=True,
        )

    summary: dict[str, Any] = {}
    for group, events in (("dry_test", DRY_TEST), ("wet", WET)):
        summary[group] = {}
        for name in SETTINGS:
            rows = [results[e][name] for e in events if name in results[e]]
            if not rows:
                continue
            summary[group][name] = {
                k: round(statistics.fmean(r[k] for r in rows), 3) for k in rows[0] if k != "labels"
            }
    print(json.dumps(summary, indent=1))
    for cfg in configs.values():
        shutil.rmtree(cfg.parent, ignore_errors=True)
    if args.json:
        out = {
            "seeds": args.seeds,
            "settings": {k: [p.as_posix() for p in v] for k, v in SETTINGS.items()},
            "races": results,
            "summary": summary,
            "seconds": round(time.perf_counter() - t0, 1),
        }
        args.json.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
