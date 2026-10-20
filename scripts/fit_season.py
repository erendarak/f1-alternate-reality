"""Fit one season's calibration chain (H2a, D-054), in order, with progress:

1. lap dataset + splits (``build_lap_dataset.py``)
2. driver race pace and lap noise (W2, data only; no pass-threshold shift: W3 supersedes it)
3. tyre life (W1; kappa/delta by simulation of the season's dry_train races)
4. race events (W4; risk scale, red flags, penalties by simulation)
5. racecraft (W3; overtaking, attack/defend, launch; spread and shift by simulation)
6. persona estimates (W5)

2024 was fitted in milestone order (W1 first, before W2-W4 existed); here every simulated fit
runs with the season's earlier fits in place. Needs the season's races fetched with telemetry
(the track spec). One season at a time (14 worker processes per simulated step).

Usage:
    uv run python scripts/fit_season.py --season 2019            # ~45-75 min
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--workers", type=int, default=14)
    parser.add_argument("--from-step", type=int, default=1, help="resume at this step")
    args = parser.parse_args()
    s, w = str(args.season), str(args.workers)
    steps = [
        ("lap dataset", ["build_lap_dataset.py", "--season", s]),
        ("driver pace (W2)", ["fit_driver_pace.py", "--season", s, "--no-shift"]),
        ("tyre life (W1)", ["fit_tyre_life.py", "--season", s, "--workers", w]),
        ("race events (W4)", ["fit_race_events.py", "--season", s, "--workers", w]),
        ("racecraft (W3)", ["fit_racecraft.py", "--season", s, "--workers", w]),
        ("personas (W5)", ["fit_personas.py", "--season", s]),
    ]
    t0 = time.perf_counter()
    for i, (name, cmd) in enumerate(steps, start=1):
        if i < args.from_step:
            continue
        print(
            f"=== {args.season} step {i}/{len(steps)}: {name} "
            f"({(time.perf_counter() - t0) / 60:.0f} min so far)",
            flush=True,
        )
        done = subprocess.run([sys.executable, str(ROOT / "scripts" / cmd[0]), *cmd[1:]])
        if done.returncode != 0:
            print(
                f"=== {args.season} step {i} FAILED (exit {done.returncode}); "
                f"resume with --from-step {i}",
                flush=True,
            )
            sys.exit(done.returncode)
    print(f"=== {args.season} done in {(time.perf_counter() - t0) / 60:.0f} min", flush=True)


if __name__ == "__main__":
    main()
