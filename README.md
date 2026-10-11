# F1 Alternate Reality — Living Championship Simulator

A persistent, forkable F1 world. It starts from real historical races (first target: **2024 Italian GP, Monza**).
From there, profile-conditioned AI strategists decide what happens next. You can pause, save, reload and
branch the timeline at any moment.

> **Status:** M0. A headless, deterministic race engine runs (placeholder lap model, no UI yet);
> see [docs/PROGRESS.md](docs/PROGRESS.md).
> The full product and technical plan is [F1_ALTERNATE_REALITY_MASTER_PLAN.md](F1_ALTERNATE_REALITY_MASTER_PLAN.md).

Alternate outcomes are **simulation estimates** built on modelling assumptions. They are not historical facts.
Driver persona values are **editorial assumptions**, not real psychometrics.

## Requirements

- Windows, macOS or Linux
- [uv](https://docs.astral.sh/uv/) (installs and manages Python 3.12 for you)
- Node.js 20+ (needed only from the frontend milestone onward)

Install uv on Windows (PowerShell):

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

## Setup

```powershell
uv sync
```

This creates `.venv/` with Python 3.12, the `f1ar` package (editable) and the dev tools.

## Everyday commands

| Task | Command |
|---|---|
| Run tests | `uv run pytest` |
| Lint | `uv run ruff check .` |
| Format | `uv run ruff format .` |
| Type-check | `uv run mypy` |
| Show Monza 2024 grid + driver personas | `uv run python scripts/show_profiles.py` |
| Run a headless race (timing sheet) | `uv run python scripts/demo_headless.py --drivers VER HAM LEC` (or `--all`) |
| Watch a live race, pause/resume with Enter | `uv run python scripts/live_race.py --speed 25` |
| Pause at lap 18 → save → reload in a new process → finish → verify | `uv run python scripts/demo_headless.py --seed 42 --pause --save --reload --resume` |
| List/verify saved snapshots | `uv run python scripts/inspect_save.py` |
| Re-fetch a race roster (Jolpica) | `uv run python scripts/build_roster.py --season 2024 --round 16 --event-id 2024_italian_gp` |

## Repository layout

```text
backend/src/f1ar/   Python package (domain, data, persistence; simulation/policy/api to come)
configs/            race rosters (verified facts) and driver persona seeds (editorial assumptions)
scripts/            CLI entry points
tests/              pytest suites
docs/               decisions, progress, assumptions, licenses, limitations, experiments
```

Folders are created when a milestone needs them, not ahead of time.

## Data and licenses

Historical data comes from FastF1 / OpenF1 / Jolpica and is cached locally. It is **not** committed.
This is a non-commercial research and learning project. See [docs/LEGAL_AND_LICENSES.md](docs/LEGAL_AND_LICENSES.md).
