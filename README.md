# F1 Alternate Reality — Living Championship Simulator

A persistent, forkable F1 world. It starts from real historical races (first target: **2024 Italian GP, Monza**).
From there, profile-conditioned AI strategists decide what happens next. You can pause, save, reload and
branch the timeline at any moment.

> **Status:** M0 — bootstrap. There is nothing to play yet; see [docs/PROGRESS.md](docs/PROGRESS.md).
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

## Repository layout

```text
backend/src/f1ar/   Python package (simulation, persistence, data, policy, api)
tests/              pytest suites (unit, replay, property, integration, ml)
docs/               decisions, progress, limitations, experiments
```

Folders are created when a milestone needs them, not ahead of time.

## Data and licenses

Historical data comes from FastF1 / OpenF1 / Jolpica and is cached locally. It is **not** committed.
This is a non-commercial research and learning project. See `docs/LEGAL_AND_LICENSES.md` (added in M0).
