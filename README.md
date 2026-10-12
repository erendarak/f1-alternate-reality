# F1 Alternate Reality — Living Championship Simulator

A persistent, forkable F1 world. It starts from real historical races (first target: **2024 Italian GP, Monza**).
From there, profile-conditioned AI strategists decide what happens next. You can pause, save, reload and
branch the timeline at any moment.

> **Status:** M0–M3 done: a deterministic race engine (pit stops, traffic, incidents, safety car)
> on real 2024 Monza/Spa grids, persistent worlds with save/load/fork, and a 2D browser app.
> Coefficients are not calibrated yet (M4). See [docs/PROGRESS.md](docs/PROGRESS.md).
> The full product and technical plan is [F1_ALTERNATE_REALITY_MASTER_PLAN.md](F1_ALTERNATE_REALITY_MASTER_PLAN.md).

Alternate outcomes are **simulation estimates** built on modelling assumptions. They are not historical facts.
Driver persona values are **editorial assumptions**, not real psychometrics.

## Requirements

- Windows, macOS or Linux
- [uv](https://docs.astral.sh/uv/) (installs and manages Python 3.12 for you)
- Node.js 20+ (for the browser app)

Install uv on Windows (PowerShell):

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

## Setup

```powershell
uv sync
```

This creates `.venv/` with Python 3.12, the `f1ar` package (editable) and the dev tools.

```powershell
uv run python scripts/fetch_event.py --season 2024 --round 16   # Monza 2024 data (once; cached)
npm --prefix frontend install
```

## Play (M3 app)

Two terminals:

```powershell
uv run python scripts/serve.py          # backend API on http://127.0.0.1:8000 (save file: saves/worlds.sqlite)
npm --prefix frontend run dev           # app on http://localhost:5173
```

World Library → create a race or continue one. In the race: Space play/pause, S save, → step (paused),
click a car for its driver panel (tyres, decisions, data sources), **Fork…** to branch the race with
another decision, **Timelines** to load any save or go back to the main story. API reference:
[docs/API_CONTRACTS.md](docs/API_CONTRACTS.md).

## Everyday commands

| Task | Command |
|---|---|
| Run tests | `uv run pytest` |
| Frontend unit tests / type-check / build | `npm --prefix frontend test` · `npm --prefix frontend run typecheck` · `npm --prefix frontend run build` |
| End-to-end test (real backend, browser, backend restart) | `npm --prefix frontend run e2e` (once: `npx --prefix frontend playwright install chromium`) |
| Lint | `uv run ruff check .` |
| Format | `uv run ruff format .` |
| Type-check | `uv run mypy` |
| Show Monza 2024 grid + driver personas | `uv run python scripts/show_profiles.py` |
| Run a headless race (timing sheet) | `uv run python scripts/demo_headless.py --drivers VER HAM LEC` (or `--all`) |
| Watch a live race, pause/resume with Enter | `uv run python scripts/live_race.py --speed 25` |
| Experiment: do real-data inputs help? (EXP-001) | `uv run python scripts/experiments/exp001_data_inputs.py --seeds 30` |
| Pause at lap 18 → save → reload in a new process → finish → verify | `uv run python scripts/demo_headless.py --seed 42 --pause --save --reload --resume` |
| List/verify saved snapshots | `uv run python scripts/inspect_save.py` |
| Fork a saved race into alternate timelines and compare | `uv run python scripts/fork_demo.py --driver HAM --at-lap 18` |
| Fetch + validate ANY race (roster, data, manifest, report) | `uv run python scripts/fetch_event.py --season 2024 --round 14` |
| Fetch + validate real Monza 2024 data (FastF1) | `uv run python scripts/fetch_monza_2024.py` (report: `docs/data/2024_italian_gp_quality.md`) |
| Verify local data files against the manifest | `uv run python scripts/fetch_monza_2024.py --verify` |
| Run the opt-in network tests | `uv run pytest -m network` |
| Replay the REAL 2024 race lap by lap (history, not simulation) | `uv run python scripts/replay_monza_2024.py` |
| **Watch the race on the real Monza map** (debug viewer) | `uv run python scripts/debug_viewer.py` then open http://127.0.0.1:8765 |
| **Watch the REAL 2024 race replayed on the map** | `uv run python scripts/debug_viewer.py --replay` (add `--event 2024_belgian_gp` for Spa) |
| Viewer controls | Space play/pause · → step · mouse wheel zoom · drag pan · click car = select · **F** follow · **B** or double-click = battle cam · Esc fit |
| Re-fetch a race roster (Jolpica) | `uv run python scripts/build_roster.py --season 2024 --round 16 --event-id 2024_italian_gp` |

## Repository layout

```text
backend/src/f1ar/   Python package: domain, data, calibration, history, simulation, policy,
                    persistence, presentation, api (FastAPI + WebSocket)
frontend/           React + TypeScript app (Vite); e2e/ = Playwright end-to-end test
configs/            race rosters (verified facts) and driver persona seeds (editorial assumptions)
scripts/            CLI entry points
tests/              pytest suites
docs/               decisions, progress, assumptions, licenses, limitations, experiments
```

Folders are created when a milestone needs them, not ahead of time.

## Data and licenses

Historical data comes from FastF1 / OpenF1 / Jolpica and is cached locally. It is **not** committed.
This is a non-commercial research and learning project. See [docs/LEGAL_AND_LICENSES.md](docs/LEGAL_AND_LICENSES.md).
