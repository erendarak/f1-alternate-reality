# F1 Alternate Reality — Living Championship Simulator

A persistent, forkable F1 world for training AI. It starts from real historical races (the 2024
season). The goal is a **live AI world**: AI drivers, strategists and teams with their own
character decide what happens next, compete, make mistakes and, over seasons, write an
alternative history. You can pause, save, reload and branch the timeline at any moment, and take
any decision yourself.

> **Status (2026-10-25): an AI-training project from here on** ([docs/PLAN_AI_FIRST.md](docs/PLAN_AI_FIRST.md)).
> The world is built:
> - a deterministic race engine (pit stops, traffic, incidents, safety car, weather, refuelling);
> - every race 2000-2025 on its own season's fits and sporting rules; 2018-2025 playable in a 2D
>   browser app with save/load/fork;
> - a **hidden race-day truth** per race (tyre life and wear): teams only know a prior from history.
>
> Knowing that truth is worth about **+1 point per car per race** in 2018-2025 (EXP-027): the prize
> a trained AI competes for. **No AI beats the rule teams yet**: earlier attempts (M5) failed
> because the rules knew the truth. Next: A1 (the environment under uncertainty), then a learner,
> a training-budget curve and a 10-team AI league season. Progress: [docs/PROGRESS.md](docs/PROGRESS.md);
> fitted models: [docs/MODEL_REGISTRY.md](docs/MODEL_REGISTRY.md); full product plan:
> [F1_ALTERNATE_REALITY_MASTER_PLAN.md](F1_ALTERNATE_REALITY_MASTER_PLAN.md).

Alternate outcomes are **simulation estimates** built on modelling assumptions. They are not historical facts.
Driver persona values are **editorial assumptions**, not real psychometrics.

## Requirements

- Windows, macOS or Linux
- [uv](https://docs.astral.sh/uv/) (installs and manages Python 3.12 for you)
- Node.js 20.19+ (for the browser app; developed on Node 24)

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
uv run python scripts/fetch_season.py --season 2024 --telemetry   # all 24 races of 2024 (once; cached)
uv run python scripts/build_lap_dataset.py --season 2024          # lap dataset for the calibrations
npm --prefix frontend install
```

## Play (M3 app)

Two terminals:

```powershell
uv run python scripts/serve.py          # backend API on http://127.0.0.1:8000 (save file: saves/worlds.sqlite)
npm --prefix frontend run dev           # app on http://localhost:5173
```

World Library → pick any 2024 race, a seed and the weather (real conditions or random
showers), or continue a saved world. In the race: Space play/pause, S save, → step (paused),
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
| M5 training environment speed (decisions per second) | `uv run python scripts/env_benchmark.py --races 5` |
| Experiment: baselines the AI must beat (EXP-008, ~2.5 h) | `uv run python scripts/experiments/exp008_baselines.py --seeds 5 --workers 14` |
| Check every model manifest against its file and dataset | `uv run python scripts/model_registry.py` (writes `docs/MODEL_REGISTRY.md`) |
| Experiments: calibrated engine, weather, overtaking/race control, ablation | `exp002_m2_engine.py --events …` · `exp005_weather.py` · `exp006_traffic_race_control.py` · `exp007_ablation.py` (in `scripts/experiments/`) |
| Run a headless race (timing sheet) | `uv run python scripts/demo_headless.py --drivers VER HAM LEC` (or `--all`) |
| Watch a live race, pause/resume with Enter | `uv run python scripts/live_race.py --speed 25` |
| Experiment: do real-data inputs help? (EXP-001) | `uv run python scripts/experiments/exp001_data_inputs.py --seeds 30` |
| Pause at lap 18 → save → reload in a new process → finish → verify | `uv run python scripts/demo_headless.py --seed 42 --pause --save --reload --resume` |
| List/verify saved snapshots | `uv run python scripts/inspect_save.py` |
| Fork a saved race into alternate timelines and compare | `uv run python scripts/fork_demo.py --driver HAM --at-lap 18` |
| Fetch + validate ANY race (roster, data, manifest, report) | `uv run python scripts/fetch_event.py --season 2024 --round 14` |
| Fetch a whole season (timing only, skips fetched races) | `uv run python scripts/fetch_season.py --season 2024` |
| Build the lap dataset + season reference metrics (M4) | `uv run python scripts/build_lap_dataset.py --season 2024` |
| Experiment: pace models + tyre/fuel coefficients (EXP-003) | `uv run python scripts/experiments/exp003_pace_models.py --json docs/experiments/exp003.json` |
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
configs/            race rosters (verified facts), driver persona seeds (editorial assumptions),
                    calibration/ (fitted engine parameters) and models/ (model manifests)
scripts/            CLI entry points
tests/              pytest suites
docs/               decisions, progress, assumptions, licenses, limitations, experiments,
                    model registry, M5 design
```

Folders are created when a milestone needs them, not ahead of time.

## Data and licenses

Historical data comes from FastF1 / OpenF1 / Jolpica and is cached locally. It is **not** committed.
This is a non-commercial research and learning project. See [docs/LEGAL_AND_LICENSES.md](docs/LEGAL_AND_LICENSES.md).
