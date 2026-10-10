# Decisions

Newest last. Each entry: context → decision → consequences. Superseded decisions stay and are marked.

## D-001 — Python tooling (2026-10-10)

- **Decision:** Python **3.12** (pinned in `.python-version`, `requires-python = ">=3.12,<3.13"`), managed with
  **uv 0.13**; hatchling build backend; package `f1ar` at `backend/src/f1ar` (the plan's layout), installed
  editable. Dev tools are pytest, ruff and mypy (strict).
- **Why:** 3.12 is mature and widely supported by the ML stack (torch, SB3, FastF1). uv gives a lockfile and
  fast, reproducible installs on Windows without Docker.
- **Consequence:** If TUM race-simulation needs Python 3.8, it gets an isolated sandbox env and never enters
  this env (see D-003).

## D-002 — Early visibility before M3 (2026-10-10)

- **Context:** The plan delivers the first real UI at M3 (~6–10 weeks in). The owner wants to see some
  progress and "a bit of the world" before then.
- **Decision:** Add two lightweight, **read-only** views. Neither changes the architecture:
  1. **Terminal timing tower** in `scripts/demo_headless.py`: live tick/lap/position/tyre table while the
     headless world runs.
  2. **M0.5 Debug Viewer**: a single build-free HTML page served by the backend. It draws the Monza
     outline (FastF1 position data or the TUM racetrack-database centerline) and car dots from the canonical
     state, by polling. Pause/step buttons only *send commands*; the page never simulates.
- **Guardrails:** The backend stays authoritative and the viewer computes no race state (plan §1.3, §21.6).
  The viewer is clearly labelled "debug". It is replaced by the React/Vite UI at M3, not grown into it.

## D-003 — Simulator fallback pre-approved (2026-10-10)

- **Decision:** The TUM race-simulation feasibility check (M0) is time-boxed. If it can't be wrapped in a
  bounded `step()` + `get_state/set_state` adapter on modern Python, we go straight to **option C**: our own
  lightweight event/segment race engine calibrated on FastF1 data (no physics). The owner approved this
  in advance.

## D-004 — Repository folder name (2026-10-10)

- **Decision:** The repo lives at `GitHub/f1-alternate-reality` (was
  `F1 ALTERNATE REALITY — LIVING CHAMPIONSHIP SIMULATOR`). Spaces and an em-dash in paths break some
  Python/Node tooling. The old folder held only the master plan, which was copied with an identical SHA-256.

## D-005 — Canonical state encoding (2026-10-10)

- **Decision:** World state is hashed via canonical JSON (sorted keys, fixed separators, UTF-8, NaN/Inf
  rejected, JSON-native types only) → `sha256:<hex>`. See `f1ar/persistence/snapshot_codec.py`. Line
  endings are normalised to LF via `.gitattributes`, so tracked fixtures hash the same everywhere.
