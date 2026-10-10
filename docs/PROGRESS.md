# Progress

## Current milestone: M0 — Feasibility & Bootstrap

Plan §22 checklist (adapted per D-002):

- [x] **1.** Repository/bootstrap: uv + Python 3.12, pyproject, ruff/mypy/pytest; first real tests pass.
- [ ] **2.** `DriverProfile` schema + source-tagged seed for the full Monza 2024 grid.
- [ ] **3.** Headless deterministic two-car world, 1 tick.
- [ ] **4.** Pause/resume with freeze test.
- [ ] **5.** Snapshot save → fresh-process load → equality test.
- [ ] **6.** Timeline fork with isolation test.
- [ ] **6b.** Terminal timing tower in `demo_headless.py` (D-002).
- [ ] **7.** FastF1 Monza 2024 fetch/cache + data quality report.
- [ ] **8.** TUM race-simulation feasibility memo → go/no-go (fallback C pre-approved, D-003).
- [ ] **M0.5** Debug viewer: Monza outline + car dots from backend state (D-002).

## Session log

### 2026-10-10 — Step 1: bootstrap

- Moved the repo to `f1-alternate-reality` (D-004). Installed uv 0.13.0.
- Added `pyproject.toml`, `.python-version`, `.gitignore`, `.gitattributes`, `README.md`, `docs/`.
- First real module: `f1ar/persistence/snapshot_codec.py` (canonical JSON + state hash), with unit tests
  that include a cross-process hash-stability check.
- Next: step 2, driver profile schema and the Monza 2024 seed.
