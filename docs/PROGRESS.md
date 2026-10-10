# Progress

## Current milestone: M0 — Feasibility & Bootstrap

Plan §22 checklist (adapted per D-002):

- [x] **1.** Repository/bootstrap: uv + Python 3.12, pyproject, ruff/mypy/pytest; first real tests pass.
- [x] **2.** `DriverProfile` schema + source-tagged seed for the full Monza 2024 grid.
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

### 2026-10-10 — Step 2: driver profiles + Monza 2024 roster

- `f1ar.domain.provenance.SourceType`: the five provenance categories (plan §3.4).
- `f1ar.domain.identities`: `RaceRoster` / `RosterEntry` (race-specific, `historical_verified`).
- `f1ar.data.jolpica_adapter` + `scripts/build_roster.py`: real Monza 2024 entry list and grid from
  Jolpica-F1 → `configs/rosters/2024_italian_gp.json` (20 entries, grid P1 Norris … P20 Zhou).
- `f1ar.domain.driver_profiles`: `DriverArchetype` (14 fields in [0, 1]), `DriverProfile`,
  `ProfileSet` with `manifest_hash`, and template + override resolution (D-007).
- Configs: `configs/drivers/archetypes.yaml` (3 templates), `seed_2024_monza_drivers.yaml` (20 drivers;
  8 from the plan's personas, 12 template-only). See `docs/ASSUMPTIONS.md`.
- `scripts/show_profiles.py` prints the grid with Stage 1 persona values (early visibility, D-002).
- Tests: 29 passing (roster transform/validation, seed coverage, provenance rules, range checks,
  template inheritance, feature vector order, value-not-name features, hash pinning, immutability).
- Not done yet: profiles aren't wired into an observation or the simulation. That happens in step 3,
  where the M0 gate "profile fields included in observation schema" gets its test.
- Next: step 3, headless deterministic two-car world, 1 tick.
