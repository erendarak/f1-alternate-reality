# Progress

## Current milestone: M0 — Feasibility & Bootstrap

Plan §22 checklist (adapted per D-002):

- [x] **1.** Repository/bootstrap: uv + Python 3.12, pyproject, ruff/mypy/pytest; first real tests pass.
- [x] **2.** `DriverProfile` schema + source-tagged seed for the full Monza 2024 grid.
- [x] **3.** Headless deterministic two-car world, 1 tick (and full races, up to 20 cars).
- [x] **4.** Pause/resume with freeze test.
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

### 2026-10-11 — Step 3: headless deterministic world

- `f1ar.domain.race_state`: frozen `WorldState`, `RaceSetup` (track, lap model, embedded profiles),
  `CarState`, `RaceEvent`, status enums (D-008).
- `f1ar.simulation.rng_streams`: keyed SHA-256 random draws, `RngSpec` (D-009).
- `f1ar.simulation.lap_model`: profile-conditioned lap-time model (D-010).
- `f1ar.simulation.lightweight_engine`: `create_world`, `start`, pure `step`, `standings`. Line
  crossings are interpolated inside the tick; events LAP_COMPLETED / CAR_FINISHED / RACE_COMPLETED.
- `f1ar.simulation.scenarios`: Monza 2024 track spec, default lap model, `build_setup`.
- `f1ar.policy.observation`: observation = state features + 14-value profile block; hidden
  `lap_target_s` excluded. **The M0 gate "profile fields in observation schema" now has a test.**
- `scripts/demo_headless.py`: lap-by-lap timing sheet, 2 to 20 cars (early visibility, D-002).
- Tests: 57 passing (~2 s). New ones cover keyed RNG, lap-model direction tests (pace, tyre mgmt, push
  vs conserve, fuel, consistency → spread, risk doesn't affect dynamics), engine single tick, events,
  full-race invariants every tick, JSON round-trip, same-seed hash, fresh-process hash, common random
  numbers, profile change affects only that driver, observation contents and no-leak.
- Fixed: `build_roster.py` now writes LF line endings.
- Next: step 4, pause/resume with a freeze test (world runner + command queue).

### 2026-10-11 — Step 4: pause / resume

- Engine: `pause`, `resume`, and `step(allow_paused=True)` for manual stepping.
- `f1ar.simulation.world_runner`: `WorldRunner`, `Command`, `CommandAck`, command log, idempotency,
  real-time pacing with speed multipliers, background thread (D-011).
- `f1ar.simulation.timing.LapTimingSheet`: shared timing-sheet formatter. `scenarios.load_event` and
  `setup_from_codes` are shared loading helpers.
- `scripts/live_race.py`: real-time race in the terminal. Enter = pause/resume, `s N` = step,
  `1/2/5/10/25` = speed (early visibility, D-002).
- Tests: 73 passing (~4 s). Plan §4.7 tests 3 (pause freezes everything), 4 (paused wall time not
  caught up), 9 (idempotent duplicate), 10 (pause exactly on a lap event) are covered with a fake
  clock. Also: pause mid-burst lands on the next tick boundary, catch-up cap, invalid commands
  rejected and logged, stepping while paused, and pacing/pauses never change the race outcome. A
  real-thread test checks pause ack < 250 ms, freeze over 1 s of real time, and no catch-up after
  resume (5/5 stable runs).
- Next: step 5, snapshot save → load in a fresh process → equality test (SQLite + checksum).
