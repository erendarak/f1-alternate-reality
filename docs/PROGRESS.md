# Progress

## World realism before more AI (research-driven, `docs/RESEARCH_REAL_F1.md`)

Why: EXP-009 showed the world gave strategy almost nothing to win. The owner asked for research
into what makes the difference in real F1 and for the world to be adapted to it; the AI work
(M5) resumes once the world is closer to 2024. Plan (2026-10-15, one slice per commit):

- [x] **W1: tyres v2 (engine v0.10.0, D-041, EXP-010)**: tyre life, management vs push, cliff,
  fresh-tyre peak, a planner that manages and needs 6.5 s per stop; stops per finisher now
  follow the real races (test error 0.56 -> 0.28). Tyre allocation deferred.
- [x] **W4: race control and random events (engine v0.11.0, D-042, EXP-011)**: debris SC/VSC,
  red flags with free tyre changes, pit crews and reliability per team, time penalties,
  retirements refit for W1's pace mix; event rates at 2024's level, order vs real 0.62 -> 0.68.
- [x] **W2: race pace (engine v0.12.0, D-043, EXP-012)**: driver pace vs teammate, race form,
  lap noise per driver from 2024 (noise 0.14 -> ~0.45 s); passes kept at 2024's level.
  In-season development deferred to seasons (M9).
- [x] **W3: racecraft (engine v0.13.0, D-044, EXP-013)**: overtaking refit with W2 noise,
  attack/defend and launch skills per driver, lap-1 risk by grid slot; DRS trains not modelled.
- [x] **W5: personas from data where measurable (world schema 15, D-045, EXP-014)**:
  consistency, overtake/defense skill rank-mapped from W2/W3; tyre management and mistakes
  not measurable reliably (stay editorial).
- [x] **W6: wet races (engine v0.14.0, D-046, EXP-015)**: fresh wet sets under SC/VSC and
  the wet incident factor fitted on the 3 wet races; track temperature has no measurable
  effect (not added).

## Current milestone: M5 — Stage 1: a field of AI strategists (plan §9, §16; design `docs/M5_DESIGN.md`, D-037)

Why: the project exists to train AI for a live F1 world. M5 trains the first real AI: one shared
policy, conditioned on driver profile, car, track and the team principal's objective, that
decides pit/tyre and pace for every car. Judged by: did it really learn (vs rules and fork-based
MPC, on held-out tracks and seeds), and is the race more alive (strategy variety vs 2024).
Plan (agreed 2026-10-13, one slice per commit; nothing started):

- [x] **Slice 1: training environment (D-038, BENCH-002)**: Gymnasium env + multi-car session,
  decision points, masks, regulation set v1, objectives and rewards, team principal rules,
  scenario sampler with championship context, AGENT controller, speed benchmark, tests.
- [x] **Slice 2: baselines (D-039, EXP-008)**: B0 real stops, B1 rules, B2 enumerator, B3
  fork-based MPC, random; paired evaluation. Result: the rule planner is the bar (MPC is worse).
- [x] **Slice 3: training stack + first learned policy (D-040, EXP-009)**: PyTorch/CUDA +
  MaskablePPO, CPU workers, policy registry + pinning, rules warm start + paired reward.
  **Result negative**: the policy doesn't beat the rules on held-out tracks (-0.029; the
  observation lacks per-compound track data). Proposed fix before slice 4: `rl_obs_v2` + retrain.
- [x] **Slice 3b: the strategist again on the W1-W6 world (D-047, EXP-016, EXP-017)**:
  `rl_obs_v2`, rule-pace teacher, tyre randomisation. The clone now generalises (ties the
  rules held out); **PPO gains nothing measurable** (~1 % of decisions changed).
- [x] **Slice 4: goals, character, racecraft intent (engine v0.15.0, D-048, EXP-018)**: the
  lever and the conditioning are built; **the AI doesn't learn them** (ties the rules, intent
  not situational, no goal/character effect): credit assignment is the blocker.
- [ ] Slice 5: all 20 cars AI (shared policy, frozen-opponent pool); realism vs 2024; exploit
  audit; FIA ban mechanism; EXP-011.
- [ ] Slice 6: weather generalisation, UI (thinking panel, behaviour cards, fork-and-try model,
  pinned versions, override hand-back), M5 done-check.

## Previous milestone: M4 — Race models and calibration (plan §16, §7.6, §8) — COMPLETE

Why: EXP-002 showed every simulated car one-stopping onto SOFT (real: mostly M-H, many 2-stops),
overtakes 2-5x too rare, no weather, editorial incident rates. M4 replaces guessed coefficients
with values estimated from the 2024 season and measures the result, before M5 trains a
strategist inside the engine. Plan (agreed 2026-10-13, one slice per commit):

- [x] **Slice 1: 2024 season dataset**: all 24 races fetched; lap dataset with feature
  availability + leakage audit; dataset manifest; splits by rule; season reference metrics (D-031).
- [x] **Slice 2: pace models + tyre curves (EXP-003, D-032)**: rolling median, structural regression
  (compound offsets, non-linear wear, fuel, car/driver), LightGBM; leave-one-race-out errors,
  per track/driver, feature importance, profile ablation (read as "driver identity", not as a
  validation of the editorial personas), model manifests.
- [x] **Slice 3: engine integration (v0.7.0, D-033, EXP-004)**: calibrated tyre/fuel model, planner for non-linear
  wear, any fetched 2024 race simulatable; EXP-002 re-run. *(Correction, slice 6: the planned
  "optional learned-pace port" was not built; D-032 keeps the LightGBM predictor out of the
  engine. The engine's predictor is the fitted lap model, with fallbacks since slice 6.)*
- [x] **Slice 4: weather (v0.8.0, D-034, EXP-005)**: rain (historical or seeded), track wetness,
  INTERMEDIATE/WET with crossovers fitted on Canada/Britain/São Paulo 2024, `wet_skill`,
  forecast without future leakage (US-06), wet-tyre rule exemption, UI.
- [x] **Slice 5: traffic + race control (v0.9.0, D-035, EXP-006)**: overtaking fitted on real
  lap-level encounters with per-track thresholds, launch spread, race-control rates from the
  season; calibration report on all 21 dry races (train vs test).
- [x] **Slice 6: wrap-up (D-036, EXP-007)**: no MLP (no gap to close), model registry with
  hash checks, safe fallbacks for every engine artifact, ablation by removal, M4 done-check.

**M4 done-check** (plan §16):

| Plan item | Where | State |
|---|---|---|
| Pace dataset + leakage audits | slice 1, D-031 | done |
| Feature schema (profile/car/track/tyre/weather) | slice 1-2 (`feature_schema_version` 1) | done |
| Baselines: rolling median / linear / LightGBM | EXP-003 | done |
| Profile-conditioned MLP "if needed" | D-036 | not built: no gap (EXP-003 ablation) |
| MAE + confidence intervals, per track / driver | EXP-003 | done |
| Pit-stop, weather, tyre curve, traffic calibration | D-033, D-034, D-035 | done (pit loss from green stops) |
| Calibration report against Monza 2024 | EXP-004, EXP-006, EXP-007 (Monza in each) | done |
| Feature importance / ablation, model registry | EXP-003, EXP-007, `docs/MODEL_REGISTRY.md` | done |
| Deterministic baseline when a model is missing | D-036, tests | done (tyre shape crashed before) |
| **DoD**: baseline report on test races | EXP-003 (online), EXP-004/006/007 (engine) | done |
| **DoD**: profile-conditioned model ablation | EXP-003 (identity and persona: no gain) | done |
| **DoD**: all model versions tracked | registry: 5 models, hashes checked by a test | done |
| **DoD**: predictor safely integrated into the engine | fitted lap model with guards (D-033) + fallbacks | done |
| **DoD**: dataset / feature / split manifest | `data/manifests/dataset_laps_2024_v1.yaml` | done |

Carried to M5: every car still one-stops at the same time (no strategy variety), so passes
come from the wrong causes; real winners win less often with the fitted start and passing;
models rest on one season (3 wet races, one race per track).

## Previous milestone: M3 — First playable 2D race (plan §16) — COMPLETE (replay in the new UI -> M6)

M3 plan (agreed 2026-10-12, one slice per commit):

- [x] **Slice 1: real API v1**: REST endpoints (plan §13.2), live session, WebSocket state stream,
  auto-pause, driver card with source labels, `docs/API_CONTRACTS.md` (D-027).
- [x] **Slice 2: React/Vite app**: TrackCanvas, TimingTower, RaceControls, live stream, mode
  indicator, event log + lap strip, minimal start screen (D-028).
- [x] **Slice 3: World Library, save / load / fork dialogs, timelines, driver panel with source
  tags, race director, keyboard shortcuts** (D-029, store schema v4).
- [x] **Slice 4: Playwright E2E** (play -> pause -> save -> kill backend + reload -> continue ->
  resume -> fork -> back to main), **M3 DoD check** (D-030).

Weather (open M2 item): **moved to M4** (owner's decision, 2026-10-12), designed together with the
tyre recalibration and estimated from a wet 2024 race. It must be in before M5 (observation and
action spaces).

## Previous milestone: M2 — Canonical race engine + persistent world (plan §16) — COMPLETE except weather (-> M4)

M2 plan (agreed 2026-10-12, one slice per commit; engine version bump per behaviour change):

- [x] **Slice 1: pit stops + tyre choice, baseline strategy, chequered flag / lapped cars** (D-022, engine v0.4.0).
- [x] **Slice 2: traffic, dirty air, overtaking cost, blue flags** (D-023, engine v0.5.0).
- [x] **Slice 3: race control: mistakes, crashes/failures -> RETIRED, VSC, safety car + restart** (D-024, engine v0.6.0).
- [x] **Slice 4: persistence: event + command logs, autosave, crash recovery in a fresh process,
  invariants every tick, policy/dataset pinning** (D-025, store schema v3, world schema v7).
- [x] **Slice 5: Spa simulation, EXP-002, BENCH-001, `RaceEngine` port + action masks, M2 DoD check** (D-026).

Already in place from M0 (plan §16 M2 items): runner + command queue + RUNNING/PAUSED state machine,
SQLite snapshots with checksum/schema version/idempotency, forks with common random numbers,
fresh-process restore test.

## Previous milestone: M1 — Historical data layer (plan §16) — COMPLETE

M0 and M0.5 are complete (gate 5/5; see below). M1 tasks, adapted to what step 7 already delivered:

- [x] Repeatable fetch with cache, typed schema, units, validation, manifest SHA-256 (step 7).
- [x] Grid, results, pit/stint data verified, with an independent Jolpica cross-check (step 7).
- [x] Track geometry (FastF1 outline + corners instead of the TUM centreline, D-016/D-017).
- [x] Offline synthetic fixtures for all data code (step 7, M1).
- [x] **Historical reference race + reference event list; `HISTORICAL_REPLAY` separated in code (D-018).**
- [x] Real race replay on the debug viewer map (play/pause/speed/seek; no fork, no actions) + camera.
- [x] More events through the same pipeline: `fetch_event.py --season --round` (D-020); 2024 Belgian GP added.
- [x] Engine inputs from data: real start compounds + team car-pace offsets (D-021, EXP-001).

**M1 DoD (plan §16) met:** one CLI fetches and validates the real race; the next run loads from
cache; 20 cars / 53 laps checked (lapping and retirement handled); gaps reported; historical replay
and simulation are separate modes.

## Previous milestone: M0 — Feasibility & Bootstrap

Plan §22 checklist (adapted per D-002):

- [x] **1.** Repository/bootstrap: uv + Python 3.12, pyproject, ruff/mypy/pytest; first real tests pass.
- [x] **2.** `DriverProfile` schema + source-tagged seed for the full Monza 2024 grid.
- [x] **3.** Headless deterministic two-car world, 1 tick (and full races, up to 20 cars).
- [x] **4.** Pause/resume with freeze test.
- [x] **5.** Snapshot save → fresh-process load → equality test.
- [x] **6.** Timeline fork with isolation test.
- [ ] **6b.** Terminal timing tower in `demo_headless.py` (D-002).
- [x] **7.** FastF1 Monza 2024 fetch/cache + data quality report.
- [x] **8.** TUM race-simulation feasibility memo → go/no-go (fallback C pre-approved, D-003).
- [x] **M0.5** Debug viewer: Monza outline + car dots from backend state (D-002).

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

### 2026-10-11 — Step 5: save / load

- `f1ar.persistence.sqlite_store.SnapshotStore`: atomic snapshot save, verified load, list,
  `latest_valid` recovery, store schema guard (D-012).
- `WorldRunner`: `SAVE` command, acked only after commit; failures reported, never faked.
- `scripts/demo_headless.py`: `--pause [--pause-at-lap N] --save --reload --resume`, where reload runs
  in a fresh child process and the result is compared with an uninterrupted run. Also `--resume-from`.
- `scripts/inspect_save.py`: lists snapshots in a save DB and verifies each.
- Bug found by the tests: SQL-level corruption can turn the BLOB into TEXT, which crashed `load` with a
  TypeError. It is now a clean `SnapshotIntegrityError`.
- Tests: 90 passing (~7 s). Plan §4.7 test 1 (save → fresh process → equal), test 2 (1000 ticks ==
  400 + save + fresh-process load + 600), test 7 (corrupt checksum rejected, last good suggested), plus
  resume to the flag from lap 18 in another process, engine/schema version refusal, immutability,
  runner SAVE while running/paused, and a failed save never reported as success. The M0 gate command
  itself is an integration test.

#### M0 acceptance gate status

```text
[PASS] python -m pytest tests/unit tests/replay
[PASS] python scripts/demo_headless.py --seed 42 --pause --save --reload --resume   -> MATCH
[PASS] A repeat run produces identical canonical state hash in deterministic mode
[PASS] DriverProfile fields validate and are included in observation schema
[TODO] Simulator decision, license notes, and data limitations documented  (steps 7-8)
```

- Next: step 6, timeline fork (branch with a different decision; original untouched).

### 2026-10-12 — Step 6: timeline forks

- `f1ar.simulation.timeline`: `fork_world`, `SeedMode` (common / independent randomness),
  `new_timeline_id` (D-013).
- First driver action, `SetPaceMode`: `engine.apply_action` and runner `SET_PACE_MODE`, effective from
  the next lap.
- `SnapshotStore`: `fork`, `list_timelines`, `delete_snapshot`; store schema v2 with an in-place v1→v2
  migration.
- **Bug found and fixed:** tyre wear was retroactively rescaled by the current pace mode. It is now
  path-dependent state (`tyre_wear_s`, `lap_pace_mode`). Engine v0.2.0, world schema v2; old saves are
  refused clearly (D-014).
- `scripts/fork_demo.py`: save at lap N, fork into main / push / conserve with the same luck, run all
  to the flag, compare, print the timeline tree, verify the source snapshot is unchanged.
- `scripts/live_race.py`: `p HAM push` changes a driver's pace mode mid-race.
- Tests: 115 passing (~7.5 s). Plan §4.7 test 5 (fork with a different decision: A unchanged, B
  differs, VER identical thanks to common random numbers) and test 6 (deleting the source snapshot
  doesn't break the branch). Also: reproducible counterfactuals, independent-seed branches, timeline
  tree, in-memory fork lineage, v1 store migration, actions (event, next-lap effect, no-op, rejection,
  observable mode), runner SET_PACE_MODE, the late-push regression, hidden tyre wear.
- M0 steps 1–6 are done. Remaining: 6b terminal tower (largely covered by `live_race.py`), 7 FastF1
  data, 8 TUM feasibility memo, M0.5 debug viewer.
- Next: step 7, FastF1 Monza 2024 fetch/cache + data quality report.

### 2026-10-12 — Step 7: real Monza 2024 data

- FastF1 3.8.3 (+ pyarrow, pandas-stubs). API checked in the installed source (`Session.load`
  signature, track status codes, position units).
- `f1ar.data.fastf1_adapter`: `load_race_session` (verified load, `DataSourceError`), pure
  normalisers, `track_outline`. `f1ar.data.quality`: typed checks + Markdown report.
  `f1ar.data.manifests`: manifest model, write/verify. `jolpica_adapter.result_records`.
- `scripts/fetch_monza_2024.py`: 7 Parquet tables (1008 laps, 20 results, 133 weather samples, 59
  race-control messages, 612,740 position samples, 317-point track outline), manifest
  `data/manifests/2024_italian_gp_fastf1.yaml`, report `docs/data/2024_italian_gp_quality.md`.
- **Quality: 15 PASS, 2 WARN, 0 FAIL.** FastF1 and Jolpica agree on every number, code, grid slot,
  team pairing, finishing position and lap count. WARN: 6 lapped cars + Tsunoda retired on lap 7;
  13 deleted laps, 81 inaccurate laps (excluded from pace stats).
- What the real race tells the simulator (inputs to M1/M2/M4): fully green and dry (32–34 °C air,
  44–55 °C track); 14 cars started on MEDIUM and 6 on HARD; Leclerc won on a one-stop (M→H lap 15),
  most front-runners two-stopped; green-flag median lap 84.8 s (our placeholder model is in the right
  range); a pre-race "RISK OF RAIN 40 %" message exists for future weather-aware strategy.
- Terms verified from the sources' own pages (see `LEGAL_AND_LICENSES.md`): Jolpica CC BY-NC-SA 4.0,
  4 req/s / 500 req/h, custom UA required (we send one); OpenF1 non-commercial, 3 req/s / 30 req/min.
- Robustness fix: FastF1 silently returned no laps under a NumPy deprecation (found with
  `-W error`); now detected and reported clearly.
- Tests: 128 offline passing (~9.5 s) + 1 opt-in network test passing against the real data.
- Next: step 8, TUM race-simulation feasibility memo (go/no-go; fallback C pre-approved).

### 2026-10-12 — Step 8: simulator decision

- TUM race-simulation was run in an isolated sandbox (own venv, outside the repo), probed with
  `scripts/research/tum_feasibility_probe.py`. The time-discrete simulator was assessed from its
  repository.
- **Decision: our engine (option C); both TUM simulators no-go** (`docs/SIMULATOR_DECISION.md`,
  D-016). Key evidence: two global RNGs (no common random numbers across forks; NumPy-only pit draws
  break `random.seed` repeatability), pickle-only persistence of private state, future events
  pre-drawn into state, the shipped Monza 2019 file fails to construct (VSCs 1.46 laps apart) and
  runs out of fuel without them, unmaintained deps (TF/cvxpy hard imports). It's fast (54–116 ms per
  race), and in-memory deepcopy forks are exact when both RNG states are captured. Both are recorded
  in fairness.
- Modelling ideas kept for M2 (re-implemented, credited): accident probability, start performance,
  pit standstill distribution, overtaking loss, FCY handling, degradation shapes.

#### M0 acceptance gate status

```text
[PASS] python -m pytest tests/unit tests/replay
[PASS] python scripts/demo_headless.py --seed 42 --pause --save --reload --resume   -> MATCH
[PASS] A repeat run produces identical canonical state hash in deterministic mode
[PASS] DriverProfile fields validate and are included in observation schema
[PASS] Simulator decision, license notes, and data limitations documented
       (SIMULATOR_DECISION.md, LEGAL_AND_LICENSES.md, KNOWN_LIMITATIONS.md, data quality report)
```

**M0 acceptance gate: 5/5.** Remaining from the adapted plan: M0.5 debug viewer (D-002); step 6b
(terminal tower) is effectively covered by `live_race.py`.
- Next: M0.5, debug viewer (Monza outline from FastF1 + car dots from backend state).

### 2026-10-12 — M0.5: debug viewer (first visual)

- Data pipeline extended: `results.team_color` (official timing colours) and `corners.parquet`
  (11 corners + 95° display rotation from FastF1 `get_circuit_info()`).
- `f1ar.presentation.track_geometry` (arc-length placement), `f1ar.presentation.view_model`
  (pure state → JSON), `f1ar.api.debug_app` (FastAPI over one live `WorldRunner`), page
  `f1ar/api/static/debug_viewer.html`, launcher `scripts/debug_viewer.py`, `.claude/launch.json`.
- Verified in the browser: real Monza outline with corner numbers and start/finish line; cars in team
  colours move continuously. Start, pause (applied at tick 763, lap 5), select driver, PUSH
  (`PACE_MODE_CHANGED … from lap 6` in the event log), resume (no catch-up). Timing tower shows
  ~1:24.2 laps.
- Assets: track from data, cars as team-colour dots (D-017). The user proposed the palm-tree pixel car
  pack; its terms are recorded and it's planned for M3 (one sprite recoloured per team).
- Dev deps: `httpx` → `httpx2` (Starlette 1.7 test client).
- Tests: 147 passing (+9 viewer tests: placement by lap fraction, view model purity, colour fallback,
  page/track served, commands relayed incl. step/pace/save/rejections, idempotency, 422 on bad input).
- Follow-up (same day): cars drawn with the user-supplied palm-tree sprite, recoloured per team and
  rotated along the track; Restart (same seed) / New race buttons + `POST /api/debug/restart`; seed
  shown in header. Tests: 155 passing.
- **M0 + M0.5 complete.** Next: M1, the historical data layer (plan §16): repeatable fetch for more
  races, real start compounds/strategies as a historical-replay reference, and engine inputs from data.

### 2026-10-12 — M1 slice 1: historical reference race

- `f1ar.history.reference.HistoricalRace`: lap records, classification after any lap, events,
  replay progress (D-018). `scripts/replay_monza_2024.py` prints the real race lap by lap with pit
  stops and the retirement, then the official classification.
- Real-data checks (network test): 0 position mismatches vs the timing feed over the whole race;
  LEC–PIA–NOR, gap 2.7 s; TSU the only retirement.
- Tests: 161 offline passing (+6 history tests) + 2 opt-in network tests.
- Next: replay the real race on the debug viewer map.

### 2026-10-12 — M1 slice 2: real-race replay on the map + camera system

- `history/replay.py` (ReplayClock), `presentation/view_model.replay_view`, `api/replay_app.py`, a shared
  route helper in `debug_app.py`, `debug_viewer.py --replay`, launch config `replay-viewer`.
- Page rewrite with a camera: wheel zoom at cursor, drag pan, Fit, **Follow**, **Battle cam**
  (follow + zoom), double-click a car = battle cam, display-only side-by-side spreading,
  to-scale track/cars when zoomed (D-019).
- Verified in the browser: battle cam on VER in the simulation; replay seek to lap 15 shows the real
  in-laps (NOR, MAG on lap 14; LEC, HAM on lap 15) and NOR's out-lap on 1-lap-old hards; battle cam
  on LEC's in-lap behind SAI.
- Bugs found and fixed while verifying: (1) per-frame camera easing stalled when the browser throttled
  frames → time-based easing; (2) base scale computed while the pane was hidden (height 0 → negative
  scale) → ResizeObserver + size guard; (3) replay tower showed "+1 lap" for everyone just after the
  leader crossed the line → laps-down from track progress (regression test).
- Tests: 170 offline passing (+10 replay clock/view/API) + 2 network.
- Next M1 items: generalise fetching to any event; engine inputs from data (real start compounds,
  pace baselines).

### 2026-10-12 — M1 slice 3: any race + on-track vs official results

- `f1ar.data.pipeline` (fetch_event, verify_event, typical_lap, event ids), `scripts/fetch_event.py`;
  `fetch_monza_2024.py` is now a shortcut onto the same pipeline (Monza re-fetched: identical data;
  outline now from LEC's typical lap 39).
- 2024 Belgian GP fetched: 15 PASS / 4 WARN / 0 FAIL (ZHO retired lap 5, RUS disqualified after 44
  laps, one lap without a time, deleted/inaccurate laps). 0 replay mismatches vs the feed.
- **Bug found by the second race:** the history layer treated any non-Finished/Lapped status as a
  retirement, so Spa's winner on the road "retired" at the flag. Fixed with `took_flag` (D-020),
  plus a `DISQUALIFIED` event and a DSQ badge in the viewer. The synthetic fixture was also corrected
  (its "lapped" car stopped before the flag, which real lapped cars don't).
- Viewer: `--replay --event 2024_belgian_gp`; the title shows the event. Verified in the browser:
  RUS P1 DSQ, HAM +0.526, PIA +1.173 at the flag.
- Tests: 177 offline passing (+6 pipeline, +1 DSQ) + 2 network.
- Next M1 item: engine inputs from data (real start compounds, team car-pace offsets).

### 2026-10-12 — M1 slice 4: data into the engine (M1 complete)

- `f1ar.calibration.race_inputs` (start compounds, car-pace offsets); `DriverEntry.car_pace_offset_s`,
  `RaceSetup.inputs`; lap-model car term; engine v0.3.0, world schema v3 (D-021).
- `setup_from_codes(data_dir=...)` + `event_data_dir`; demo_headless / live_race / fork_demo /
  debug_viewer use real inputs by default (`--synthetic` to opt out) and print them.
- **EXP-001** (`scripts/experiments/exp001_data_inputs.py`, `docs/EXPERIMENTS.md`): Spearman vs the
  real order 0.743 → 0.879, position error 3.27 → 1.99, better in 30/30 paired seeds. But LEC's
  strategy win can't happen without pit stops (McLaren wins every seed).
- Tests: 183 offline passing (+6 inputs/engine/scenario) + 2 network.

### 2026-10-12 — M2 slice 1: pit stops, tyre choice, baseline strategy, chequered flag

- Engine v0.4.0, world schema v4 (D-022). Actions `RequestPitStop`, `CancelPitStop`,
  `SetStrategyController`; runner commands `PIT_REQUEST`, `PIT_CANCEL`, `SET_CONTROLLER`.
- A stop happens at the line crossing: new compound, tyre age/wear reset, and an out-lap that starts
  with a **hold** (pit-lane loss + log-logistic standstill, keyed by driver and lap; TUM idea) during
  which the car is parked at the line. 2024 dry rule (two compounds) checked at the finish.
- `f1ar.policy.baseline_strategy`: the cheapest legal 0/1/2-stop plan under the expected lap model,
  using the driver's own wear rate (`tire_management`, pace mode). Closed form + convexity, so a plan
  costs O(laps x compound pairs); proven equal to brute-force enumeration in tests. It replans after
  each stop and after a pace change. Any manual pit action takes the car over (`USER`); the
  baseline can be handed control back.
- **Chequered flag:** after the leader finishes, every car finishes at its next crossing, so lapped
  finishers exist (full Monza grid seed 42: 7 cars +1 lap; the real race had 6). Standings, timing
  sheet and results print `+1 lap`.
- Pit loss from data: `calibration.estimate_pit_loss` (in-lap + out-lap - 2 x median). Real Monza
  26.9 s over 30 stops, Spa 19.9 s over 34. The data setup uses it (`estimated_from_data`); the
  standstill part stays editorial.
- Viewer: Stops column, `PIT` / `BOX -> S` in the mode column, Box S/M/H / Stay out / Baseline
  strategy buttons, readable pit events. `live_race.py`: `b HAM h`, `o HAM`, `a HAM`.
  `fork_demo.py`: new **box now** branch. `demo_headless.py` prints stops and tyre sequence.
- Verified in the browser: NOR/PIA shown `PIT` while parked, HAM/SAI `BOX -> S`; selecting HAM ->
  Stay out (`BASELINE->USER`, `PIT_CANCELLED`), Box H -> `PIT_STOP MEDIUM->HARD ... 1.8 s stationary,
  26.2 s total`; race to the flag; HAM finished P5 on the user strategy.
- Gate demo: MATCH. Throughput unchanged (~0.44 s per 20-car race).
- **Finding:** with the placeholder compound coefficients (D-010) every car plans a one-stop onto
  SOFT around lap 27; real Monza was M->H. The planner is right for the model it's given; the
  compound model is what's wrong (M4 calibration; EXP-002 will measure it).
- Tests: 213 offline passing (+30: planner = brute force, legal plans, pit mechanics and hold,
  pit-loss arithmetic, cancel/take-over/hand-back, tyre-rule breach, standstill distribution,
  flag/lapped finishers, standings after the flag, timing sheet, view model, runner commands,
  **plan §4.7 test 5 with a pit decision** in a fork) + 2 network.
- Next: slice 2, traffic and overtaking.

### 2026-10-12 — M2 slice 2: traffic, dirty air, overtakes, blue flags

- Engine v0.5.0, world schema v5, observation schema v3 (D-023). New `simulation/traffic.py`
  (pure `resolve`) and `RaceSetup.traffic: TrafficParams` (editorial, with an `enabled` switch).
- Traffic is resolved when a car **joins the track**: line crossing, pit exit, or the start (lap 1
  in grid order). Line crossings and pit exits inside a tick are processed **in time order**, so
  every car reacts to cars ahead whose laps are already fixed.
- Rules: dirty air within 1.2 s; a car that would arrive within `min_gap` of a car ahead is
  **held** at exactly that gap, or **overtakes** with probability `1 - exp(-(margin - 0.3) / 0.8)`
  (keyed by attacker, defender, lap, so forks share it); lapped cars yield under **blue
  flags**. Defender/lapped-car losses are carried to their next lap. Events `OVERTAKE`,
  `BLUE_FLAG`; per-car `traffic_loss_s` (hidden) and `last_lap_traffic_loss_s` (observed).
- **Bug found while verifying in the browser:** the log said "VER passes HAM on lap 1", but VER
  stayed behind. A pass followed by a hold behind a car further ahead (or by the time lost
  passing more cars) left the attacker level with or behind the car it had "passed". Now every
  pass is re-checked after each adjustment: a pass that can't end ahead is undone (event and
  defender loss removed) and becomes a hold. In the seed-42 Monza race this removed 14 of 20
  "overtakes". A new test checks both directions in full 20-car races (3 seeds): no road pass
  without an event, and every OVERTAKE ends with the attacker crossing the line first; it fails
  without the fix.
- Effects now visible: grid position matters; **the undercut exists** (`fork_demo`: HAM boxing
  early loses 0.7 s overall but finishes P3 ahead of SAI instead of P4); a driver's decision
  changes rivals' races (HAM conserving holds SAI up by 12 s).
- Tests that pin "only that driver changes" (common random numbers) now run on a traffic-off
  setup (`isolated_setup` / `isolated_world` fixtures); new tests cover the interaction.
- Viewer: OVERTAKE / BLUE FLAG lines in the log, last-lap time tinted when time was lost in
  traffic (tooltip shows how much). Verified in the browser (screenshots at lap 1 and lap 28).
- Gate demo: MATCH. Throughput ~0.5 s per 20-car race (was 0.44 s).
- Seed-42 Monza (real inputs): 6 overtakes, 52 blue-flag passes, 9 cars lapped. Whether 6 is
  realistic is open; EXP-002 will count real on-track passes in the data.
- Tests: 227 offline passing (+14) + 2 network.
- Next: slice 3, race control (SC/VSC) and the mistake/DNF model.

### 2026-10-12 — M2 slice 3: incidents, retirements, VSC and the safety car

- Engine v0.6.0, world schema v6, observation schema v4 (D-024). New `simulation/incidents.py`;
  `RaceSetup.race_control: RaceControlParams` (editorial, with an `enabled` switch);
  `WorldState.race_control` (phase GREEN/VSC/SC, safety-car position, laps left) and
  `race_clock_s`; `CarStatus.RETIRED`.
- **Incidents** per car and lap from keyed randomness, drawn at lap start into hidden fields:
  mistakes (time loss), crashes and mechanical failures (retire where the car stops). Risk
  follows `consistency` and `composure` and the pace mode (PUSH x1.6, CONSERVE x0.6); lap 1 x6
  for crashes. `risk_appetite` stays out (D-010).
- **VSC:** lap progress runs on a racing clock at 70 % of sim time; pit-lane holds run in sim
  time, so a VSC stop is cheap. **Safety car:** a real car on track (placed 150 m ahead of the
  leader), the field queues behind it at 40 m with no overtaking, it pulls in after 3-5 laps,
  and the restart re-expresses each car's lap from its position and re-resolves traffic (keyed
  separately). The baseline strategy brings a planned stop forward if it's within 10 laps.
- **Race director:** `deploy_neutralisation` / runner `DEPLOY_SC`, `DEPLOY_VSC` / viewer buttons
  / `live_race.py` `sc` and `vsc`, for scenario exploration (M6 "what if a safety car on lap
  18?"); logged and reproducible like any command.
- **Viewer:** a safety-car sprite of our own (generic pixel-art road car, silver body, flashing
  amber light bar, "SC" tag; no real make or livery), the field queuing behind it, a blinking
  SAFETY CAR / VIRTUAL SAFETY CAR banner with laps/seconds left and cause, amber track edges,
  retired cars greyed out with a red cross where they stopped, readable race-control log lines.
- Verified in the browser: Deploy SC at lap 5 (banner "4 laps to go", amber edges, SC car ahead
  of NOR, train SC-NOR-PIA at the queue spacing, ~2:00 laps under SC, restart at tick 1821 with
  the field bunched within 3.6 s); Deploy VSC (banner with countdown); seed 14 where HAM crashes
  on lap 1 at Parabolica and race control sends the safety car out ("HAM crashed").
- **Bugs found while building it:**
  1. An overtake decided at the start of a lap was logged even when a safety car came out
     before it happened on the road. OVERTAKE is now announced when the attacker actually moves
     ahead (pending passes are dropped under SC, or if the rival retires or pits first).
  2. Re-resolving traffic mid-lap at a restart would have made cars jump; the restart keeps each
     car's position and stretches the rest of its lap to the resolved arrival time.
  3. Incident draws were recomputed every tick (1.38 s per race); drawn once per lap now
     (0.62 s per race).
- Rates over 40 seeded Monza races (real inputs): 0.88 retirements (0.40 crashes, 0.47
  failures), 0.30 safety cars, 0.17 VSCs, 5 mistakes per race; 48 of 781 stops were taken under
  SC/VSC. These are editorial targets in a plausible range, not calibrated (EXP-002).
- Gate demo: MATCH. Tests: 244 offline passing (+17: incident rates by pace mode, race-control
  choices, retirement handling, SC deployment, no overtaking and closing up behind the safety
  car, SC laps and jump-free restart, cheap stops, VSC speed and duration, pit lane at full speed
  under VSC, save/load mid-SC, reproducibility, race director) + 2 network. The independence
  tests now also switch race control off (a crash of one car can bring out a safety car for all).
- Next: slice 4, persistence (event + command log tables, named saves, autosave).

### 2026-10-12 — M2 slice 4: persistence (logs, autosave, crash recovery, invariants, pinning)

- **Store schema v3** (in-place migration from v1/v2, tested; the real `saves/viewer.sqlite` was
  migrated on open): `events` and `commands` tables per timeline; snapshots record their event
  and command cursors, the policy manifest and the dataset hash; timelines record the fork's
  event/command cursors. A save writes the snapshot and the log records up to it in one
  transaction. `store.events(timeline)` returns a branch's full history (parent prefix + own).
- **History is never rewritten:** `store.resume(snapshot)` continues the timeline only if the
  snapshot is its latest point (no later snapshot, event or command); otherwise it branches.
- **Runner persistence** (`WorldRunner(store=, autosave=AutosavePolicy(...))`): buffers every event
  and command record; writes them with each save, on pause, every 240 ticks and at shutdown;
  autosaves every 5 leader laps, after critical commands (pace, pit, strategy, race director)
  and at the flag, keeping the newest 5 autosaves (manual saves never pruned). Failures are
  reported in `runner.persistence`, never raised (US-09). `shutdown()` pauses a running race,
  autosaves and writes the logs (plan §4.7 test 8); the debug viewer calls it on New race /
  Restart and on app exit. Stored command keys of the timeline *and its ancestors before the
  branch point* are preloaded, so a command re-sent after a restart isn't applied twice.
- **Pinning:** `RaceSetup.policy_manifest` (`{"strategy": "baseline_strategy_v1"}`) and
  `inputs["dataset_sha256"]`; a save made with other policies is refused on load, and
  `create_world` refuses a setup that pins other policies. `WorldState.branch_event_seq`
  (plan's `branch_point_event_id`).
- **Invariants** (`simulation/invariants.py`, plan §18.2): per-state and per-step checks (clocks,
  distances, laps, tyres, statuses, race-control consistency, contiguous events, nothing changes
  after finishing/retiring). Checked on every tick of full 20-car races with traffic, incidents,
  a VSC and a safety car; `WorldRunner(check_invariants=True)` debug mode.
- **Crash-recovery integration test** (fresh process): a scripted client drives a race (pit
  request, race-director SC, push), the runner is abandoned mid-SC without shutdown, a child
  process restores the latest valid save and re-plays the client script. Final race and the
  full stored event history equal an uninterrupted run, and the re-sent pit request is applied
  once. Both paths: periodic log flushes (recovery branches) and saves-only (same timeline).
- **Bugs found while building it:**
  1. A recovery that branches would have re-applied re-sent commands (HAM pitting twice): the
     branch didn't know its parent's command keys. Fixed with command cursors and lineage-aware
     idempotency; the integration test fails without the fix (checked).
  2. Ids weren't unique: every world's main timeline was `timeline_main`, the viewer restarted
     at `viewer_0` each session and the demo always used `demo`, so two runs in one save file
     would have mixed their event histories. Now `new_world_id()` / `<world>.main`, and the
     store refuses a conflicting event or a timeline owned by another world instead of
     silently ignoring it (regression test). An older store test was mixing two worlds this way.
- Verified in the browser + save file: race, Deploy SC, New race -> `inspect_save.py` shows the
  autosaves "lap 5", "after DEPLOY_SC", "shutdown" (all OK, PAUSED) and 49 events / 5 commands
  on the timeline; the gate demo gives MATCH twice in a row in one save file (two worlds).
- Tests: 265 offline passing (+21) + 2 network.
- Next: slice 5, batch benchmark + EXP-002 + M2 DoD check.

### 2026-10-12 — M2 slice 5: Spa simulation, EXP-002, benchmark, engine port, DoD check

- **More than one simulated event** (D-026): `configs/drivers/seed_2024_belgian_drivers.yaml`
  (the same personas; SAR template-only), per-event track specs (`track_for`, Spa 7.004 km / 44
  laps), and a data-estimated base lap time (`calibration.estimate_base_lap`: Monza 81.7 s vs the
  82.0 placeholder; Spa 106.2 s). `debug_viewer.py --event 2024_belgian_gp` simulates Spa
  (launch config `spa-viewer`); verified in the browser: Spa outline, ~1:48.5 laps, a DRS-style
  train and an overtake on lap 3.
- **EXP-002** (`scripts/experiments/exp002_m2_engine.py`, results in `docs/EXPERIMENTS.md` and
  `docs/experiments/exp002.json`): finishing order and race time are close at both tracks
  (Spearman 0.885 / 0.905; race time within 0.3 min; Spa's official winner HAM wins 20/30); but
  every simulated car one-stops onto SOFT (real: mostly M-H with many 2-stops) and overtakes are
  2-5x too rare (10 vs 53, 17 vs 39). Incident/SC rates consistent with the two green races.
  These set M4's calibration order.
- **BENCH-001** (`scripts/benchmark.py`): ~1.5 s/race sequential on a loaded laptop, ~6,000
  races/hour on 16 processes (~145k/day vs the plan's 100/day floor). ~55 % of the time is
  pydantic copies. A worktree check showed the slice-3 code runs no faster on the same machine
  today, so the day's slower timings are machine load, not a regression.
- **`RaceEngine` port** (`simulation/engine_port.py`, plan §7.3): Protocol + `LightweightRaceEngine`
  (initialize, get/set_state with version and policy checks, observation, **valid actions**,
  apply_commands, bounded `step(max_sim_delta_seconds)`, events, positions, terminal) and a pure
  `lightweight_engine.valid_actions()`. A property test checks over a whole race that every
  listed action is accepted and every unlisted one rejected (M5's action masks rely on it).
- Bug fixed in EXP-002 itself before reporting: the tyre-sequence helper merged repeated
  compounds, hiding real M-M stints (ALO, STR at Spa); sequences are now per stint.
- Tests: 273 offline passing (+8) + 2 network.

#### M2 definition of done (plan §16) — check

| M2 task | Status | Evidence |
|---|---|---|
| Engine adapter contract and `step(max_delta)` | done | `engine_port.py`, `test_engine_port.py` |
| Tyres, tyre age, pit transitions | done | D-022, `test_pit_stops.py` |
| **Weather event and grip proxy states** | **not done** | no weather/wet tyres in the engine; see note |
| Driver profile -> pace/tyre effects (parametric baseline) | done (weather part not) | lap model (D-010/D-014), incidents (D-024) |
| WorldRunner, command queue, RUNNING/PAUSED | done (M0) | D-011, `test_world_runner.py` |
| SQLite snapshot + events + commands; idempotency, schema version, checksum | done | D-012, D-025 |
| Named saves and autosave | done | D-025, `test_persistence_logs.py` |
| Fork timelines + common-exogenous-randomness test | done | D-013, `test_fork.py` |
| Restore in a fresh process | done | `test_save_load.py`, `test_crash_recovery.py` |
| Invariants every tick; profile/model hash pinning | done | `invariants.py`, D-025 |
| Batch headless benchmarking | done | BENCH-001 |

| DoD criterion | Status |
|---|---|
| replay/property/integration test packages pass | yes (273 + 2 network) |
| after pause no canonical field advances | yes (`test_world_runner.py`, M0 §4.7 test 3) |
| a fork doesn't change the original | yes (`test_fork.py`, `fork_demo.py`) |
| state hash equal 1000 ticks after a save | yes (`test_save_load.py`, gate demo MATCH) |
| real state content stored, not an M0 placeholder | yes (full world + event/command logs) |

**M2 is complete except weather**, which the plan lists under M2 but which needs wet compounds,
a weather process and calibration (M4 territory). Decision for the owner: a weather slice now,
or move it to M4.

### 2026-10-12 — M3 slice 1: real API v1 (REST + WebSocket over one live session)

- **API** (`backend/src/f1ar/api/`: `main.py`, `routes_worlds.py`, `routes_races.py`,
  `websocket.py`, `session.py`, `catalog.py`, `schemas.py`, `views.py`; contract in
  `docs/API_CONTRACTS.md`; D-027): health, races (catalog of simulated + fetched events, verified
  entry list, track), world library, create, start/pause/resume/speed/step/save, actions (pit,
  pace, controller, race director), load ("continue" or a chosen snapshot), fork (with seed mode
  and decision overrides), timelines with snapshots, event history, driver card.
  `scripts/serve.py` (port 8000, `saves/worlds.sqlite`); launch config `api`.
- **One live world per backend** (`LiveSession`): create/load/fork shut the current runner down
  the safe way first (pause, autosave, logs). Command responses are the runner's ack (200
  accepted, 409 refused); save success only after the commit. Load is always PAUSED (a running
  save gets a logged `SYSTEM` PAUSE). Fork from a live world saves a canonical fork point first.
- **WebSocket** `/ws/worlds/{id}`: state frames on change (≤10 Hz, heartbeat 1 s) with
  `stream_seq`, `generation` and per-frame events; `reset` frames carry the timeline's recent
  stored history (after connect, fork, load); `closed` when another world becomes live.
  **Auto-pause** when the last client of a running world disconnects (plan §13.4 MVP).
- **Driver card** with a `source_type` on every value: identity and grid `historical_verified`,
  start tyre / car offset from the world's input notes, persona `editorial_assumption` (+
  confidence, template, overrides), live race values `simulated_dynamic`.
- Shared `presentation/event_assets.py` (moved out of `scripts/debug_viewer.py`); store
  `list_worlds()` (a query, no schema change).
- **Bugs found:**
  1. **Every load of a manual save branched** after a real restart (M2 slice 4 bug): the SAVE
     command's own log record is written after its snapshot, so `resume` counted it as later
     history once the logs were flushed. The existing test resumed before the flush and missed
     it. Now later commands that can't change the world (SAVE, SET_SPEED, refused ones) don't
     count; regression test both ways (a later STEP still branches).
  2. **Stream race during a world switch:** the session cleared the live world before the new
     one was ready, so a stream polling in between said "closed, live world: none" (2 of 6 runs
     of the test). The old runner now stays visible (paused, read-only) until its replacement is
     active. 15/15 runs green after the fix.
  3. The debug viewer's track payload had Monza's length hardcoded as `engine_length_m` (wrong
     for Spa; nothing read it). Now the event's own length.
- **Verified against the real server** (real Monza data, 20 cars, runner thread, real WebSocket
  client): create in 0.15 s; at x25, ticks 3..300 in 6 s (= 6 x 25 / 0.5) with 55 frames and
  monotonic `stream_seq`; pause/save acked at tick 300; driver card for NOR (`historical_verified`
  start tyre, `estimated_from_data` car offset, `editorial_assumption` persona); **backend
  restarted** -> library lists the world, not live -> load: tick 300, PAUSED, same timeline ->
  fork "NOR boxes for hards" (override accepted) -> 2 timelines. Swagger UI at `/docs`.
- Leftover fixed: `exp002_m2_engine.py` writes its JSON with `newline="\n"`.
- Tests: 292 offline passing (+19: 18 API incl. the restart -> load -> resume -> fork flow with a
  1000-tick determinism check, stream order/reset/close/refusal, auto-pause, crash-then-load; 1
  store regression) + 2 network.
- Next: slice 2, the React/Vite app.

### 2026-10-12 — M3 slice 2: the React/Vite app

- **`frontend/`** (D-028): Vite 8 + React 19 + TypeScript 5.9 (strict), exact versions in
  `package-lock.json`, `npm audit`: 0 vulnerabilities (the first pick, vitest 3, had critical
  tinypool advisories, so the dev tools are on current majors). Dev server on 5173 proxies `/api`
  and `/ws` to the backend (launch configs `api` + `ui`). `npm run build` -> 242 KB JS (77 KB gz).
- **Typed client + stream** (`src/api/`): command helpers return the ack (success only from an
  accepted ack); a pure stream reducer (stale frames dropped by `stream_seq`, reset frames
  replace the log unless it's a reconnect to the same runner and timeline, full history from
  REST merged after every reset) and a hook that reconnects after drops and stops when the world
  stops being live.
- **TrackCanvas**: real outline, corner numbers, start/finish line, team-coloured car markers with
  codes, selected car ring, retired cars greyed with a red cross, safety car marker, amber track
  edges under SC/VSC. 60 fps display interpolation along the outline by lap progress
  (`laps_completed + lap_fraction`), one frame interval behind the newest frame; exact positions
  when paused, in the pit lane, or on jumps. Backend: `lap_fraction` added to the car and
  safety-car views (display only; no engine change).
- **TimingTower** (position, team colour, code, gap, last lap with traffic-loss tint, tyre as a
  letter in a coloured ring so it reads without colour, tyre age, stops with a USER-strategy mark,
  PIT/BOX/OUT/FIN), **RaceControls** (Start/Pause/Play from the authoritative status, +1 tick /
  +10 s / +1 min while paused, x1/x5/x10/x25, Save: "Saving…" until the ack, then the tick, or the
  refusal reason), **RaceHeader** (race, lap, clock, status, "Alternate simulation" mode badge with
  engine/seed and an editorial-assumption note, Branch badge, connection state, SC/VSC banner),
  **EventLog** (readable lines with the race lap), **LapStrip** (pit / SC / VSC / retirement /
  overtake markers per lap; laps derived from the leader's lap completions, since SC/VSC events
  carry no lap). **StartScreen**: attach to the live world, "Continue" the newest world, or a new
  race (the World Library is slice 3).
- **Verified in the browser pane** (real Monza data, 20 cars): start screen through the proxy;
  new race -> Start at x5 (positions plausible: Lesmo at 0:39); x25; a race-director safety car
  (banner "SAFETY CAR · 2 laps", amber edges, SC marker ahead of NOR, the field queued behind);
  later a VSC; Save -> "Saved at tick 5074" only after the ack; selecting NOR in the tower rings
  the car on the map; backend restart -> the page falls back to the start screen ("That world is
  no longer loaded") with Continue; a page reload auto-paused the race (by design, D-027); no
  console errors; the narrow layout stacks the panels.
- **Bugs found:**
  1. In the first browser run the log put the safety car on lap 6 while it was lap 8: the client
     was missing two laps of lap-completion events. Two later runs (512 frames, 312 events) had no
     gap, so the cause isn't proven; most likely the hidden browser pane throttled the page. The
     client now **heals itself**: any frame or event sequence gap requests the full history again
     (unit-tested). After a reload the label was right (lap 8), confirming the mapping itself.
  2. A reconnect would have replaced the full log with the 50 events of the reset frame, and not
     refetched the history (the fetch was tied to new generations only). Fixed before testing.
  3. Overtake lines said "(lap 14)" next to the race lap "L15" (the car's lap vs the leader's);
     the car lap is no longer printed there. The save notice repeated the default label.
- Not measured: the live frame rate. The browser pane was hidden when I tried, and hidden pages
  don't run animation frames; interpolation is covered by unit tests only.
- Tests: 292 backend offline (unchanged count, one more assertion for `lap_fraction`) + 2 network;
  frontend 14 (`npm test`: track mapping vs the backend rule, interpolation, stream ordering /
  reconnect / reset / gap healing / history, formatting, event laps); `tsc` strict clean.
- Next: slice 3, World Library + save/load/fork UI + driver panel.

### 2026-10-12 — M3 slice 3: World Library, save/load/fork UI, driver panel

- **World Library** (replaces the start screen): every saved world with its race, creation time,
  timeline and save counts, LIVE badge and the save "Continue" resumes; saves from another
  engine version are marked as not loadable instead of failing; "Saves & timelines" lists every
  save to load; New race form (race, optional seed, speed). "← Library" in the race header (the
  race auto-pauses when you leave, D-027).
- **Timelines tab**: main timeline and branches as a tree, each save (manual/auto, race time,
  status) with **Load** and **Fork…**; the live timeline is marked; loading an older point says
  when it created a new branch (history kept).
- **Save as…** (named save), **Fork dialog** (from now or a chosen save; one decision: box for
  S/M/H, a pace mode, or take over the strategy; same luck or new luck; refused decisions are
  reported), **Race director** buttons (Safety car, VSC).
- **Driver panel** (click a car or a row): name, team, number; *Verified history* (identity,
  grid, start tyre), *Estimated from data* (car pace), *Editorial assumption* (persona, with
  label, template, adjusted fields, confidence, explanation), *Simulated* (live tyre, stint,
  plan); persona traits split into the four **used by the race engine** (`race_pace_skill`,
  `consistency`, `tire_management`, `composure`) and the ones not used yet. Decisions: Box S/M/H,
  Stay out, Push/Balanced/Conserve, Baseline/My strategy. Backend: `used_by_engine` in the
  driver card, guarded by a test that searches the engine sources.
- **Keyboard**: Space play/pause, S save, → +1 tick, Shift+→ +10 s (paused); ignored while typing
  or with a dialog open. Layout: three columns on wide screens (track | tower | panel).
- **Store schema v4** (D-029): `worlds.active_timeline_id`, set whenever a world becomes live;
  "Continue" resumes the newest valid save of that timeline (`continue_point`); in-place
  migration v3 -> v4 (the real save file migrated on open).
- **User stories done by hand in the browser** (real Monza data, 20 cars):
  US-01 create (seed 42, x10) -> new world; US-02 play/pause (also with Space); save as "before
  HAM decision" -> acked; US-04 fork "What if HAM boxes for HARD (lap 14)" -> Branch badge, HAM
  BOX + user strategy, played on: HAM pitted onto hards and dropped to P16; back to the main
  timeline (Load its newest save, no new branch): HAM P6 on 13-lap mediums, no stop; US-03
  stepped with Shift+→ (19:27 -> 19:57), saved with S, **backend hard-killed** -> library ->
  Continue -> main timeline at 19:57, PAUSED. Driver panel decisions (VER box for soft -> BOX)
  and race director (SC banner) checked too.
- **Bugs found (all in the browser):**
  1. **"Continue" after a crash resumed the wrong timeline**: after a fork and a return to the
     main timeline, the branch had the newest saves, so Continue dropped the user on the branch.
     Fixed with store v4 `active_timeline_id` (plan §13.1 had it); test with a crash in the API
     tests and in the store tests.
  2. **Dialogs**: `showModal()` focused the ✕ button, so typing went nowhere and Enter closed
     the dialog; and the dialog `close` event never fired in the browser pane, so Esc left a
     closed dialog mounted that couldn't be reopened. Now focus goes to the first field and Esc
     is handled through the synchronous `cancel` event.
  3. **Reopening the same world right after its stream closed bounced back to the library** (the
     old "closed" stream state was still there); the stream state is now reset on disconnect
     (unit-tested), and the exact sequence was re-run in the browser.
  4. The active tab rendered as a white box (style clash); the driver panel was squeezed under
     the tower (three-column layout now); snapshot actions wrapped.
- Keyboard shortcuts were exercised with dispatched key events (the browser pane was hidden
  for that part, so real key presses couldn't be sent); Space was also pressed for real earlier.
- Tests: 295 backend offline (+3: engine-traits guard, continue point, crash -> continue) + 2
  network; frontend 17 (+3: timeline tree, stream reset); `tsc` clean; build 263 KB JS.
- Not in this slice: historical replay inside the new UI (still the debug viewer); proposal: with
  M6's real-vs-simulated comparison.
- Next: slice 4, Playwright E2E + M3 DoD check.

### 2026-10-12 — M3 slice 4: end-to-end test and the M3 done-check

- **Playwright E2E** (`frontend/e2e/m3_flow.spec.ts`, `npm --prefix frontend run e2e`; D-030):
  real backend (`scripts/serve.py`, own save file, port 8001), its own Vite server (5174),
  Chromium, real Monza 2024 data (skips with a message if not fetched). One flow in 8 steps:
  create (seed 7) -> Start (clock moves) -> Pause (clock frozen for 1.5 s) -> +2 min -> Save as
  "e2e checkpoint" (dialog focus checked) -> **backend killed abruptly** and restarted -> page
  reloaded -> library lists the world -> Continue: same race time, same 20-car order, PAUSED,
  main timeline -> Play / Pause -> select HAM (driver panel with an editorial tag) -> Fork with
  "box for hards": Branch badge, HAM BOX, "original timeline is unchanged" -> Timelines: two,
  main has the checkpoint -> Load main's newest save: no branch, HAM not boxing.
- **Is it a real test?** A mutation run (last assertion inverted) failed exactly there, with HAM
  on main at P6 on mediums; 3 consecutive clean runs passed (~9 s each). A separate check showed
  that killing the backend process really frees the port, so the "restart" can't be the old
  server answering.
- Found while building it: Windows keeps a killed process's SQLite files locked for a moment, so
  the test's save file lives in Playwright's output folder (wiped at the next run) instead of
  being deleted at the end. The `[WebServer] ws proxy error: ECONNRESET/ECONNREFUSED` lines in
  the output are the Vite proxy seeing the deliberate backend kill.
- `README.md` was still at M0.5: now has the status, a "Play" section and the frontend/E2E
  commands.
- Tests: 295 backend offline + 2 network; frontend 17 unit + 1 E2E; `tsc` clean.

#### M3 definition of done (plan §16) — check

| M3 task | Status | Evidence |
|---|---|---|
| FastAPI GET/POST endpoints, data contracts, WebSocket state stream | done | D-027, `docs/API_CONTRACTS.md`, `test_api.py` (19 tests) |
| React/Vite app, TrackCanvas / TimingTower / RaceControls | done | D-028, `frontend/src/components/` |
| Monza centerline + 20 cars, race progress, UI event markers | done, with notes | FastF1 racing-line outline (not a surveyed centreline); 20 coloured markers with codes, not sprites (plan §14.4 "simple markers first"); lap strip + event log |
| Real backend connection for pause / save / load / fork | done | slices 1-3, E2E |
| Select a car/driver, see tyre and profile | done | driver panel (slice 3) |
| Profiles' `editorial_assumption` labels shown | done | source tags on every value, engine-used traits marked (D-029) |
| Race mode indicator: historical replay / alternate simulation | **partial** | badge shows "Alternate simulation"; replay mode stays in the debug viewer (proposed for M6) |
| x1, x5, x10 (+x25); save success only on backend ack | done | RaceControls, `useCommand`, E2E checks the ack notice |
| E2E: play -> pause -> save -> reload browser/backend -> resume -> fork | done | `e2e/m3_flow.spec.ts`, 3/3 runs, mutation-checked |

| DoD criterion | Status |
|---|---|
| The 2D demo really works with user input | yes: create, play/pause, step, save, fork with a decision, driver decisions, race director (by hand and E2E) |
| The full app can be reopened and continue where it left off | yes: backend killed abruptly + browser reloaded -> Continue restores the same race time, order and timeline (E2E; store v4 makes this follow the timeline the user was on) |
| Optional GIF/video for a portfolio | not made (optional; can be recorded from the E2E trace or by hand) |

Plan §13.2 endpoints not built in M3: `GET /api/races/{id}/reference` (historical reference
data, natural with M6's comparison), `GET /api/drivers/{id}/profile` (served per world instead,
`/api/worlds/{id}/drivers/{driver_id}`, so a save shows the profile it was created with), and
`GET /api/models` (no learned models before M5).

**M3 is complete except the historical-replay mode inside the new UI.** Next milestone: M4
(models and calibration: tyres + weather first, per EXP-002 and the weather decision).

### 2026-10-13 — M4 slice 1: the 2024 season dataset

- **All 24 races of 2024 fetched** (`scripts/fetch_season.py`; timing, weather, race control,
  no position telemetry: ~16 s and ~9 MB each; Monza and Spa keep their telemetry). Rosters,
  manifests and quality reports for the 22 new races are tracked.
- **Lap dataset** (`f1ar/calibration/lap_dataset.py`, `scripts/build_lap_dataset.py`; D-031):
  26,604 laps (22,719 representative green-flag laps), one row per lap. Every column is a
  **feature** with its availability (`pre_race`, `lap_start`, `decision`) and source type, the
  **target** (`lap_time_s`), or a **lap fact** (during-lap information for filtering only,
  e.g. the lap's track-status string and end-of-lap position). Only real-data values: personas
  are joined at training time, labelled. Features: circuit, driver, team, grid, scheduled laps,
  lap, laps remaining, compound, tyre age at lap start, fresh set, stint/stint lap, pit out/in,
  position at lap start, gap ahead, gap behind at the previous crossing, track status and
  weather at lap start.
- **Leakage audit** (property test): change everything after a time T (lap times, end
  positions, flags, later weather and track-status samples) and check that no feature of a lap
  that started by T moves; on a synthetic race at every lap start, and on the real Monza race.
- **Dataset manifest** `data/manifests/dataset_laps_2024_v1.yaml`: file and content hashes, row
  counts per race, feature schema with availability/source, the source manifests' hashes, the
  split rule and assignment. The build is deterministic (identical outputs across two builds
  with different Python hash seeds).
- **Splits by rule** (fixed before any model): wet = Canada, Britain, São Paulo; dry_test =
  rounds divisible by 4 (Japan, Monaco, Monza, Mexico, Abu Dhabi); dry_train = 16 races;
  leave-one-race-out folds over dry_train.
- **Season reference metrics** (`docs/data/season_2024_reference.md/.json`): per race stops per
  finisher, 1-stop / 2+-stop shares, top compound sequences, on-track passes, retirements, SC,
  VSC, red flags, wet-tyre share, winner, counted exactly like EXP-002 counts simulated races
  (the counter moved from the EXP-002 script to `f1ar/calibration/reference_metrics.py`).
  Dry-race means (21 races): 1.67 stops per finisher, one-stop share 0.42, **53.6 passes**,
  1.95 retirements, 0.48 safety cars, 0.38 VSCs per race.
- **Bugs and data findings:**
  1. **The leakage audit caught a leak on its first run**: "gap to the car behind" at the line
     where a lap starts is only known once that car crosses, after the lap has begun. Replaced
     by the gap behind at the previous crossing, used only if that car had crossed before the
     lap started (a car about to be lapped can cross later).
  2. **Red-flag tyre changes counted as pit stops**: 2024 Japan showed 3.0 stops per finisher
     and Monaco 1.44; entering the pit lane under a red flag (free tyre change) isn't a stop.
     Fixed: Japan 2.0, Monaco 0.44 (most drivers ran from the lap-1 red flag to the flag on one
     set, as in reality). Sequences still show the new stint.
  3. **13 of 22 new races failed a quality check** that compared lap rows with classified laps:
     FastF1 keeps a row for the lap a car retired on, and a disqualified car is classified with
     0 laps (Hülkenberg, São Paulo). Explained differences are now WARN, others still FAIL
     (tested); all 24 races WARN, none FAIL. Monza and Spa never had a mid-race retirement, so
     the check hadn't met this.
  4. **Spain was classified wet** by one stray rain sample (1 of 154, no wet tyres; the race was
     dry). Wet now means a wet-tyre lap or rain at >= 5 % of lap starts.
  5. Top compound sequences had ties ordered by set iteration (not reproducible); now sorted.
  6. The as-of joins failed on real data (nullable floats, laps without a start time), which
     the synthetic fixture didn't have; fixed, and the real-race audit covers it.
- Tests: 302 offline (+7: leakage audit synthetic + real Monza, conventions, wet rule and
  splits, red-flag periods, retirement/DSQ quality check) + 2 network; frontend 17 + 1 E2E.
- Next: slice 2, pace models and tyre curves (EXP-003).

### 2026-10-13 — M4 slice 2: pace models and tyre/fuel coefficients (EXP-003)

- **Engine shape** (`calibration/pace_models.py`): compound offsets, linear + quadratic wear per
  compound, fuel (+ track evolution) and a following cost, fitted by least squares with a pace
  term per driver per race; per-race adjustment (wear multiplier + offset deltas, ridge-shrunk
  to the season shape); the placeholder converted into the same form for comparison.
- **Online predictors** with as-of in-race context (`calibration/pace_features.py`: driver's last
  3 representative laps, field pace in the last 5 min; all from laps that had ENDED; the leakage
  audit now covers them too): rolling median, rolling median + shape correction, LightGBM on the
  residual, with ablations (no weather, + identity, + persona) and p10/p90 quantile and
  split-conformal intervals.
- **Results (EXP-003):** shape residual 0.579 -> 0.52 leave-one-race-out and 0.80 -> 0.66 on the
  held-out races vs the placeholder; + per-race adjustment 0.455 / 0.618. Online: rolling 0.578,
  Baseline A 0.549, LightGBM 0.530 s/lap on held-out races. Personas and identity add nothing;
  weather doesn't help in dry races; intervals 64 % (quantile) / 75 % (conformal) vs 80 %.
- **Key finding for the engine:** the placeholder's SOFT is 0.6 s faster than MEDIUM; real race
  trim says ~0.13 s (wide interval), with SOFT wearing fastest. This is why EXP-002 had every car
  one-stop onto SOFT. Degradation differs strongly by track (multiplier ~0 Monaco/Saudi, ~1.6
  Austria/Hungary), so the engine needs a per-race wear level.
- **Registry:** `configs/calibration/tyre_shape_2024_v1.json` (the fitted shape, tracked) and
  manifests `configs/models/tyre_shape_2024_v1.yaml`, `pace_lgbm_2024_v1.yaml` (dataset content
  hash, training races, config hash, seed, metrics, status); the LightGBM file itself is a local
  artifact (`artifacts/models/`, gitignored, rebuilt by the script). EXP-003 reproduces
  byte-for-byte.
- Found while building it: inside a stint tyre age and fuel are perfectly collinear (-1.0), so
  the fit needed a per-driver-per-race (not per-stint) pace term to separate them through stops;
  the fitted quadratic would make tyres faster beyond ~33 laps and Qatar's multiplier is
  negative: both need guards before engine use (slice 3).
- New dependencies: LightGBM 4.7 (MIT), scikit-learn 1.9 (BSD-3).
- Tests: 308 offline (+6: shape recovery exact/noisy, race multiplier, placeholder, rolling
  correction, recent-pace features; the leakage audit extended) + 2 network.
- Next: slice 3, the calibrated shape in the engine (v0.7.0) and EXP-002 re-run.

### 2026-10-13 — M4 slice 3: the calibrated engine (v0.7.0) and EXP-004

- **Wear by tyre age** with the guard: each lap adds the fitted curve's step at the set's age,
  never less than 0 (a concave fit flattens out). Linear (quad = 0) behaves exactly as before.
  Engine v0.7.0, world schema v8 (older saves refused; the library flags them, verified).
- **Calibrated lap model for data-backed races** (`calibration/lap_model_fit.py`): season shape x
  this race's wear multiplier (floor 0.1) and S/H offset deltas (in-sample), wear normalised by
  the field's persona wear factor; base lap and car offsets re-anchored with the new wear;
  `inputs["tyre_model"]` explains it. Skill, noise, pace modes stay editorial. A data folder
  without the race tables falls back to the season shape (labelled).
- **Planner rewritten** on per-age wear tables; the two-fresh-stint split is a min-plus
  convolution of convex costs (exact, ~0.5 ms per full replan); brute-force tests for linear,
  convex and concave wear from a used set.
- **Any 2024 race with its track outline is simulatable**: season persona file (Monza/Spa values
  verbatim, tested; Bearman, Lawson, Doohan template-only), track spec from data (laps from
  results, length from the outline, 1-2 % short). Position data fetched for Japan, Monaco,
  Mexico, Abu Dhabi. The app's library now offers 6 races; Monaco checked in the browser on its
  real outline (LEC leads from pole, ~1:19-1:22 laps).
- **EXP-004** (six races x 30 seeds): compound choices now match the real races where the data
  is clear (Monza and Spa M-S -> M-H; Mexico M-H/H-M like reality), order Spearman
  0.80-0.91, real winners dominate where they were fastest. Still one stop for every car
  (the empty-track planner never finds a second stop worth the pit loss with real race-trim
  wear) and overtaking isn't track-specific (Monaco 31 passes vs 11 real; Monza 10 vs 53).
- **Bugs found:**
  1. **Per-race prior scaled wrongly (from slice 2)**: it vanished for rarely used compounds, so
     Monza's single representative SOFT lap (a late fastest-lap attempt) gave SOFT -0.96 s and
     kept Monza on M-S. Fixed (prior = 200 laps of that compound), regression test; EXP-003
     re-run: the per-race row 0.455/0.618 -> 0.474/0.626 (corrected in the entry), Qatar's
     multiplier -0.24 -> 0.60; season shape and online results unchanged.
  2. The base-lap anchor still used the old linear wear; it now uses the engine's own wear.
  3. A data folder with only the lap table (tests) broke the setup, and the shape file was
     looked up relative to the data folder; now an explicit `configs_dir` and a labelled
     fallback. The race summary lost its lap count for races built without data; fixed.
- Tests: 322 offline (+14: curved-wear planner vs brute force x4, more stops with steep wear,
  wear step clamp, season personas x2 + every 2024 race, track spec from data, calibrated model
  floor/normalisation/deltas, shape file, Monza plans M-H, rare-compound prior, setup fallback)
  + 2 network; frontend 17 + E2E 1 (passes on v0.7.0).
- Next: slice 4, weather.

### 2026-10-13 — M4 slice 4: weather (engine v0.8.0) and EXP-005

- **Track wetness** (hidden, 0-1) with penalty lines per tyre; INTERMEDIATE and WET compounds;
  wet laps are noisier and incidents likelier; driver factor from persona `wet_skill`,
  normalised to the field's mean. Engine v0.8.0, world schema v9 (older saves refused).
- **Two weather modes:** HISTORICAL (real rain flags; wetness revealed from the real field's tyre
  choices, capped in the inter band unless most ran full wets) and RANDOM (seeded per-minute
  showers, editorial rates). Forks share the weather. Rain start/stop events.
- **Fitted** wetting/drying rates and slick/inter lines on Canada, Britain, São Paulo
  (`configs/calibration/weather_2024_v1.json` + manifest); full-wet line and factors editorial.
- **Baseline in the wet:** switch when gain x min(15, laps left - 1) beats the pit cost; back to
  slicks by laps left; slick plans dropped while slicks are wrong; USER cars never switched;
  wet tyre lifts the two-dry-compound rule.
- **Forecast without the future (US-06):** observation schema v5 with raining, wetness and a
  noisy 10-minute rain forecast; changing rain beyond the horizon doesn't change it (tested).
- **Wet-race inputs:** dry pace estimated on weather-adjusted laps; pit loss from green stops
  only (also affects dry races slightly); non-starters get the field's common start tyre.
- **UI/API:** library weather select (real conditions / random showers), `weather` in
  `POST /api/worlds`, header weather pill with 10-minute forecast, I/W tyre badges, Box I / Box W
  in the driver panel and fork dialog, rain events in the log. Checked in the browser on
  Britain (seed 1): 19 cars on inters at lap 32 (80 % wet, no full wets), 19 back on slicks by
  lap 42 (real ~39).
- **EXP-005:** fit is sensible on all three races (crossovers 0.56 / 0.85, lap MAE 2.28 vs 3.36 s)
  but unstable when a race is left out (negative result). Simulated switches within 2-3 laps
  of reality, partly circular because conditions are inferred from the real choices. Stop counts
  off; Canada/São Paulo winners don't match (wet skill is editorial).
- **Bugs found:**
  1. The first weather fit was degenerate (lap times alone); fixed with the choice term + bounds.
  2. A 5-lap switch horizon kept cars on slicks far too long; now 15 laps.
  3. Rain flags end long before tracks dry; HISTORICAL uses revealed wetness.
  4. Britain ran on full wets (browser); inferred wetness capped to the inter band.
  5. Wet-race pit loss 37-54 s (wet laps, SC/red-flag stops); fixed as above.
  6. ALB/STR (did not start São Paulo) got the default MEDIUM; now the field's common start tyre.
  7. **The intermittent auto-pause API test (since M3) was a real bug:** the test client cancels
     the WebSocket handler on close, which could cut off the disconnect cleanup, so the server
     still counted a viewer and never auto-paused. Cleanup now shielded; 3 full runs clean
     (before: 2 of 3 failed).
- Tests: 334 offline (+12: wetness dynamics, tyre bands, timelines, keyed random rain, forecast
  bounds and no-future, shower race inters and back, forks share weather, USER car not
  switched, observation, revealed wetness band, params round-trip, green-only pit loss)
  + 2 network; frontend 17. ruff/mypy clean.
- Next: slice 5, traffic and race-control calibration per track.

### 2026-10-13 — M4 slice 5: overtaking, the start and race control (engine v0.9.0) and EXP-006

- **Diagnosis:** passes split by cause, the same code for real and simulated races
  (`reference_metrics.pass_causes`: lap 1 / restart / fresher tyres / green). v0.8.0 had almost
  no lap-1 passes, few tyre-offset passes and no track dependence (Monaco 32 green passes, real 0).
- **Overtaking fitted on 9,405 real lap-level encounters** (not on pass counts), with the
  engine's own rule and its lap noise (`calibration/traffic_fit`): min gap 0.47 s, scale 1.04 s,
  per-track thresholds from Monaco +0.46 s to Las Vegas -0.19 s, cross-validated shrinkage.
  Well calibrated: predicted vs observed pass rate agree across the whole advantage range.
- **Launch spread** (new in v0.9.0): lap 1 gets a keyed `N(0, 1.71 s)` per car, matched by
  simulation to the real lap-1 passes of the training races. Checked in the browser (Monza seed
  7: HAM P6 -> P3 and VER P7 -> P4 by lap 2; a lap-1 crash brought a 6-lap safety car).
- **Race control from the season** (`calibration/race_control_fit`): retirement hazard on lap 1
  and after, SC/VSC per retirement, SC 4-6 laps; crash/failure split editorial.
- **All 24 races simulatable:** position data fetched for the other 15 races (16 manifests and
  quality reports refreshed; lap data unchanged).
- **EXP-006** (21 dry races x 10 seeds x 3 settings): pass error per race 30 -> 14, mean 25 -> 44
  (real 54); per-track thresholds help on the test races (correlation 0.74 vs 0.20), not on the
  training races; the mix is wrong (green 26 vs 14, fresher-tyre 4 vs 21: one stop for all,
  M5); Monaco 27 vs 11; retirements 2.1 vs 2.0, SC 0.62 vs 0.48, VSC 0.52 vs 0.38.
- **Bugs found:**
  1. **Fitted negative overtake thresholds broke loading a save** (found by the M0 gate test):
     `model_copy` skips validation, the schema said >= 0. Builders now validate; negative
     thresholds are allowed; the fit got the engine's "cars must meet" condition; EXP-006 re-run.
- Refactor: EXP-002's simulated-race metrics moved to `calibration/simulated_metrics` (shared).
- Tests: 346 offline (+12: pass curve, noise convolution vs Monte Carlo, parameter recovery on
  synthetic encounters, encounter selection, traffic/race-control builders, moments, launch
  spread keyed and additive, fitted files, save/load round-trip for every fitted track, "must
  meet" rule, pass causes) + 2 network; frontend 17; E2E passes, but 2 of 10 runs failed, one
  a Playwright worker crash (0xC0000409), the other not captured (see known limitations).
- Next: slice 6, M4 wrap-up.

### 2026-10-13 — M4 slice 6: wrap-up (registry, fallbacks, ablation, done-check)

- **No MLP** (D-036): EXP-003 showed no gain from driver identity or personas and only 3 % from
  LightGBM over the corrected rolling median; nothing for an MLP to close.
- **Model registry**: `calibration/registry.py` + `scripts/model_registry.py` check every
  `configs/models/*.yaml` against its artifact hash and the lap dataset hash and write
  `docs/MODEL_REGISTRY.md` (5 models, 4 used by the engine, LightGBM local-only). Artifact paths
  in the manifests made repo-relative; the writers updated to match.
- **Safe fallbacks** for every engine artifact, each labelled `editorial_assumption`:
  1. **Bug:** a missing tyre-shape file crashed the setup (FileNotFoundError); now the editorial
     tyre model is used.
  2. Missing traffic / race-control files fell back silently with the placeholder's note; now
     labelled like the others.
- **EXP-007 ablation** (8 races x 10 seeds x 5-6 settings): each part does its own job (tyre
  shape -> real compound sequences, traffic -> pass error 37 -> 20, race control -> retirement
  error 1.86 -> 1.32, weather -> wet tyres at all); none improves the finishing order
  (editorial 0.875 vs full 0.852); the real winner wins less with the fitted start and passing
  (0.32 vs 0.58).
- **Correction:** slice 3's checkbox mentioned an "optional learned-pace port" that was never
  built (D-032 kept LightGBM out of the engine); corrected in the plan list above.
- Shared helper: `spearman` moved to `calibration/simulated_metrics` (EXP-002 and EXP-007).
- Tests: 353 offline (+7: registry matches, registry flags changed/missing artifacts, a race
  builds and runs without each fitted file and without all four) + 2 network; frontend 17.
- **M4 complete.** Next: M5, Stage 1 strategy learning (plan §16).

### 2026-10-13 — M5 design agreed (D-037, `docs/M5_DESIGN.md`)

- Discussed M5 against the project's purpose (a live AI world). Owner decisions: winning depends
  on the championship and is a team principal decision; all 20 cars AI by the end of M5;
  characters change between seasons later; train on the GPU; engine-bug exploits are fixed,
  real novel strategies allowed and the FIA can ban them; the user can decide for anyone, the AI
  continues afterwards.
- Model technology: small PyTorch RL networks (one shared policy per role), not LLMs; LLMs only
  for Stage 3 language/negotiation (local, validator-gated).
- No code yet. Next: M5 slice 1.

### 2026-10-13 — M5 slice 1: the training environment (D-038, BENCH-002)

- **`f1ar/training/`**: `RaceSession` runs a race where any subset of cars is AI-run and stops at
  every decision (lap end; SC/VSC; rain start/stop); `RaceStrategyEnv` (Gymnasium, one learning
  car, optional AI opponents via a given policy, `action_masks()` for MaskablePPO); `run_race`
  for whole AI fields.
- **Actions:** 18 (pit choice x pace), masked by the engine and the FIA rule set
  (`fia_2024_v1`, no bans yet; a ban removes actions, tested).
- **Observation `rl_obs_v1`:** 58 numbers, nothing from the future (tested on the vector).
- **Objectives:** time / result / points / championship (vs the title rival); rewards at the
  race end through each driver's risk utility (character, editorial); a rule-based team
  principal picks title vs points from the standings and car strength.
- **Scenarios:** random race, seed, weather, AI cars; synthetic championship standings
  (labelled editorial) so the AI sees title fights and points chances.
- **AGENT controller:** AI cars are labelled as AI (tower "A", driver panel "AI policy"); a
  human pit or pace decision takes the car over (owner rule); the API can't set AGENT or use an
  AGENT actor id.
- **BENCH-002:** one AI car 81 decisions/s per core, a 20-car AI field ~1,200 decisions/s.
- Smoke check on real Monza: Gymnasium's env checker passes; a random policy pits 32 times and
  finishes P17 (as it should), the baseline never touches the AI car.
- Not checked in the browser: no AI runs in the app yet (slice 6); the UI labels are covered by
  the type check.
- Tests: 371 offline (+18: action encoding, masks and bans, decisions every lap, AI pit carried
  out and never pits on its own, race events trigger decisions, same seed same race, a whole AI
  field, human takes over, API can't impersonate the AI, observation names/values and no
  future, objectives, risk utility shapes, team principal, synthetic standings, Gymnasium
  checker + reproducibility, AI opponents) + 2 network; frontend 17; E2E passes.
- Next: slice 2, baselines (rules, enumerator, fork-based MPC) and EXP-008.

### 2026-10-13 — M5 slice 2: the baselines and EXP-008 (D-039)

- **Strategists:** B0 real 2024 stops, B1 the rule planner, B2 enumerator (every legal 0-3-stop
  compound sequence by dynamic programming, then the best 6 re-ranked by simulated races), B3
  fork-based MPC (every option played out in imagined futures, the rules driving afterwards),
  random. A paired evaluation harness (same race for every method).
- **Imagined futures** never see the real future: new random draws, and rain drawn from the
  forecast within its 10-minute horizon only.
- **Two-compound rule:** a breach is now scored as a disqualification for the AI (an AI would
  otherwise learn to skip the mandatory stop: an exploit, D-037).
- **EXP-008** (100 paired cases x 5 methods, 2 h 22 min): the rule planner is the bar. The
  enumerator ties it; **MPC is worse** (-0.037, CI below 0: noisy estimates, picking the best of
  noisy options); real 2024 strategies much worse in our engine; random far worse.
- **Bugs found:**
  1. **Imagined rain:** the 10-minute forecast became a whole-race rain rate, so dry races
     imagined showers in almost every future (a test caught it; the run in progress was stopped
     and repeated).
  2. **Tyre-rule exploit:** the engine only reported a missed mandatory stop; now penalised in
     the AI's scoring.
- Also: EXP-008 took 2 h 22 min instead of ~40 min estimated: each worker got only a third to a
  half of a core (likely Windows throttling background processes; the machine was 25-48 %
  loaded). Added progress output to the script and a tick safety net to the imagined-race loop.
- Tests: 381 offline (+10: plan enumeration legal and cheapest-first, DP vs brute force, real
  stops without red-flag entries, fixed plan carried out, imagined futures never use the real
  seed or later real rain, imagined rain follows the forecast, rollouts reproducible and
  side-effect free, MPC deterministic, paired evaluation, enumerator picks its best simulated
  plan) + 2 network; frontend 17.
- Next: slice 3, GPU training stack and the first learned policy vs B1.

### 2026-10-15 — M5 slice 3: the training stack and the first trained AI (D-040, EXP-009)

- **Stack:** `train` dependency group (torch 2.11 + CUDA 12.8, sb3-contrib 2.9 MaskablePPO);
  `training/ppo.py` (config, workers, progress + paired checks, checkpoints),
  `training/vec_env.py` (several envs per worker process, masks returned with each step),
  `training/warm_start.py` (rules teacher demonstrations, stop-weighted cloning, softening),
  `scripts/train_policy.py`. Update on the RTX 4070; simulation on 14 CPU workers.
- **Registry + pinning:** `training/policy_registry.py`: manifest (`configs/policies/`) + actor
  weights (`models/policies/`, tracked), hash / observation schema / feature list / actions /
  regulations / engine version checked on load, lineage fields; torch-free numpy runtime
  (`NumpyPolicy`, equal to torch in a test); `LearnedStrategist` (argmax, seeded sampling or
  forced pace); `RulesTeacher` (B1 as actions, identical to B1 in 17/18 checked races).
- **Pilots** (EXP-009 part 1): from scratch nothing is learned (flat reward: a near-random car
  is last whatever it does); reward scaling doesn't help; a stay-out prior learns but slowly;
  the rules warm start needs stop-weighted cloning (plain cloning learned "never stop") and
  softening (otherwise PPO freezes); the paired reward (my score - the rules' score in the same
  race) makes PPO progress near B1.
- **Main run:** 602k decisions in 84 min, stopped by hand (flat checks since 300k, throughput
  down to ~72/s); registered `strategist_m5s3_v1` (best on training cases, +0.052).
- **EXP-009 (negative):** on the 5 held-out tracks it is slightly *worse* than the rules: -0.029
  [-0.053, -0.009] on 200 new-seed cases, -0.028 [-0.061, +0.001] on EXP-008's cases; about MPC's
  level (+0.010 n.s.). Same stops as B1, no pace use; it prefers a SOFT final stint (Monaco
  0.40 -> 0.31). The cloned policy fails the same way before PPO: `rl_obs_v1` lacks each
  compound's pace/wear on the track, so compound choice can't generalise to unseen tracks.
- **Bugs / problems found and fixed along the way:** env factories weren't picklable for the
  grouped workers (cloudpickle); SB3's `has_attr` pulled whole envs through pickle; the paired
  rules race run at the episode end stalled the lockstep workers (now run alongside: 53 ->
  136 decisions/s); plain cloning silently learned "never stop" (caught by the stop-recall
  metric).
- **Throughput:** 130-235 decisions/s (i7-12650H: workers get ~0.6 core each in lockstep).
- Tests: 400 offline (+19 in `tests/unit/test_policy_training.py`: numpy runtime, registry refusals, repo
  manifests load, numpy = torch, paired reward = B1's score, grouped workers = DummyVecEnv,
  stay-out prior, cloning a rare stop rule, softening keeps argmax, deterministic and seeded
  play, forced pace) + 2 network. One full run sat >40 min (killed); two re-runs took 2 min 15 s:
  not reproduced.
- Next (owner's call): `rl_obs_v2` (per-compound offsets and wear, tyre wear in seconds) +
  retrain + EXP-009 again; then slice 4.

### 2026-10-15 — Research: what makes the difference in real F1 (`docs/RESEARCH_REAL_F1.md`)

- Four web research passes (cars, tyres/weather, drivers, race dynamics; sourced numbers) plus
  our own measurements on the 2024 laps; a factor-by-factor comparison with the engine and a
  sliced adaptation plan (W1-W6). Biggest gap: tyres (real 1.74 stops per finisher, ours 1.00);
  also lap noise 3x too small, SCs only after retirements, no red flags / DRS / penalties.

### 2026-10-15 — W1: tyre life (engine v0.10.0, D-041, EXP-010)

- Finding first: on the same lap of a 2024 race, compounds are about equally fast; the softer
  tyre's advantage shows as shorter life. So W1 models **tyre life** with management (CONSERVE)
  vs PUSH, a cliff and the fresh-tyre peak, instead of bigger compound pace gaps.
- Engine v0.10.0, world schema 11, `baseline_strategy_v2` (manages to its plan, 6.5 s per-stop
  margin from McLaren's 2024 review); `calibration/tyre_life_fit.py`, `scripts/fit_tyre_life.py`,
  `configs/calibration/tyre_life_2024_v1.json` + registry manifest; globals kappa 1.2 / delta 0.0
  fitted on the dry_train races only.
- EXP-010 on all 21 dry races (W1 vs v0.9, same seeds): stops per finisher error 0.66 -> 0.34
  (train) and 0.56 -> 0.28 (test); test correlation 0.75; many real sequences reproduced.
  Misses: Las Vegas, Qatar, less variety inside a race, same-lap compound pace, retirements up.
- Bugs found and fixed: an inverted managed-share formula in the planner, lost convexity beyond
  full management, a hand-back to the baseline silently dropped.
- Tests: 411 offline (+10 in `tests/unit/test_tyre_life.py`, +1 registry fallback; registry and two calibrated-engine
  tests updated for W1).
- Next (owner's call): W4 (race control and random events) per the plan, or a W1b joint fit.

### 2026-10-15 — W4: race control and random events (engine v0.11.0, D-042, EXP-011)

- From the 2024 timing: SC/VSC without a retirement (5 in 21 dry races), red flags (3 in 24),
  623 green stops' pit-lane times (team crew offsets, slow-stop tail 7.4 %), retirements per team
  (proxy for reliability), the 47 stewards' penalties by type.
- Engine v0.11.0 / world schema 12: debris neutralisations, red flags after crashes with a free
  tyre change for everyone and an SC restart, per-team pit crews and failure rates, 5/10 s
  penalties added at the finish, crash/mistake rates renormalised (risk_scale 0.61) for W1.
- `calibration/race_events_fit.py`, `scripts/fit_race_events.py`,
  `configs/calibration/race_events_2024_v1.json` + registry manifest; frontend log texts for
  RED_FLAG / TIME_PENALTY.
- EXP-011 (21 dry races, W4 vs v0.10, per-race seeds): red flags 0.12 (real 0.095), penalties
  2.03 (2.10), VSCs 0.37 (0.38), SCs 0.51 (0.43), retirements 1.49 (1.95; train 1.51 vs 1.63),
  passes 52.9 (53.6), order vs real 0.675 (v0.10 0.616).
- Found: keyed random draws don't include the event, so shared seeds repeat the same luck on
  every track (the first fit saw zero red flags); fits and EXP-011 use per-race seeds now.
- Checked in the app: a Bahrain race shows a false-start penalty in the race log.
- Tests: 421 offline (+9 in `tests/unit/test_race_events.py`; registry and inputs tests
  updated); frontend 17 (penalty and red-flag texts).
- Next (per the plan): W2 race pace realism.

### 2026-10-15 — W2: driver race pace, race form, lap noise (engine v0.12.0, D-043, EXP-012)

- Fitted on the 16 dry_train races (same-lap regression per race): each driver's pace vs his
  teammate (shrunk), race form SD 0.236 s, lap noise per driver (0.34-0.56 s, field 0.45 s);
  they replace the persona skill term and the persona noise in data-backed races. Overtaking
  threshold shifted +0.34 s (fitted) to keep passes at 2024's level.
- Engine v0.12.0 / world schema 13; `calibration/driver_pace_fit.py`, `scripts/fit_driver_pace.py`,
  `configs/calibration/driver_pace_2024_v1.json` + registry manifest.
- EXP-012: lap noise 0.48 vs 0.44 s (driver correlation 0.90), teammate gaps 0.21 vs 0.25 s,
  passes 54.2 vs 53.6 (72.5 before the shift); order vs the specific real race 0.675 -> 0.658
  (random race form: a trade-off, documented).
- Checked in the viewer: Spa lap times now vary realistically.
- Tests: 425 offline (+3 in `tests/unit/test_driver_pace.py`, registry updated).
- Next (per the plan): W3 racecraft.

### 2026-10-15 — W3: racecraft (engine v0.13.0, D-044, EXP-013)

- Overtaking refit on the 9,405 real encounters with the W2 lap noise (threshold 0.63 s, scale
  0.37 s) + attack/defend per driver (ridge-shrunk); launch skill per driver (lap-1 gains beyond
  grid and car pace); lap-1 risk by grid slot (Dartmouth shape); race-level shift +0.07 s and
  launch spread 1.62 s fitted by simulation on dry_train.
- Engine v0.13.0 / world schema 14; `calibration/racecraft_fit.py`, `scripts/fit_racecraft.py`,
  `configs/calibration/racecraft_2024_v1.json` + registry manifest.
- EXP-013: passes 54.9 vs 53.6; passes made per driver correlation 0.93 (0.91 without); lap-1
  gain per driver 0.21 (0.12); lap-1 retirements now concentrated in the midfield; order and
  stops unchanged.
- Bugs found: launch skill skipped when the launch spread was 0; racecraft traffic kept an
  "editorial" label without the old traffic file.
- Checked in the viewer: Spa start, VER gains four places from P6.
- Tests: 430 offline (+4 in `tests/unit/test_racecraft.py`; registry, inputs and W2 tests
  updated).
- Next (per the plan): W5 drivers' character, W6 weather.

### 2026-10-15 — W5: personas estimated from data where measurable (D-045, EXP-014)

- Rule: a persona field is replaced only if 2024 measures it reliably (odd vs even races,
  season reliability >= 0.3). Consistency (W2 lap noise, 0.76), overtake/defense skill (W3
  attack/defend) are rank-mapped; tyre management fitted (wear vs teammate/team) but reliability
  0.13: not applied; mistake proneness not measurable (blue flags).
- World schema 15 (`DriverEntry.persona_estimates`); `calibration/persona_fit.py`,
  `scripts/fit_personas.py`, `configs/calibration/personas_2024_v1.json` + registry manifest;
  driver panel tags estimated traits.
- EXP-014: personas vs data 0.33 / 0.32 / -0.14 / -0.23; race metrics unchanged (stops 1.39,
  passes 53.6, order 0.658).
- Bug found in the experiment itself: a lap-1 residual per grid slot within one race is always 0
  (one driver per slot); removed (W3 measures the launch over the season).
- Checked in the UI: Bahrain, VER's consistency tagged "Estimated from data".
- Tests: 435 offline (+4 in `tests/unit/test_persona_fit.py`, +1 registry case; inputs tests
  updated); frontend 17.
- Next (per the plan): W6 weather extensions, then M5 AI.

### 2026-10-15 — W6: wet races (engine v0.14.0, D-046, EXP-015)

- Measured first: track temperature has no effect on tyre wear in 2024 (within races and across
  tracks): not added. Real wet races had stops our baseline never made: new inters under SC/VSC.
- Fresh wet sets under SC/VSC fitted on the real neutralisations (sets >= 11 laps: 77% replaced);
  wet incident factor fitted by simulation on the 3 wet races (4.35, range 2-8; was 2.0).
- Engine v0.14.0 / world schema 16; `calibration/wet_race_fit.py`, `scripts/fit_wet_race.py`,
  `configs/calibration/wet_race_2024_v1.json` + registry manifest.
- EXP-015: wet stops Canada 1.0 -> 1.64 (real 2.2), São Paulo 0 -> 0.74 (real 1.0); incident
  totals match, the mix doesn't (too many retirements, too few neutralisations); order vs the
  real wet races lower; dry races identical.
- Checked in the app: Canada, SC at lap 17, most cars took new inters, VER stayed out.
- Tests: 441 offline (+5 in `tests/unit/test_wet_race.py`, +1 registry case).
- World realism plan W1-W6 complete. Next: M5 AI resumes.

### 2026-10-16 — M5 slice 3b: the strategist on the W1-W6 world (D-047, EXP-016, EXP-017)

- Env speed on v0.14: 69 decisions/s per core (BENCH-002: 81).
- Found and fixed before training: teacher, enumerator and MPC drove BALANCED although the rule
  planner manages its tyres since W1 (`rule_pace`); teacher demos were cached by size and seed
  only (would have reused v1 demos).
- EXP-016: on v0.14 the rules vary stops (24 % two-stops); a plan fixed at the start loses to them
  (-0.025, significant); MPC loses more (-0.085). B1 stays the bar.
- `rl_obs_v2` (86 features: per-compound tyres, own tyre life, the planner's advice, neighbours'
  tyres) + tyre randomisation; clone on held-out tracks ties B1 (EXP-009's lost -0.039).
- PPO 800k decisions (99 min): ~1 % of decisions changed, no held-out gain (-0.004 / -0.006, CIs
  straddle 0). `strategist_m5s4_v1` registered (negative verdict in its manifest).
- Machine: a run launched with a shell `&` survived and a relaunch hit WinError 1455 (paging file):
  one 14-worker torch job at a time.
- Tests: 445 offline (+4 `tests/unit/test_obs_v2.py`; cloning test made width-robust).
- Next (owner's call): slice 4, objectives and character, where the rules aren't optimal.

### 2026-10-16 — M5 slice 4: goals, character, racecraft intent (D-048, EXP-018)

- Engine v0.15.0 / world schema 17: ATTACK / NORMAL / DEFEND per lap (editorial sizes on W3/W4);
  54 actions; `rl_obs_v3` (rival, points at stake, neighbours' racecraft); user command, API and
  driver-panel buttons (checked in the app: "VER racecraft NORMAL -> ATTACK from lap 2").
- Hand-written racecraft rule: -0.007 vs rules (n.s.), changes 26 of 100 results.
- Attempt 1 (clone incl. intent): never attacks; ties the rules. Attempt 2 (clone pit/pace only,
  intent kept uniform): explores (attack 40 % early), but ends tying the rules with intent not
  situational and no goal/character conditioning (EXP-018). Registered `strategist_m5s4b_v1`.
- Diagnosis: a decision's effect (0.05 % crash chance per ATTACK lap) is hundreds of times below
  the race noise (sd 0.15) with a terminal reward.
- Bugs found: handing a BASELINE car back kept ATTACK; demos cache key missing character
  randomisation; cloning made intents collapse (rich get richer); tests on the 18-action layout.
- Tests: 454 offline (+8 `test_racecraft_intent.py`, +1 world runner), frontend 17.
- Next (owner's call): decision-level credit with common random numbers.
