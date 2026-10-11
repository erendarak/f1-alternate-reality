# Progress

## Current milestone: M1 — Historical data layer (plan §16)

M0 and M0.5 are complete (gate 5/5; see below). M1 tasks, adapted to what step 7 already delivered:

- [x] Repeatable fetch with cache, typed schema, units, validation, manifest SHA-256 (step 7).
- [x] Grid, results, pit/stint data verified, with an independent Jolpica cross-check (step 7).
- [x] Track geometry (FastF1 outline + corners instead of the TUM centreline, D-016/D-017).
- [x] Offline synthetic fixtures for all data code (step 7, M1).
- [x] **Historical reference race + reference event list; `HISTORICAL_REPLAY` separated in code (D-018).**
- [x] Real race replay on the debug viewer map (play/pause/speed/seek; no fork, no actions) + camera.
- [x] More events through the same pipeline: `fetch_event.py --season --round` (D-020); 2024 Belgian GP added.
- [ ] Engine inputs from data: real start compounds, per-driver/team pace baselines from
      representative laps (feeds M2/M4 calibration).

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
