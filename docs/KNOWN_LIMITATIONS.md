# Known limitations

Honest list of what does **not** work or is only approximate. Update whenever one is found or fixed.

## Current (M2 slice 1)

- **Every baseline car plans a one-stop onto SOFT** (~lap 27 at Monza) because the placeholder
  compound coefficients (D-010) make the soft a race tyre. The real race was M->H. The planner is
  correct for its model; the compound model needs calibration (M4, measured in EXP-002).
- **Strategy ignores traffic, rivals and safety cars** (none are modelled yet): there's no
  undercut/overcut, and the baseline plans an "empty track" optimum. Slices 2-3.
- **Pit stop happens on the start/finish line.** The pit lane isn't drawn; a pitting car parks at
  the line for the whole hold (pit-lane loss + standstill), so it sits on top of the line in the
  viewer, and its display gap reads +0.0 while parked. Entry/exit time loss is lumped into the
  out-lap.
- **Pit loss is one race's median of whole-stop losses** (it includes cold-tyre out-laps and any
  traffic); the standstill distribution (median 2.4 s, shape 9) is editorial.
- **Tyre rule breaches are only logged** (`TYRE_RULE_BREACHED`); no penalty or disqualification.
  No tyre allocations (sets per compound) yet: any compound can be fitted any number of times.
- A `USER`-controlled car gets no automatic stops at all; handing it back to the baseline replans.
- **Car pace offsets are in-sample, one-race estimates** (Monza 2024 only, mixing fuel/tyre/traffic
  effects). See EXP-001.

- **Replay positions are interpolated linearly in time within each lap** (the timing feed gives lap
  start/end, not continuous speed). Cars move at constant speed through each lap, including in/out
  laps: the **pit lane isn't drawn**, so pitting cars drive the racing line, just slower.
- The side-by-side spreading in the viewer is cosmetic and can't tell who is really on which side.
- Fetched so far: 2024 Italian GP and 2024 Belgian GP (any race can be fetched). Driver personas, and so
  the *simulation*, exist only for Monza 2024; other events are replay-only.

- **Debug viewer:** it polls at 10 Hz with linear interpolation; at x25, dots move in visible steps.
  There's no pit lane and no zoom/follow camera (M3); car sprites are oversized (24 px) and overlap
  when close. Gaps are display estimates from
  track position at a nominal pace. It needs `scripts/fetch_monza_2024.py` to have run (outline,
  corners, colours are local data).
- Cars drive the sampled racing path at constant speed per lap (no corner slow-down), so dots move
  uniformly. The race result is unaffected.

- **Engine throughput** is about 0.5 s per 20-car race at 0.5 s ticks (~2 races/s). That's fine for live
  play and Monte Carlo of tens of races, but possibly slow for large RL training. Plan: a lap-level
  fast path sharing the same lap model (SIMULATOR_DECISION.md follow-up 3).
- The engine has no safety car, VSC, failures, accidents or retirements yet (M2 slice 3).
  TUM-derived ideas used so far: log-logistic standstill only.

## M0, step 7

- Real Monza 2024 data is fetched and validated but **not yet used by the engine**. The race still
  runs on the placeholder lap model, with every car on MEDIUM, no pit stops, no lapping and no DNFs,
  whereas the real race had 6 HARD starters, 31 pit stops, 6 lapped cars and a lap-7 retirement.
  Wiring the data in is M1/M2.
- The track outline is one car's ~4 Hz path over one lap: good enough for a 2D view, not a precise
  geometry (no width, no pit lane).
- FastF1 under NumPy ≥ 2.5 prints deprecation warnings. With warnings as errors it fails to load.
  The lockfile pins working versions; re-run `uv run pytest -m network` after dependency upgrades.
- Only one event (2024 Italian GP) has been fetched. Generalising beyond it needs more races (M1/M4).

- The only decision a fork can differ by is **pace mode**. Pit stops and tyre choice come in M2.
- Without pit stops, pushing usually pays off, because tyres can't "fall off a cliff" and must last
  the whole race anyway. The real push/save trade-off needs pit stops and a non-linear wear model.
- Saves written by engine v0.1.0 (before 2026-10-12) can't be loaded by v0.2.0. This is deliberate,
  see D-014. Delete `saves/demo.sqlite` or ignore its INVALID entry.
- There's no runner-level FORK command or UI yet. Forks go through `SnapshotStore.fork` (API in M3).

- Saves hold the full world state, but the **event log and command log are not persisted yet**
  (in memory only). There's no autosave yet.
- Loading requires the exact same engine/schema/codec version. Old saves become unloadable when the
  engine version is bumped, until migrations exist (deliberate: refusing beats silent drift).
- The demo's reference determinism check compares final states only, not every intermediate tick.
  The test suite covers intermediate ticks (1000 vs 400 + 600).
- The real-thread pause test waits 1 s of real time, not the plan's 5 s. The exact 5 s case is covered
  with a fake clock.
- Runner listeners run on the runner thread and must be quick; a slow listener slows the sim.
- **The race model is a synthetic placeholder.** Its coefficients are uncalibrated, and there are
  no pit stops, strategy decisions, weather, safety cars or DNFs.
- Car performance now comes from data (D-021) when the race is fetched; with `--synthetic` or
  without data, cars are equal again.
- **No traffic or overtaking model:** cars pass through each other with no time cost, so grid
  position barely matters.
- Every car finishes all 53 laps; nobody is lapped or flagged at the leader's finish.
- There's no real lap data or UI yet.
- 12 of 20 drivers use unmodified generic templates, so several drivers share identical personas
  (see `ASSUMPTIONS.md`).
- The roster comes from a single source (Jolpica). It gets cross-checked against FastF1 in step 7.
- Determinism guarantees will cover **CPU mode only**. BLAS/GPU/parallel bitwise equality is out of scope
  (plan §4.5).
