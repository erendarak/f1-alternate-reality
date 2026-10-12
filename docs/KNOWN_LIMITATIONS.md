# Known limitations

Honest list of what does **not** work or is only approximate. Update whenever one is found or fixed.

## Current (end of M2)

- **No weather at all** (no rain, wet/intermediate tyres, temperature or grip changes), although
  plan §16 lists weather states under M2. Both reference races were dry.
- **Strategy and overtaking don't match real races** (EXP-002): every simulated car one-stops onto
  SOFT (real: mostly M-H, many 2-stops) and on-track passes are 2-5x too rare. Root cause: the
  placeholder compound/degradation and traffic coefficients (calibration is M4).
- Data-derived inputs (car offsets, base lap, pit loss, start tyres) are estimated from the same
  race they're used on, at every track; EXP-002's Spa run is out-of-sample only for the
  coefficients and personas set at Monza.
- Only 2 events have personas (Monza, Spa 2024); other fetched events are replay-only.
- Engine speed: ~55 % of time is pydantic copying; fine for M5's needs (BENCH-001), but it grows
  with every CarState field.

## M2 slice 4

- **Crash recovery loses the race since the last autosave** (default: up to 5 laps or the last
  critical command); a client must re-send its commands, as the scripted test does. If logs were
  flushed past that autosave, recovery continues in a new branch rather than the same timeline.
- **Snapshots are full copies** (~10-60 KB each); no deltas or compression yet.
- Idempotency keys are per timeline lineage; reusing a key for a *different* command on another
  branch is allowed.
- Store schema migrations exist (v1 -> v2 -> v3), but **world saves from older engine versions are
  still refused** (deliberate, D-014); there is no world-state migration.
- The debug viewer only autosaves; there's no load/resume UI yet (scripts and the store API only;
  the real UI is M3).

## M2 slice 3

- **Incident and neutralisation rates are editorial** (about 0.9 retirements, 0.3 safety cars and
  0.17 VSCs per race at Monza); every car has the same mechanical reliability; there are no red
  flags, penalties, damage-without-retirement, or start-performance (launch) model.
- **Safety-car model simplifications:** no lapped cars un-lapping themselves, no "safety car in
  this lap" lights before the line, no formation behind the car at the restart beyond the 40 m
  spacing; a car can stop dead for a tick if it starts closer than the spacing. Under SC the
  pit-lane entry/exit isn't drawn either.
- **VSC slows everyone by exactly the same fraction**; real VSC delta times vary per sector and
  drivers gain/lose a little.
- A crashed car stays on track (drawn greyed out) but doesn't block or slow anyone.
- The **retired car's followers** keep the hold they had computed against it until their next
  line crossing (small, local inaccuracy).

- **Traffic parameters are editorial** (min gap 0.4 s, threshold 0.3 s, dirty air 0.15 s, ...);
  overtaking frequency isn't calibrated. Seed-42 Monza gives 6 overtakes; the real count is
  unknown to us until EXP-002 counts on-track passes in the data.
- **Overtakes are decided when the attacker joins the track** (line or pit exit) for the whole
  lap; a car can't react mid-lap, and the pass point on screen is wherever the linear motion
  crosses, not a braking zone. There's no DRS, no side-by-side battle, and no "defending" choice.
- **The baseline strategy ignores traffic**: it doesn't time stops for clean air or to undercut,
  even though the undercut now exists in the engine.
- No start performance (launch, lap-1 chaos): lap 1 is resolved in grid order with the normal
  rules.

- **Every baseline car plans a one-stop onto SOFT** (~lap 27 at Monza) because the placeholder
  compound coefficients (D-010) make the soft a race tyre. The real race was M->H. The planner is
  correct for its model; the compound model needs calibration (M4, measured in EXP-002).
- **Strategy reacts to SC/VSC only with a simple rule** (bring a stop within 10 laps forward);
  no tyre-choice change for the shorter remaining stint, no "stay out to gain track position".
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
- TUM-derived ideas used so far: log-logistic standstill, overtaking time loss, per-driver
  accident probability with a lap-1 multiplier, FCY phases (all re-implemented, editorial values).

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
- Every car finishes all 53 laps; nobody is lapped or flagged at the leader's finish.
- There's no real lap data or UI yet.
- 12 of 20 drivers use unmodified generic templates, so several drivers share identical personas
  (see `ASSUMPTIONS.md`).
- The roster comes from a single source (Jolpica). It gets cross-checked against FastF1 in step 7.
- Determinism guarantees will cover **CPU mode only**. BLAS/GPU/parallel bitwise equality is out of scope
  (plan §4.5).
