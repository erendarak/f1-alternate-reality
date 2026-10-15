# Known limitations

Honest list of what does **not** work or is only approximate. Update whenever one is found or fixed.

## M5 slice 3 (training stack, first learned policy)

- **The first learned policy is slightly worse than the rule planner on held-out tracks**
  (EXP-009: -0.029 on new seeds): it learned compound choices that fit the 16 training tracks
  (a soft final stint) and lose on unseen ones (Monaco most). Cause: **`rl_obs_v1` has no
  per-compound track data** (each compound's pace offset and wear rate), which the rule planner
  uses. Not to be used in a world; fix proposed (`rl_obs_v2`).
- **It learned no pace behaviour**: PUSH/CONSERVE are never chosen (the teacher never used them
  and PPO didn't find them worth it from there); stop counts equal the rules'.
- **The first policy starts from the rules** (behaviour cloning of B1's planner, then PPO,
  D-040): where it plays like B1 it copies the teacher. Training from scratch didn't reach the
  rules within the pilots' budget (flat reward for near-random play; a stay-out prior learns
  but slowly).
- **One learning car vs the rule field, dry races, points objective only**: no championship
  context, no character/objective conditioning beyond the observation (slice 4), no AI field
  (slice 5), no weather (slice 6).
- **Best checkpoint chosen on 24 training cases**: a noisy choice (the check moves by a few
  hundredths between checkpoints); held-out tracks are only used in EXP-009.
- **The observation has no tyre wear in seconds** (only tyre age, compound and the track's
  medium wear rate): the policy infers wear from age; adding it means `rl_obs_v2`.
- **The main run was stopped by hand** at 602k of 800k planned decisions (flat checks, falling
  throughput); recorded in the manifest.
- **Training throughput is CPU-bound and low on this laptop** (130-235 decisions/s, 14 workers
  at ~0.6 core each): long runs take hours; the Windows background-process throttling setting
  (`powercfg /powerthrottling`) is the owner's call.
- **GPU training is reproducible from its seeds in practice** (two starts gave identical progress
  lines), but CUDA arithmetic isn't guaranteed bit-identical across machines/drivers; a world
  pins the exported weights, not the training run.
- **The paired reward needs the rule planner's race** for every training race; it only exists
  for a car the rules could drive (fine in M5; a whole AI field needs another baseline).

## M5 slice 2 (baselines)

- **MPC (B3) is weaker than the rules** in EXP-008 (estimation noise with 3 imagined futures);
  it isn't a strong upper reference, and making it one costs far more compute.
- Imagined futures keep the lap in progress as the engine already drew it (including an
  incident drawn at its start): a small look-ahead advantage for B3 and B2's re-ranking.
- The enumerator plans on an empty track with fixed BALANCED pace and dry tyres only; the
  baselines don't optimise pace and aren't meaningful for wet races yet.
- The engine's standings in the app still classify a car that missed the mandatory stop
  normally; only the AI's scoring disqualifies it.
- One AI car per race against a rule field; whole AI fields come in slice 5.
- Long experiments run 2-3x slower than single-race timings suggest (Windows throttles the
  background worker processes on this laptop).

## M5 slice 1 (training environment)

- **Editorial pieces of the AI's goals:** the team principal's rules, the risk-utility curvature,
  the position tie-breaker in the points objective, and the synthetic championship standings
  used in training. They shape what the AI wants; they are not learned or measured.
- **Rewards only at the race end** (sparse): learning may be slow; slice 3 decides on shaping
  or MPC warm-starting, with the reason recorded.
- **Points:** only classified finishers in the top 10 score (no fastest-lap point, no sprints,
  no 90 % distance rule for retirements).
- **Gaps ahead/behind** are estimated from track positions x the base lap, not from timing loops.
- **Forecast changes don't trigger a decision** (the forecast is only in the observation).
- The pace mode's effects (PUSH/CONSERVE) are editorial (D-010): an AI that learns to push learns
  our assumption, which the slice 5 exploit audit must check.

## End of M4 (slice 6, EXP-007)

- **No calibrated part improves the finishing order** (EXP-007: the editorial engine agrees
  slightly better, 0.875 vs 0.852): order comes from grid and car pace estimated from each race.
- **Real winners win less often** with the fitted start and passing (0.32 vs 0.58 of runs on
  the dry test races).
- **No red flags** in the simulation: race times at Japan, Monaco and São Paulo 2024 are 27-39
  min off.
- Season race-control rates come from dry races; on wet races the fitted rates do slightly
  worse than the editorial ones (3 races).
- The LightGBM predictor exists only locally (gitignored) and is not used by the engine; M5 must
  rebuild it with `exp003_pace_models.py` if the strategist uses it.
- Everything rests on one season (2024): 3 wet races, one race per track.

## M4 slice 5 (overtaking, start, race control, engine v0.9.0, EXP-006)

- **The pass mix is wrong:** too many same-tyre green passes (26 vs 14 per race) and far too
  few fresher-tyre passes (4 vs 21), because every simulated car still one-stops at about the
  same time (baseline planner, M5). Totals look closer partly because one error covers another.
- **One race per track:** per-track thresholds are in-sample and noisy (they don't beat one
  season threshold on the training races). Monaco still has too many passes (27 vs 11).
- **The launch spread is season-wide** and was matched by simulation (one parameter, not a model
  of starts): no grid spacing, no first-corner contact model; lap-1 passes on the test races
  12 vs 20.
- **Restarts rarely change the order** (1.2 vs 3.2 passes per race).
- **Race control:** "Retired" doesn't say why, so the crash/failure split (and the lap-1 crash
  multiplier of 37) is editorial; SC/VSC per retirement is an average over causes (debris SCs
  have no cause in the engine); SC and VSC slightly too frequent (0.62 vs 0.48, 0.52 vs 0.38).
  Rare events: a training-only fit predicted 0.9 lap-1 retirements for the test races, 9
  happened. Mistakes, pace-mode risk factors, VSC length and passing time losses stay editorial.
- No red flags in the simulation (Monaco, Japan 2024 had lap-1 red flags).
- **E2E intermittent on Windows:** 2 of 10 runs failed, one a Playwright worker crash
  (0xC0000409), the other not captured; no failed test assertion was seen.

## M4 slice 4 (weather, engine v0.8.0, EXP-005)

- **The weather model rests on three races** and is unstable when one is left out (EXP-005:
  held-out crossovers anywhere from 0.09 to 1.9; held-out Britain lap error worse than no weather at
  all). The all-race fit is used; treat it as experimental.
- **HISTORICAL switch timing is partly circular:** track wetness is inferred from the real field's
  tyre choices, so simulated switches matching the real laps is expected, not proof of a
  learned behaviour. The inference also lags real conditions (a minute grid, a logistic on the share
  of wet-tyre runners) and can't tell inters from full wets: it is capped in the inter band unless
  most wet-tyre runners were on full wets.
- **Editorial:** the full-wet line, `wet_skill` effect size (persona `wet_skill` itself is
  editorial), wet noise and incident factors, switch horizon (15 laps), forecast noise,
  random-rain rates. The drying rate sits at its lower bound (5 min).
- **Baseline stops in the wet are off:** Britain about 2.8 stops per finisher (real 2.4); Canada and
  São Paulo too few, because the baseline never stops for fresh wet tyres (wear on wet tyres is
  an editorial placeholder) and the real red flag at São Paulo (free tyre change) isn't simulated.
- **Drivers who did not start** the real race (São Paulo: ALB, STR) start in the simulation, on
  the field's common start tyre.
- No standing water, aquaplaning, red flags for rain, or per-corner drying; wetness is one number for
  the whole track.
- Dry races are slightly affected too: pit loss now comes from green stops only (EXP-004 numbers
  are engine v0.7.0).

## M4 slice 3 (calibrated engine v0.7.0, EXP-004)

- **Every simulated car still makes exactly one stop** (real 0.44-2.0 per finisher): with
  real race-trim wear a second stop never pays on an empty track, and the baseline planner
  ignores undercuts, traffic and track position. A strategy problem for M5, not a tyre-model
  bug.
- **Overtaking isn't track-specific** (Monaco 31 simulated passes vs 11 real; most tracks far
  too few). Slice 5.
- Final-stint compound can flip on small, in-sample offset differences (Abu Dhabi plans M-S,
  real M-H).
- Per-race wear level and offsets are estimated from the same race (in-sample), like base lap
  and car offsets; only the season shape is out-of-sample for the dry_test races.
- No red flags in the simulation (Japan's real race time includes a ~30 min stoppage).
- Track length for new races comes from the outline (1-2 % shorter than official); 18 of 24
  races still lack position data and can't be simulated yet.
- Persona wear factor normalisation makes the field average match the data, but individual
  drivers' wear differences are still editorial (tire_management).

## M4 slice 2 (pace models, EXP-003)

- **Compound offsets are poorly identified** from race laps (selection: teams choose compounds and
  stint lengths; tyre management): SOFT vs MEDIUM -0.13 s with an interval from -0.44 to +0.28.
- **Fuel and track evolution can't be separated** with timing data (one coefficient).
- **The following cost (0.43 s) mixes dirty air and being held up**; not usable directly as the
  engine's dirty-air parameter.
- Per-race adjustments and nuisance pace terms are in-sample; only the season shape is tested on
  unseen races. Concave wear is not established (intervals include 0).
- Online intervals are too narrow on new races (75 % conformal coverage vs 80 % target).
- One season, 16 training races; Japan and Monaco (red flags, cruising) dominate the test error.
- The LightGBM artifact is local (gitignored) and rebuilt from the script; only its manifest and
  hash are tracked.

## M4 slice 1 (season dataset)

- **One season only (2024)**: 21 dry and 3 wet races. Leave-one-race-out over 16 training races
  is small for some per-track conclusions; wet modelling rests on three races.
- **No position telemetry for the 22 new races**: they can't be drawn on a map yet (no outline);
  slice 3 decides how to get track length/outline for simulating them.
- **Gaps are line-crossing gaps on the same lap**: the car ahead may be a car on another strategy
  far away in race terms; lapped/lapping traffic isn't represented. Mini-sector live gaps are not
  available in this data.
- Fuel is only represented by laps remaining (no fuel data); tyre temperatures and setups are
  unknown (plan §6.3).
- The pass counter counts line-crossing order changes between consecutive laps (pit laps
  excluded), so passes and re-passes within one lap, or passes during a lap a car pits, aren't
  counted; the same rule applies to simulated races, so comparisons are fair, not absolute.
- Red-flag periods come from track status code 5; a red flag shown at a time with no status
  sample would be missed.

## M3 slice 4 (E2E)

- The E2E test **needs the fetched Monza data** and a local `.venv`; without the data it skips. It
  isn't in CI (there is no CI yet).
- Verified on Windows only (Chromium); macOS/Linux should work (the helper picks `.venv/bin/python`)
  but haven't been run.
- One flow, one browser; no visual regression checks, no mobile viewport run.
- The Vite proxy prints `ws proxy error` lines when the test kills the backend; that is expected.

## M3 slice 3 (library, save/load/fork UI)

- **No rename or delete** of worlds, timelines or saves in the UI (nor in the API).
- **No timeline comparison** ("Compare Timelines", plan §14.2): branches can be opened one at a
  time, not shown side by side (M6 Counterfactual Lab).
- The fork dialog applies **one** decision; several decisions need several steps (decide, then
  act on the branch).
- Persona values are shown but **can't be edited** ("Edit Driver Profile (fork)", plan §14.2).
- **Historical replay isn't in the new UI** (debug viewer only); the mode badge supports it.
- "Continue" for worlds saved before store v4 falls back to the newest save of any timeline
  (their active timeline was never recorded).
- No "step one lap" (ticks / 10 s / 1 min only); no ← key (no rewind by design).
- The library list isn't refreshed live (reload or reopen it to see changes made elsewhere).

## M3 slice 2 (React app)

- **Cars all start from the same point** (the engine has no grid slots, lap 1 is resolved in grid
  order), so at the start the markers sit on top of each other. Close cars' code labels overlap.
- The **live frame rate wasn't measured** (the browser pane was hidden); smoothing is unit-tested.
- Gaps in the tower are the backend's display estimates; laps of events in the log are derived
  from the leader's lap completions (exact at line crossings, not a timing-loop lap count).
- No pit lane on the map (as in the debug viewer): a pitting car parks on the start/finish line.
- The first browser run lost two laps of events on a live connection without a reproducible
  cause (probably a hidden, throttled page); the client now refetches the history whenever it
  detects a gap, so the display heals, but the root cause is unconfirmed.
- Production use of `npm run build` isn't wired into the backend yet (dev server + proxy only).
- Build tools pull in MPL-2.0 packages (12, dev-only, not in the shipped bundle); the bundle
  itself is React (MIT).

## M3 slice 1 (API)

- The debug viewer is separate and still uses its own `/api/debug/*` routes.
- **One live world per backend**, one user, no authentication; bound to 127.0.0.1. Two browser
  tabs share (and both control) the same live world.
- **Auto-pause on disconnect** also pauses on a page reload or a brief network drop.
- A command that reached the old runner just as another world was made live isn't applied and
  times out (504) instead of being re-routed.
- `step` takes ticks only; "step one lap / sector" (plan §14.2) isn't there yet.
- Stream frames are full state views (~11 KB for 20 cars at ≤10 Hz), not deltas; fine locally.
- The library can't rename or delete worlds; world names are derived (race + seed). Snapshots of
  older engine versions are listed but refused on load (`snapshot_unusable`), as before.
- Replay mode (`HISTORICAL_REPLAY`) isn't in the new API; it's still only in the debug viewer.

## End of M2

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
