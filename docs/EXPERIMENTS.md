# Experiments

Every experiment uses the report template from plan §19.3 (blank template first, then entries).

```text
Experiment ID:
Research question:
Hypothesis:
Dataset manifest and exclusion criteria:
Profile revisions and driver IDs:
Ruleset + engine version:
Baseline policy versions:
Model version + architecture:
Training seeds / Evaluation seeds:
Scenarios (weather, track, car, traffic):
Metrics and confidence intervals:
Profile behavior/strategy visualizations:
Failures and negative results:
Reproducibility commands:
Decision / next experiment:
```

---

## EXP-001 — Do data-derived inputs make the simulated Monza 2024 look like the real race? (2026-10-12)

```text
Experiment ID:        EXP-001
Research question:    Does starting each car on its real tyre and giving each team a data-estimated
                      car-pace offset bring the simulated finishing order closer to the real one?
Hypothesis:           Yes: higher rank correlation and lower position error than equal cars/all-MEDIUM.
Dataset:              data/manifests/2024_italian_gp_fastf1.yaml (FastF1 3.8.3); 914 representative
                      laps; TSU excluded from estimates (<10 laps) and from scoring (DNF).
Profile revisions:    seed_2024_monza_drivers.yaml v1 (editorial personas, unchanged)
Ruleset + engine:     lightweight_engine_v0.3.0, placeholder lap model (default_lap_model)
Baseline:             A = synthetic defaults (all MEDIUM, car offsets 0)
Model:                B = real lap-1 compounds + team car offsets (D-021), no learned components
Seeds:                1..30, paired (same seeds for A and B); no training involved
Scenario:             full 20-car grid, dry, no pit stops/traffic/lapping (engine limits)
Metrics (mean ± sd):  Spearman vs official order   A 0.743 ± 0.029   B 0.879 ± 0.016
                      mean |position error|        A 3.27            B 1.99
                      B > A on Spearman in 30/30 paired seeds
                      winner team                  A Red Bull 26/30  B McLaren 30/30
                      real podium teams (FER/MCL/MCL) reproduced  A 0/30  B 0/30
Failures / negatives: The real winner (LEC, Ferrari) won on STRATEGY: a one-stop vs McLaren's
                      two-stops. Our engine has no pit stops, so raw pace decides and McLaren
                      always wins. Lapping isn't modelled (Saubers end ~140 s behind instead of a
                      lap down).
Caveats:              In-sample: offsets are estimated from this same race. Not evidence of
                      predictive skill on other races.
Reproduce:            uv run python scripts/fetch_monza_2024.py
                      uv run python scripts/experiments/exp001_data_inputs.py --seeds 30
Decision / next:      Keep data inputs on by default (--synthetic to opt out). M2: pit stops +
                      tyre choice + lapping, so strategy can win races; then re-run as EXP-002 and
                      test out-of-sample on 2024 Belgian GP.
```

---

## EXP-002 — The full M2 engine vs two real races: Monza 2024 (in-sample) and Spa 2024 (2026-10-12)

```text
Experiment ID:        EXP-002
Research question:    With pit stops + baseline strategy, traffic, incidents and VSC/SC (engine
                      v0.6.0), how close are simulated races to the real ones, and which parts of
                      the model are furthest off?
Hypothesis:           Order and race time close (data inputs); strategy and overtaking not yet
                      (placeholder tyre/traffic coefficients); incident rates plausible.
Dataset:              data/manifests/2024_italian_gp_fastf1.yaml, 2024_belgian_gp_fastf1.yaml
                      (FastF1 3.8.3). Real metrics from laps/results/track_status tables.
Profile revisions:    seed_2024_monza_drivers.yaml v1; seed_2024_belgian_drivers.yaml v1 (the
                      same personas; SAR template-only). Editorial values.
Ruleset + engine:     lightweight_engine_v0.6.0, world schema v7; placeholder lap/tyre/traffic/
                      incident coefficients (D-010, D-022..D-024)
Baseline policy:      baseline_strategy_v1 (cheapest 0/1/2-stop plan under the model)
Model:                no learned components
Inputs per event:     real start compounds; team car offsets, base lap and pit loss estimated from
                      the SAME race's laps (D-021, D-022, D-026) -> those inputs are in-sample at
                      both tracks. What Spa tests out of sample is everything set at Monza only:
                      personas, tyre/traffic/incident coefficients, strategy, engine structure.
Seeds:                1..30 per event; no training
Method:               overtakes counted identically for real and sim: line-crossing order at each
                      lap end (lap 0 = grid), passes between consecutive lap ends, pairs where
                      either car pitted at the end of either lap excluded.
```

| Metric | Monza real | Monza sim mean (p10–p90) | Spa real | Spa sim mean (p10–p90) |
|---|---|---|---|---|
| Spearman vs real order | – | **0.885** (0.858–0.926) | – | **0.905** (0.877–0.928) |
| Mean position error | – | **1.82** (1.44–2.11) | – | **1.86** (1.67–2.21) |
| Winner's race time (min) | 74.68 | 74.83 (74.10–77.48) ✓ | 79.96 | 80.23 (79.25–83.29) ✓ |
| Stops per finisher | 1.58 | **1.00** (1.00–1.00) ✗ | 1.79 | **1.00** (1.00–1.00) ✗ |
| Share of 1-stop finishers | 0.47 | **1.00** ✗ | 0.26 | **1.00** ✗ |
| On-track overtakes | 53 | **10.1** (8–15) ✗ | 39 | **17.5** (13–22) ✗ |
| Retirements | 1 | 0.97 (0–2) ✓ | 1 | 0.87 (0–2) ✓ |
| Safety cars / VSCs | 0 / 0 | 0.27 / 0.20 (0–1) ✓ | 0 / 0 | 0.27 / 0.17 (0–1) ✓ |
| Lapped finishers | 6 | 7.2 (6–10) ✓ | 0 | 0.0 ✓ |
| Real winner wins | LEC | 0/30 (NOR 22, PIA 8) | RUS on road / HAM official | RUS 0/30, **HAM 20/30**, LEC 9 |
| Tyre sequences | M-H-H 7, M-H 7, H-M 2, H-H-M 2 | **M-S 400, H-S 171** | M-H-H 6, M-H-M 3, M-M-H 2, M-M 2 | **M-S 489, H-S 57, S-M 28** |

```text
Findings:
  1. Order and pace: good, and slightly better than EXP-001 at Monza (Spearman 0.879 -> 0.885,
     position error 1.99 -> 1.82). At Spa, 0.905 / 1.86 with Monza-designed coefficients, and
     the official winner HAM wins 20/30 (Mercedes is the fastest car in the Spa data). Caveat:
     car offsets come from each race itself, so this mostly confirms the inputs, not prediction.
  2. Race time within 0.3 min at both tracks (the base lap is estimated from each race).
  3. STRATEGY IS WRONG EVERYWHERE: every simulated finisher one-stops onto SOFT; the real races
     were mostly M->H with many 2-stops (Monza 47 % one-stop, Spa 26 %). The planner is right
     for its model; the placeholder compound offsets/degradation make the soft a race tyre and
     a second stop never pays. This is the single biggest gap.
  4. OVERTAKING IS 2-5x TOO RARE (10 vs 53 at Monza, 17 vs 39 at Spa). Partly a consequence of
     (3): with identical strategies there are few tyre-offset pace differences to pass on;
     partly the editorial traffic parameters (threshold, dirty air, min gap).
  5. Incident/neutralisation rates are consistent with two fully green races with one retirement
     each, but two races are no evidence about rates; season-level data is needed.
  6. LEC's strategy win at Monza still never happens (0/30): it needs (3).
Negative results:     3, 4, 6 above.
Reproduce:            uv run python scripts/fetch_event.py --season 2024 --round 16
                      uv run python scripts/fetch_event.py --season 2024 --round 14
                      uv run python scripts/experiments/exp002_m2_engine.py --seeds 30 \
                          --json docs/experiments/exp002.json
Decision / next:      M4 calibration order: (a) compound offsets + degradation curves (non-linear
                      wear) from stint data, so 2-stops and M->H emerge; (b) overtaking parameters
                      fitted to on-track pass counts with the same counter; (c) incident/SC rates
                      from a season of race-control data. Re-run EXP-002 after each.
```

## EXP-003 — Pace models and tyre/fuel coefficients from the 2024 season (2026-10-13)

```text
Experiment ID:        EXP-003
Research question:    (1) Which tyre/fuel coefficients explain real race laps, and how much
                      better than the engine's editorial placeholder? (2) How well can a lap be
                      predicted from what is known when it starts, and do driver personas help?
Hypothesis:           Fitted coefficients beat the placeholder clearly; a learned model beats a
                      rolling median somewhat; personas add little beyond the driver's own laps.
Dataset:              data/manifests/dataset_laps_2024_v1.yaml (content
                      sha256:e43cd3185e213cee…), representative dry laps (green,
                      accurate, no pit in/out, not lap 1); wet races excluded (slice 4).
Splits:               leave-one-race-out over the 16 dry_train races; final fit on all 16,
                      evaluated on the 5 dry_test races (Japan, Monaco, Monza, Mexico, Abu Dhabi).
Profile revisions:    persona ablation only: Monza/Spa 2024 seed personas (editorial; 1.8 % of
                      laps have no persona: drivers not in those races)
Models:               shape = OLS with a pace term per driver per race (absorbed); LightGBM 4.7
                      (L1 loss, 400 rounds, deterministic, seed 42) on the residual vs the
                      driver's recent median; quantile models (alpha 0.1/0.9); split-conformal
                      intervals from the leave-one-race-out residuals.
Code:                 scripts/experiments/exp003_pace_models.py; f1ar/calibration/pace_models.py,
                      pace_features.py. Reproduced byte-for-byte on a second run.
```

**Task 1 — engine shape.** Mean absolute residual (s/lap) after the shape and a pace term per
driver fitted on the evaluated race (in-sample nuisance, as in the engine); [95 % interval when
races are resampled].

| Shape | Leave-one-race-out (16) | dry_test (5) |
|---|---|---|
| Placeholder (engine v0.6.0, editorial) | 0.579 [0.52–0.65] | 0.801 [0.43–1.18] |
| Fitted, linear wear | 0.520 [0.47–0.57] | 0.714 [0.40–1.05] |
| Fitted, quadratic wear | 0.521 [0.47–0.57] | 0.664 [0.40–0.91] |
| Quadratic + per-race adjustment (in-sample) | 0.474 [0.44–0.51] | 0.626 [0.39–0.88] |

Fitted season shape (all 16 dry_train races) with 95 % intervals from refits on resampled races:

| Coefficient | Value [95 %] | Placeholder |
|---|---|---|
| Fuel + track evolution, s per lap of fuel left | +0.0485 [+0.0401, +0.0603] | +0.035 |
| Following within 1.2 s ("dirty air"), s | +0.426 [+0.319, +0.498] | +0.15 (+ holds) |
| Fresh SOFT vs MEDIUM, s | -0.132 [-0.437, +0.278] | −0.6 |
| Fresh HARD vs MEDIUM, s | +0.070 [-0.148, +0.309] | +0.4 |
| Wear SOFT, s per lap of age | +0.0942 [+0.0200, +0.1449] | +0.09 |
| Wear MEDIUM | +0.0845 [+0.0507, +0.1202] | +0.055 |
| Wear HARD | +0.0609 [+0.0404, +0.0804] | +0.035 |
| Quadratic wear S / M / H, s per age² | -0.0014 / -0.0013 / -0.0004 (intervals include 0) | 0 |

Per-race wear multiplier (in-sample, ridge-shrunk to 1): monaco 0.39, saudi_arabian 0.47, japanese 0.59, qatar 0.60, azerbaijan 0.62, belgian 0.67, miami 0.75, las_vegas 0.81, united_states 0.82, italian 0.88, australian 0.98, mexico_city 0.99, chinese 1.08, abu_dhabi 1.09, dutch 1.10, singapore 1.19, spanish 1.23, bahrain 1.31, emilia_romagna 1.34, hungarian 1.42, austrian 1.48.

> **Correction (M4 slice 3, 2026-10-13):** the per-race prior was scaled wrongly in the first run (it vanished for rarely used compounds; see D-033). This table and the multipliers are the corrected re-run: the per-race row went from 0.455 / 0.618 to 0.474 / 0.626 (the first numbers were flattered by over-fitting rare compounds) and Qatar's multiplier from -0.24 to 0.60; the season shape, the coefficient table and the online task are unchanged.

**Task 2 — online lap prediction** (MAE s/lap [race-bootstrap 95 %], median AE):

| Model | Leave-one-race-out (16) | dry_test (5) | median AE (test) |
|---|---|---|---|
| Rolling median of the driver's last 3 laps | 0.446 [0.42–0.48] | 0.578 [0.38–0.82] | 0.284 |
| Rolling median + shape correction (Baseline A) | 0.415 [0.39–0.45] | 0.549 [0.35–0.79] | 0.283 |
| LightGBM (base + weather) | 0.397 [0.37–0.42] | 0.530 [0.34–0.75] | 0.287 |
| LightGBM without weather | 0.387 [0.36–0.41] | 0.524 [0.33–0.76] | 0.284 |
| LightGBM + driver/team identity | 0.392 [0.37–0.42] | 0.530 [0.33–0.76] | 0.290 |
| LightGBM + editorial persona vector | 0.397 [0.37–0.42] | 0.527 [0.33–0.74] | 0.284 |

Intervals (dry_test): quantile p10–p90 covered 64% of laps (target 80 %;
20% below p10, 15% above p90); split-conformal p10–p90 covered
75% (width 1.16 s). Top features by gain:
field_vs_driver 15%, gap_ahead_s 9%, fuel 8%, track_temp_c 7%, air_temp_c 7%, gap_behind_prev_s 7%.

```text
Findings:
  1. The fitted shape beats the placeholder: -10 % residual leave-one-race-out (0.579 -> 0.52)
     and -17 % on dry_test (0.80 -> 0.66); per-race adjustment adds another -9 % / -6 %
     (0.474 / 0.626, corrected run).
     The race-resampled intervals overlap on the 5 test races (Japan and Monaco dominate).
  2. THE PLACEHOLDER'S SOFT IS THE BUG behind EXP-002's "everyone one-stops onto SOFT": it is
     0.6 s faster than MEDIUM; in real race trim the fresh SOFT is only ~0.13 s faster
     (interval -0.44..+0.28) and wears fastest. Compound offsets are poorly identified from race
     laps (teams choose compounds and stint lengths; drivers manage tyres), so their intervals
     are wide.
  3. Fuel (+ track evolution) 0.048 s per lap of fuel left; wear 0.06-0.09 s per lap of age;
     degradation differs hugely by track (multiplier ~0 at Monaco/Saudi, 1.5-1.7 at Bahrain,
     Imola, Hungary, Austria). One track-independent wear rate can't produce realistic strategy
     everywhere: the engine needs a per-race wear level.
  4. Wear looks slightly concave (quadratic terms negative) but the intervals include 0; a
     negative quadratic would make tyres faster after ~33 laps if extrapolated. (First run:
     Qatar's multiplier came out -0.24; that was the prior-scaling bug, 0.60 when corrected.)
  5. "Dirty air" 0.43 s for following within 1.2 s is a FOLLOWING cost that includes being held
     up, not pure aerodynamic loss; the engine models holds separately, so it can't be used as
     is for the engine's dirty-air parameter (slice 5 calibrates traffic as a whole).
  6. Online: LightGBM is best (-11 % / -8 % vs the rolling median, -4 % / -3 % vs Baseline A); the
     median error barely moves (~0.28 s): gains come from the tails (traffic, stops of others).
  7. PROFILE ABLATION: neither driver/team identity nor the editorial persona vector improves
     prediction (0.53 with and without). The driver's own recent laps already carry driver and
     car pace. This does not validate or invalidate the personas: with 20 drivers they act as an
     identity code, and identity adds nothing here. -> no evidence that a profile-conditioned
     MLP is needed (slice 6 decides).
  8. Weather features don't help in dry races (slightly worse): temperatures change little
     within a dry race and the recent laps already reflect conditions.
  9. Quantile intervals are too narrow on new races (64 % vs 80 %); conformal calibration from
     the leave-one-race-out residuals gets 75 %: test races (Japan, Monaco) are harder than the
     training races.
Negative results:     3 (concavity not established), 5, 7, 8, 9 above.
Reproduce:            uv run python scripts/fetch_season.py --season 2024
                      uv run python scripts/build_lap_dataset.py --season 2024
                      uv run python scripts/experiments/exp003_pace_models.py --json docs/experiments/exp003.json
Decision / next:      Slice 3 uses the fitted season shape with a per-race wear multiplier and
                      offset deltas estimated from each race (in-sample, labelled), with guards:
                      wear never decreasing with age (quadratic clamped at its peak), multiplier
                      >= a small positive floor. The LightGBM model stays an experimental online
                      predictor (registry manifest), not in the engine's lap model.
```

## EXP-004 — The calibrated engine (v0.7.0) on six real races (2026-10-13)

```text
Experiment ID:        EXP-004
Research question:    With the tyre/fuel shape fitted on other races (EXP-003) plus each race's
                      own wear level, do simulated strategies, order and race time match real
                      races better than the M2 engine (EXP-002)?
Hypothesis:           Compound sequences move from "everyone onto SOFT" to M-H; order and race
                      time stay good; stop counts and overtakes stay off (traffic/undercut and
                      overtaking aren't calibrated yet).
Engine:               lightweight_engine_v0.7.0, world schema v8 (D-033); baseline_strategy_v1
Races:                Monza, Spa (in-sample for personas and M2 coefficients), Japan, Monaco,
                      Mexico, Abu Dhabi (dry_test: the tyre shape never saw them). Per-race inputs
                      (start tyres, car offsets, base lap, pit loss, wear multiplier, offset
                      deltas) are estimated from each race itself: in-sample everywhere.
Seeds:                1..30 per race; same harness and counters as EXP-002
Code:                 scripts/experiments/exp002_m2_engine.py --events ... -> docs/experiments/exp004.json
```

Real / simulated mean (✗ = real value outside the simulated p10-p90):

| Race | Stops per finisher | 1-stop share | Passes | Winner time (min) | Spearman | Real winner wins | Real sequences | Simulated sequences (all cars x 30) |
|---|---|---|---|---|---|---|---|---|
| Italian | 1.58 / 1.00 ✗ | 0.47 / 1.00 ✗ | 53 / 10 ✗ | 74.7 / 75.2 | 0.889 | 0/30 | M-H 7, M-H-H 7, H-H-M 2 | M-H 400, H-S 171 |
| Belgian | 1.79 / 1.00 ✗ | 0.26 / 1.00 ✗ | 39 / 18 ✗ | 80.0 / 80.6 | 0.907 | 0/30 | M-H-H 6, M-H-M 3, M-M-H 2 | M-H 489, H-S 57, S-H 28 |
| Japanese | 2.00 / 1.00 ✗ | 0.12 / 1.00 ✗ | 63 / 21 ✗ | 114.4 / 86.2 ✗ | 0.903 | 28/30 | M-M-M-H 3, M-H-H-M 2, S-S-H-H 2 | M-H 345, S-H 225 |
| Monaco | 0.44 / 1.00 ✗ | 0.31 / 1.00 ✗ | 11 / 31 ✗ | 143.3 / 144.3 | 0.870 | 15/30 | M-H 7, H-M 3, H-M-H 3 | H-S 303, M-H 250 |
| Mexico City | 1.24 / 1.00 ✗ | 0.82 / 1.00 ✗ | 68 / 47 ✗ | 100.9 / 97.8 ✗ | 0.872 | 24/30 | M-H 10, H-M 4, H-M-M-S 1 | M-H 390, H-M 169 |
| Abu Dhabi | 1.44 / 1.00 ✗ | 0.69 / 1.00 ✗ | 88 / 34 ✗ | 86.6 / 87.1 | 0.799 | 15/30 | M-H 10, M-H-H 4, H-M 1 | M-S 542, H-M 27 |

Against EXP-002 (engine v0.6.0):

| Race | Simulated sequences | Spearman | Winner time (min) |
|---|---|---|---|
| Italian | M-S 400, H-S 171 -> M-H 400, H-S 171 | 0.885 -> 0.889 | 74.8 -> 75.2 (real 74.7) |
| Belgian | M-S 489, H-S 57, S-M 28 -> M-H 489, H-S 57, S-H 28 | 0.905 -> 0.907 | 80.2 -> 80.6 (real 80.0) |

```text
Findings:
  1. COMPOUND CHOICE FIXED where the data is clear: Monza and Spa now run M-H (EXP-002: M-S for
     every car); Mexico M-H / H-M exactly like the real race; Japan M-H / S-H. The placeholder's
     too-fast SOFT was the cause (EXP-003).
  2. Abu Dhabi plans M-S (real M-H): its in-sample offsets (HARD +0.09 vs MEDIUM, SOFT about
     equal) make a short final SOFT stint look free. Small offset differences decide the
     compound of the final stint; race-trim data can't pin them down well (EXP-003).
  3. STILL ONLY ONE STOP for every car (1.00 vs 0.44-2.0 real). With the fitted race-trim wear
     (about 0.06-0.13 s per lap of age, flattening) a second stop never beats ~20-24 s of pit
     loss on an empty track. Real two-stops also come from undercuts, track position, traffic
     and safety-car timing, which the baseline planner ignores. This is not something to tune
     the tyre model for; it is what M5's strategist must learn (and the planner could weigh
     traffic later).
  4. Order and pace stay good: Spearman 0.80-0.91; real winners
     dominate where they had the fastest car on the day (Japan VER 28/30, Mexico SAI 24/30,
     Monaco LEC 15/30). Race time within ~1 min except Japan (real time includes a ~30 min
     red-flag stoppage; no red flags in the simulation) and Mexico (-3 min).
  5. OVERTAKING IS NOT TRACK-SPECIFIC: too few passes almost everywhere (Monza 10 vs 53, Abu
     Dhabi 34 vs 88), far too many at Monaco (31 vs 11). Slice 5 calibrates passing per track.
  6. Speed 0.58-1.06 s per race (Monaco's 78 laps the slowest); a full replan ~0.5 ms.
Negative results:     2, 3, 5.
Reproduce:            uv run python scripts/fetch_season.py --season 2024 --telemetry --rounds 4 8 20 24
                      uv run python scripts/experiments/exp002_m2_engine.py --seeds 30 --events \
                        2024_italian_gp 2024_belgian_gp 2024_japanese_gp 2024_monaco_gp \
                        2024_mexico_city_gp 2024_abu_dhabi_gp --json docs/experiments/exp004.json
Decision / next:      Keep the calibrated model. Slice 4 (weather), slice 5 (traffic and race
                      control per track). Stop counts are a strategy problem (M5), recorded as a
                      known limitation of the baseline planner.
```

## EXP-005 — Weather: fitting wet-tyre crossovers on the three wet 2024 races (2026-10-13)

```text
Experiment ID:        EXP-005
Research question:    Can track wetness, wetting/drying rates and the slick/intermediate penalty
                      lines be estimated from Canada, Britain and São Paulo 2024, and does the
                      engine then switch tyres when the real field did?
Hypothesis:           Lap times alone can't separate wetness from the penalty slope; the teams'
                      tyre choices pin the crossover. Switch timing should land within a few laps.
Engine:               lightweight_engine_v0.8.0, world schema v9 (D-034); baseline_strategy_v1
Data:                 2024_canadian_gp, 2024_british_gp, 2024_sao_paulo_gp (the wet split, D-031):
                      2521 green laps, 3366 tyre choices; rain flags from FastF1 weather data
Seeds:                1..30 per race
Code:                 scripts/experiments/exp005_weather.py --json docs/experiments/exp005.json
Artifact:             configs/calibration/weather_2024_v1.json (+ configs/models/weather_2024_v1.yaml)
```

Fit on all three races (robust least squares, lap residuals + logistic choice term):

| Parameter | Value |
|---|---|
| Wetting time constant | ~10 min |
| Drying time constant | 5 min (**at its lower bound**) |
| Slick penalty | 14.06 s x w |
| Intermediate penalty | 5.92 + 3.52 x w s |
| Full wet (editorial) | intermediate + 3 s at w = 0, crossing at 0.85 |
| Crossovers slick/inter, inter/wet | 0.562, 0.851 |
| Lap MAE (pace per driver in-sample) | 2.28 s (3.36 s without weather) |
| Mean P(chosen tyre) | 0.627 |

Leave one wet race out (fit on two, test on the third):

| Held out | Lap MAE weather / none (s) | P(chosen tyre) | Crossovers |
|---|---|---|---|
| Britain | 14.70 / 4.45 | 0.628 | 0.085, 0.956 |
| Canada | 2.81 / 4.11 | 0.463 | 1.247, 0.85 |
| São Paulo | 2.58 / 1.35 | 0.189 | 1.926, 0.85 |

Simulation, HISTORICAL weather (real / simulated mean over 30 seeds):

| Race | Majority on wet tyres from lap | Back on slicks lap | Wet-tyre lap share | Full-wet share | Stops per finisher | Winners (real) |
|---|---|---|---|---|---|---|
| British | 27 / 29.0 | 39 / 40.8 | 0.24 / 0.22 | 0.00 / 0.00 | 2.44 / 2.84 | HAM 19, RUS 11 (HAM) |
| Canadian | 1 / 1.0 | 45 / 47.6 | 0.62 / 0.67 | 0.015 / 0.07 | 2.2 / 1.0 | NOR 22, HAM 5, PIA 3 (VER) |
| São Paulo | 1 / 1.0 | never / never | 1.00 / 1.00 | 0.016 / 0.00 | 1.0 / 0.0 | NOR 18, VER 5, RUS 4, HAM 3 (VER) |

```text
Findings:
  1. The lap-time-only fit was degenerate (wetness and slope trade off); adding the choice term
     and bounds gives sensible crossovers on all three races together (0.56 / 0.85).
  2. NOT STABLE OUT OF SAMPLE: leaving a race out moves the crossovers from 0.09 to 1.9 and makes
     held-out Britain worse than no weather model at all (14.7 vs 4.45 s). Three races are too
     few; the all-race fit is used and labelled experimental.
  3. Rain flags can't say how long a track stays wet (Britain's flag stops well before the field
     returned to slicks). HISTORICAL mode therefore reveals wetness from the real field's tyre
     choices. With that, switches land within 2-3 laps of reality and the wet-tyre share within
     0.05. THIS IS PARTLY CIRCULAR: conditions are inferred from the same choices the
     simulation is compared with. It shows the engine and baseline react consistently, not
     that the timing is predicted.
  4. The inference first put an all-intermediate Britain field on full wets (found in the
     browser); it is now capped in the inter band unless most wet runners were on full wets.
  5. Stops: Britain too many (2.84 vs 2.44); Canada and São Paulo too few (the baseline never
     stops for fresh wet tyres, wear on wet tyres is editorial, and the São Paulo red flag's free
     tyre change isn't simulated).
  6. Winners: Britain HAM (real) most often; Canada and São Paulo favour NOR (real VER, 5/30 at
     São Paulo). Wet-weather driver skill is editorial (wet_skill) and VER's São Paulo drive came
     from P17 with a red-flag tyre change; neither is modelled from data.
  7. Wet-race dry-pace inputs: pit loss from wet laps and SC/red-flag stops had been 37-54 s;
     estimating on weather-adjusted laps and green stops only fixed it (also used for dry races).
Negative results:     2, 5, 6.
Reproduce:            uv run python scripts/fetch_season.py --season 2024 --telemetry --rounds 9 12 21
                      uv run python scripts/experiments/exp005_weather.py --json docs/experiments/exp005.json
                      (re-run on the final code: identical numbers)
Decision / next:      Keep the model as experimental, editorial parts labelled. A second season of
                      wet races would be the real fix (M9 data scope). Slice 5: traffic and race
                      control per track.
```

## EXP-006 — Overtaking, the start and race control fitted on the 2024 season (2026-10-13)

```text
Experiment ID:        EXP-006
Research question:    Fitted on real lap-level encounters (not on pass counts), does the engine's
                      overtaking become track-specific and reproduce real pass counts? Do race
                      control rates from the season match real retirements and neutralisations?
Hypothesis:           Totals and the track ranking improve; the split of passes by cause stays off
                      where it depends on strategy (one stop for everyone, EXP-004).
Engine:               lightweight_engine_v0.9.0, world schema v10 (D-035)
Data:                 21 dry 2024 races (16 dry_train, 5 dry_test); 9,405 encounters, 614 passes
Seeds:                1..10 per race and setting
Code:                 scripts/experiments/exp006_traffic_race_control.py -> docs/experiments/exp006.json
Artifacts:            configs/calibration/{traffic,race_control}_2024_v1.json (+ configs/models/*.yaml)
```

Overtaking fit (pass probability by expected advantage, all encounters):

| Expected advantage (s) | Encounters | Observed | Predicted |
|---|---|---|---|
| ≤ -1 | 5656 | 0.004 | 0.006 |
| -1 .. -0.5 | 1874 | 0.030 | 0.018 |
| -0.5 .. 0 | 1138 | 0.137 | 0.142 |
| 0 .. 0.5 | 425 | 0.379 | 0.392 |
| 0.5 .. 1 | 186 | 0.597 | 0.637 |
| 1 .. 1.5 | 78 | 0.833 | 0.772 |
| 1.5 .. 2 | 29 | 0.966 | 0.855 |
| > 2 | 19 | 0.842 | 0.912 |

Min gap 0.47 s, scale 1.04 s, season threshold +0.08 s (training races only: +0.09 s), unexplained
pass floor 0.6 % per encounter. Track terms: odd/even-lap cross-validated log-likelihood
-1291 (almost no track terms) vs -1252 (shrinkage 20-50): tracks differ. Launch spread 1.71 s
(lap-1 passes on the training races: 1.3 at spread 0, 10.6 at 1.5, 17.8 at 2.0; target 13.6).

Simulated vs real, three settings (v0.8 = editorial traffic and race control, no launch spread;
season = fitted, one training-race threshold everywhere; per track = engine default):

| Setting | Pass MAE train / test | Correlation train / test | Mean passes (real 53.6) | Lap 1 / restart / tyre / green (real 15.1 / 3.2 / 20.9 / 14.4) | Retirements (1.95) | SC (0.48) | VSC (0.38) |
|---|---|---|---|---|---|---|---|
| v0.8 | 28.2 / 36.3 | 0.55 / 0.14 | 25.5 | 0.7 / 0.4 / 3.1 / 21.3 | 1.22 | 0.31 | 0.51 |
| season | 12.6 / 26.3 | 0.44 / 0.20 | 44.5 | 13.0 / 1.2 / 3.9 / 26.3 | 2.14 | 0.62 | 0.52 |
| per track | 12.2 / 18.9 | 0.42 / 0.74 | 44.0 | 13.2 / 1.2 / 3.8 / 25.8 | 2.14 | 0.62 | 0.52 |

| Race | Group | Threshold (s) | Real passes (lap 1 / restart / tyre / green) | v0.8 | Season | Per track (lap 1 / restart / tyre / green) |
|---|---|---|---|---|---|---|
| Abu Dhabi | test | +0.00 | 88 (57 / 2 / 20 / 9) | 33 | 54 | 59 (16 / 2 / 3 / 38) |
| Australian | train | +0.10 | 30 (5 / 6 / 11 / 8) | 11 | 34 | 32 (12 / 1 / 4 / 15) |
| Austrian | train | -0.06 | 44 (7 / 1 / 23 / 13) | 10 | 35 | 41 (15 / 1 / 1 / 24) |
| Azerbaijan | train | +0.05 | 57 (7 / 6 / 28 / 16) | 37 | 54 | 48 (12 / 0 / 0 / 35) |
| Bahrain | train | -0.12 | 44 (15 / 0 / 13 / 16) | 14 | 33 | 43 (14 / 1 / 0 / 28) |
| Belgian | train | +0.00 | 39 (10 / 0 / 16 / 13) | 18 | 39 | 45 (14 / 1 / 3 / 27) |
| Chinese | train | -0.10 | 69 (11 / 11 / 21 / 26) | 28 | 48 | 58 (19 / 1 / 1 / 37) |
| Dutch | train | +0.32 | 51 (13 / 0 / 25 / 13) | 33 | 54 | 43 (12 / 2 / 11 / 18) |
| Emilia Romagna | train | +0.35 | 35 (6 / 0 / 27 / 2) | 30 | 46 | 36 (10 / 1 / 13 / 13) |
| Hungarian | train | +0.32 | 57 (22 / 0 / 27 / 8) | 25 | 53 | 36 (9 / 2 / 6 / 20) |
| Italian | test | +0.03 | 53 (13 / 0 / 27 / 13) | 10 | 26 | 28 (12 / 1 / 0 / 16) |
| Japanese | test | -0.05 | 63 (0 / 0 / 27 / 36) | 20 | 34 | 41 (12 / 1 / 1 / 26) |
| Las Vegas | train | -0.19 | 67 (6 / 0 / 41 / 20) | 43 | 58 | 78 (22 / 1 / 5 / 50) |
| Mexico City | test | +0.02 | 68 (25 / 0 / 27 / 16) | 48 | 64 | 67 (12 / 2 / 14 / 40) |
| Miami | train | +0.07 | 45 (12 / 3 / 14 / 16) | 13 | 39 | 34 (13 / 1 / 2 / 18) |
| Monaco | test | +0.46 | 11 (4 / 2 / 5 / 0) | 32 | 48 | 27 (8 / 1 / 1 / 17) |
| Qatar | train | +0.22 | 91 (45 / 29 / 3 / 14) | 28 | 41 | 33 (11 / 1 / 0 / 21) |
| Saudi Arabian | train | +0.18 | 34 (4 / 2 / 9 / 19) | 18 | 43 | 38 (15 / 1 / 6 / 15) |
| Singapore | train | +0.19 | 35 (11 / 0 / 22 / 2) | 27 | 42 | 38 (11 / 1 / 6 / 20) |
| Spanish | train | -0.10 | 63 (14 / 0 / 31 / 18) | 19 | 36 | 44 (16 / 2 / 3 / 24) |
| United States | train | +0.02 | 82 (30 / 6 / 22 / 24) | 37 | 54 | 55 (13 / 1 / 0 / 40) |

Race control, training races -> test races: lap-1 retirements predicted 0.9, observed 9 (Monaco 4,
Japan 2, Mexico 2, Abu Dhabi 1); later retirements predicted 7.0, observed 6.

```text
Findings:
  1. THE LAP-LEVEL MODEL IS WELL CALIBRATED: predicted pass rates follow the observed ones from
     0.4 % to 97 % across the advantage range (largest gap 1.5-2 s: 0.86 vs 0.97, 29 cases).
  2. TOTALS ARE MUCH CLOSER: mean pass error per race 30 -> 14; simulated mean 25 -> 44 (real 54).
     Per-track thresholds help on the held-out test races (correlation 0.74 vs 0.20 with one
     season threshold), not on the training races (0.42 vs 0.44): with one race per track,
     the ranking is partly noise.
  3. THE MIX IS WRONG (negative): too many same-tyre green passes (26 vs 14) and far too few
     fresher-tyre passes (4 vs 21). Every simulated car still one-stops at about the same time
     (EXP-004), so the tyre-offset passes of real races don't exist, and passing at a stable
     pace order carries more of the total. The lap-level fit is right; the situations differ.
     This is the strategy problem M5 addresses, not something to tune here.
  4. Monaco still too many passes (27 vs 11; v0.8: 32): its threshold is shrunk to +0.46 s and
     the launch spread is season-wide (Monaco real lap 1: 4 passes, simulated 8).
  5. LAP 1 matched on the training races by construction (13.7 vs 13.6); test races 11.8 vs 19.8
     (Abu Dhabi's 57 real lap-1 passes include cars dropping back after contact).
  6. Restart passes too few (1.2 vs 3.2): simulated restarts rarely change the order (the restart
     itself isn't calibrated; real counts are small and dominated by Qatar's 29).
  7. Race control: retirements 2.1 vs 2.0 (v0.8: 1.2), SC 0.62 vs 0.48, VSC 0.52 vs 0.38 (both
     a little high; not investigated: SC/VSC per retirement is an average over causes). Rare events with small counts: the training races predict 0.9 lap-1 retirements
     for the test races, 9 happened; the engine uses all 21 races.
  8. Bug found by the M0 gate: fitted negative thresholds passed model_copy and broke loading
     a save; builders now validate, the schema allows negative thresholds, and the fit uses the
     engine's "cars must meet" condition (re-run; numbers above are after the fix).
Negative results:     3, 4, 6; per-track thresholds don't beat one season threshold in-sample.
Reproduce:            uv run python scripts/fetch_season.py --season 2024 --telemetry --refresh \
                        --rounds 1 2 3 5 6 7 10 11 13 14 15 17 18 19 22 23
                      uv run python scripts/experiments/exp006_traffic_race_control.py --seeds 10 \
                        --json docs/experiments/exp006.json
Decision / next:      Keep the per-track fit (engine default) and the fitted race control. The
                      pass mix waits for real strategy variety (M5). Slice 6: M4 wrap-up.
```

## EXP-007 — Ablation: what each calibrated part of the engine adds (2026-10-13)

```text
Experiment ID:        EXP-007
Research question:    Which real-race properties does each fitted part (tyre shape, weather,
                      traffic, race control) bring, against the editorial engine?
Hypothesis:           Each part improves the property it was fitted for; none of them improves
                      the finishing order much (that comes from grid and car pace, estimated
                      from each race in every setting).
Engine:               lightweight_engine_v0.9.0 (D-036)
Method:               ablation by removal: the setup is built from a copy of configs/ without
                      the artifact(s) and falls back to the editorial model (as on a machine
                      without the file); "all editorial" removes all four.
Races:                dry_test (Monza, Japan, Monaco, Mexico, Abu Dhabi) and wet (Canada,
                      Britain, São Paulo); 10 seeds per race and setting
Code:                 scripts/experiments/exp007_ablation.py -> docs/experiments/exp007.json
```

Dry test races (mean of 5 races; errors are |simulated - real|, lower is better):

| Setting | Spearman (order) | Real winner wins | Stops error | Sequences a real finisher used | Passes error | Retirements error |
|---|---|---|---|---|---|---|
| full (v0.9.0) | 0.852 | 0.32 | 0.56 | 0.44 | 20.5 | 1.32 |
| no tyre shape | 0.861 | 0.32 | 0.88 | 0.00 | 20.5 | 1.32 |
| no traffic fit | 0.862 | 0.58 | 0.56 | 0.44 | 37.1 | 1.32 |
| no race control fit | 0.858 | 0.30 | 0.56 | 0.44 | 19.6 | 1.86 |
| all editorial | 0.875 | 0.52 | 0.87 | 0.00 | 37.2 | 1.86 |

Wet races (mean of 3):

| Setting | Spearman | Real winner wins | Stops error | Sequences a real finisher used | Passes error | Retirements error |
|---|---|---|---|---|---|---|
| full | 0.702 | 0.43 | 0.89 | 0.39 | 16.1 | 1.83 |
| no tyre shape | 0.728 | 0.33 | 0.82 | 0.54 | 16.4 | 1.87 |
| no weather | 0.705 | 0.33 | 1.22 | 0.00 | 28.1 | 1.43 |
| no traffic fit | 0.727 | 0.30 | 0.89 | 0.39 | 28.3 | 1.83 |
| no race control fit | 0.701 | 0.40 | 0.90 | 0.33 | 9.9 | 1.53 |
| all editorial | 0.693 | 0.23 | 1.29 | 0.00 | 25.5 | 1.63 |

```text
Findings:
  1. EACH PART DOES ITS OWN JOB: the tyre shape is what makes compound choices real (sequences a
     real finisher used 0.44 -> 0 without it; stops error 0.56 -> 0.88); the traffic fit halves
     the pass error (37 -> 20); the race-control fit cuts the retirement error (1.86 -> 1.32) on
     the dry test races; weather is what puts cars on wet tyres at all (0.39 -> 0) and halves the
     wet pass error.
  2. NO PART IMPROVES THE FINISHING ORDER (negative): all editorial 0.875 vs full 0.852 on the dry
     test races. The order comes from grid and car pace (estimated from each race in every
     setting); the fitted parts add realistic variation (lap-1 shuffle, passing, incidents)
     that costs a little order agreement.
  3. THE REAL WINNER WINS LESS with the traffic fit (0.32 vs 0.58; Mexico 0.4 vs 0.9): the
     launch spread and easier passing make races less predictable. Real races are not
     deterministic either; one real result can't say which is right.
  4. Wet races without the fitted race control get a lower pass error (9.9 vs 16.1) and
     retirement error (1.53 vs 1.83): the season rates come from dry races only and the wet
     incident factor (x(1 + 2w)) is editorial. Not tuned here (3 races).
  5. Winner time error is dominated by red flags the engine doesn't simulate (Japan ~27 min,
     Monaco ~39 min, São Paulo ~28 min); Monza 1.9 min, Mexico 2.6, Abu Dhabi 1.5.
  6. Britain: removing the tyre shape raises its sequence overlap (0.27 -> 0.72): the editorial
     offsets happen to pick the final slick compound more real finishers used; one race, no
     conclusion.
Negative results:     2, 4.
Reproduce:            uv run python scripts/experiments/exp007_ablation.py --seeds 10 \
                        --json docs/experiments/exp007.json
Decision / next:      Keep all four parts. Order agreement is not what they are for; M5's
                      strategist and better strategy variety are the next lever.
```

## EXP-008 — The baselines a learned strategist must beat (2026-10-13)

```text
Experiment ID:        EXP-008
Research question:    How good are the rule planner (B1), the real 2024 strategies (B0), an
                      enumerator of 0-3-stop plans (B2) and a fork-based MPC (B3) in our engine,
                      on the same races? What is the bar for the learned AI (slice 3)?
Hypothesis:           MPC (the engine plus look-ahead) beats the rules; the enumerator matches or
                      beats them; the real strategies are worse (chosen for a different race).
Engine:               lightweight_engine_v0.9.0; training env D-038; baselines D-039
Cases:                5 dry test races (Monza, Japan, Monaco, Mexico, Abu Dhabi) x 4 cars (grid
                      P1, P5, P10, P15) x seeds 1-5 = 100; every method on every case (paired).
                      One AI car per race; the other 19 follow the rules. Scoring: points under
                      the driver's risk utility (no championship context).
Code:                 scripts/experiments/exp008_baselines.py -> docs/experiments/exp008.json
Compute:              2 h 22 min on 14 processes (MPC ~1,080 s per race under load, ~557
                      imagined races and 46 deliberations per race)
```

| Method | Score | Points | Position | Stops (cases) | Score vs B1 [95 % CI] | Better / worse than B1 |
|---|---|---|---|---|---|---|
| B1 rules | 0.388 | 6.83 | 9.28 | 1: 94, 0: 6 (retired) | | |
| B0 real 2024 stops | 0.174 | 3.46 | 13.33 | 0: 25, 1: 43, 2: 32 | -0.214 [-0.300, -0.139] | 2 / 68 |
| B2 enumerator | 0.384 | 6.64 | 9.59 | 1: 93, 2: 1 | -0.004 [-0.032, +0.026] | 2 / 10 |
| B3 MPC | 0.351 | 6.28 | 9.68 | 1: 69, 2: 19, 3: 6, 4: 1 | **-0.037 [-0.070, -0.008]** | 11 / 32 |
| random | 0.002 | 0.00 | 17.90 | 32-54 | -0.386 [-0.496, -0.286] | 1 / 92 |

Score per race (B1 / B2 / B3): Monza 0.354 / 0.354 / 0.339, Japan 0.475 / 0.475 / 0.434,
Monaco 0.455 / 0.489 / 0.379, Mexico 0.416 / 0.404 / 0.409, Abu Dhabi 0.241 / 0.198 / 0.194.

```text
Findings:
  1. THE RULE PLANNER IS THE BAR. The enumerator ties it (its 0/1/2-stop optimum is B1's; 3-stop
     plans never won; re-ranking by 3 imagined races changed little: 2 better, 10 worse).
  2. MPC IS WORSE THAN THE RULES (negative, significant): -0.037 [-0.070, -0.008], 11 cases
     better, 32 worse. It stops more (2-4 stops in 26 cases) and prefers a final SOFT stint
     (M-S 31 vs rules M-H 41). Diagnosis: with 3 imagined futures each option's estimate is
     noisy (incidents, safety cars, traffic), and taking the best of 4-5 noisy estimates
     systematically favours an option that got lucky; staying out only wins ties. A one-step
     look-ahead over the rules should be at least as good *in expectation* (rollout
     improvement), so the loss is estimation noise, not the idea. Fixing it needs many more
     futures (cost: already ~18 min per race under load) or a margin before leaving the
     default. Not pursued: MPC is a yardstick, not the target.
  3. Real 2024 strategies are far worse in our engine (-0.214): they were chosen for a different
     race; 25 of 100 have no stop outside a red flag (Monaco, Japan changed tyres under the red
     flag, which the simulation doesn't have) and are disqualified under the two-compound rule.
     A reference, never a fair opponent.
  4. Retirements are identical across methods (7 cases, all methods): incidents come from the
     same keyed draws, so the comparison is paired as designed.
  5. Random policy: 32-54 stops, P17.9 on average: the env punishes nonsense.
Negative results:     2 (MPC), 1 (no gain from 3-stop plans or simulated re-ranking).
Reproduce:            uv run python scripts/experiments/exp008_baselines.py --seeds 5 --workers 14 \
                        --json docs/experiments/exp008.json
Decision / next:      Slice 3's learned policy is judged against B1 (paired, held-out tracks and
                      seeds); B3's gap is reported as is. The training budget can't be "beat MPC".
```

## EXP-009 — The first trained strategist vs the rule planner on held-out races (2026-10-15)

```text
Experiment ID:        EXP-009
Research question:    Does a policy trained with MaskablePPO (one learning car vs the rule field,
                      16 dry_train races) make better pit/tyre/pace decisions than the rule
                      planner (B1) on the 5 held-out dry races and new seeds?
Hypothesis:           Starting from the rules and trained with a paired reward, it at least matches
                      B1 and gains a little (safety-car stops, pace), held-out included.
Engine / rules:       lightweight_engine_v0.9.0, fia_2024_v1, rl_obs_v1 (58 features), D-040
Policy:               strategist_m5s3_v1 (configs/policies/, sha256:ccedb8a9...), MLP 58-128-128-18
                      tanh; behaviour cloning of the rules teacher (1,500 races) -> softened
                      (T = 2.6) -> PPO with paired reward, 602k decisions in 84 min (stopped by
                      hand: flat checks since 300k, throughput down to ~72/s); best checkpoint
                      on training cases = step 501,760. Training seed 0.
Cases (held out):     Monza, Japan, Monaco, Mexico, Abu Dhabi x grid P1/P5/P10/P15;
                      set A = EXP-008's cases (seeds 1-5, 100), set B = new seeds 6-15 (200).
                      One AI car, 19 rule cars, points objective under the driver's risk utility.
Code:                 scripts/experiments/exp009_learned_vs_rules.py -> docs/experiments/exp009.json
                      (+ exp009_diagnostics.json); 4 min on 14 processes
```

Pilots before the main run (part 1, training cases only; all 14 workers, one learning car):

| Pilot | Setup | Decisions | Result (paired check on 24 training cases, argmax) |
|---|---|---|---|
| A | from scratch, plain reward | 100k | no learning: 36 -> 40 stops/race, explained variance ~ -5; -0.45 (0/24 better) |
| B | + reward normalisation | 100k | value fits (EV +0.1), policy flat; -0.45 |
| C | + stay-out prior 0.97 | 300k | learns (sampled: P17.1 -> P13.0, 0.9 -> 3.2 pts, DSQ 31 -> 8 %, PUSH 32 -> 68 %); argmax never stops: -0.45 |
| D | rules warm start, plain cloning | 0 | cloning learned "never stop" (stop recall 0); -0.45 |
| D2 | weighted cloning (x5, 150 ep), PPO ent 0.001 | 100k | starts = B1 (+0.002); PPO frozen (entropy 0.02): +0.023 |
| D3 | same, PPO ent 0.01 | 50k | still frozen: +0.010 |
| D4 | + softened clone (0.25 nats), raw reward | 150k | learns back from the exploration noise; check drifts +0.002 / +0.024 / -0.027 |
| main | + paired reward | 602k | +0.024, +0.043, +0.044, +0.049, **+0.052 (500k, chosen)**, +0.045 |

Held-out results:

| Set | Method | Score | Points | Position | Stops | Most common sequences |
|---|---|---|---|---|---|---|
| A (100) | B1 rules | 0.388 | 6.83 | 9.28 | 1: 94, 0: 6 | M-H 41, H-S 19, M-S 19 |
| A (100) | learned | 0.361 | 6.52 | 9.37 | 1: 94, 0: 6 | **M-S 53**, H-M 19, S-H 10 |
| B (200) | B1 rules | 0.416 | 7.26 | 9.41 | 1: 186, 0: 14 | M-H 82, M-S 39, H-S 38 |
| B (200) | learned | 0.387 | 6.82 | 9.62 | 1: 186, 0: 14 | **M-S 101**, H-M 35, M-H 20 |

| Paired difference (score) | Set A | Set B |
|---|---|---|
| learned - B1 | -0.028 [-0.061, +0.001], 11 better / 15 worse | **-0.029 [-0.053, -0.009]**, 15 / 33 |
| learned (pace forced BALANCED) - B1 | -0.030 [-0.064, -0.001] | -0.028 [-0.052, -0.008] |
| learned - B2 enumerator (EXP-008 rows) | -0.023 [-0.068, +0.017] | |
| learned - B3 MPC (EXP-008 rows) | +0.010 [-0.029, +0.047], 27 / 10 | |
| learned - its own clone before PPO | +0.035 [-0.004, +0.079], 26 / 12 | +0.010 [-0.010, +0.032], 32 / 28 |

Per race (set B, B1 / learned): Monza 0.457 / 0.442, Japan 0.466 / 0.472, **Monaco 0.400 / 0.306**,
Mexico 0.383 / 0.376, Abu Dhabi 0.371 / 0.339. B1's EXP-008 rows were recomputed identically.
Held-out curve over training (set B, vs B1): clone -0.039, 100k -0.029, 300k -0.036, 500k -0.029,
602k -0.050.

```text
Findings:
  1. NEGATIVE: THE LEARNED POLICY DOES NOT BEAT THE RULES ON HELD-OUT TRACKS. It is slightly
     worse: -0.029 on new seeds (CI excludes 0), -0.028 on EXP-008's cases (CI touches 0),
     about -0.4 points per race. On the training tracks it was +0.05 (24 cases): the gain doesn't
     transfer.
  2. It learned no stop-count or pace behaviour of its own: stops are identical to B1 in every
     held-out case (one stop; zero = retired), no disqualification, PUSH/CONSERVE never used
     (forcing BALANCED changes nothing). The whole difference is compound choice: it finishes
     on SOFT (M-S) where B1 runs M-H; Monaco is the big loss.
  3. DIAGNOSIS: the observation can't tell which compound suits an unseen track. The rule planner
     reads each track's per-compound pace offsets and wear rates; rl_obs_v1 has only the medium
     wear rate. The cloned policy (before PPO) already fails the same way (-0.039, Monaco
     0.400 -> 0.266): imitation on 16 tracks memorises them instead of learning the rule.
     PPO improved on the clone a little (+0.010 / +0.035, not significant).
  4. Training from scratch can't get started in this env (flat reward for near-random play);
     a stay-out prior learns but slowly; the rules warm start needs stop-weighted cloning and
     softening; the paired reward makes PPO progress near B1 (pilots, D-040).
  5. Learned vs MPC +0.010 (n.s.): the policy is about as good as B3 and below B1/B2.
Negative results:     1 (the main question), plus pilots A, B, D, D2, D3.
Reproduce:            uv run python scripts/train_policy.py --name main_v1 --steps 800000 \
                        --warm-start 1500 --bc-entropy 0.25 --lr 3e-4 --ent-coef 0.005 \
                        --reward-baseline rules --eval-every 100000 --register strategist_m5s3_v1
                      (stopped by hand at 602k; best = step 501,760)
                      uv run python scripts/experiments/exp009_learned_vs_rules.py \
                        --policy strategist_m5s3_v1 --workers 14 --json docs/experiments/exp009.json
Decision / next:      Proposed (owner's call): observation rl_obs_v2 with each compound's pace
                      offset and wear rate on this track (+ tyre wear in seconds), retrain the same
                      way and re-run EXP-009, before slice 4. M5's done-check item "beats B1 on
                      held-out tracks" is NOT met yet.
```

## EXP-010 — The tyre-life world (W1) against the 2024 races (2026-10-15)

```text
Experiment ID:        EXP-010
Research question:    Does the W1 tyre model (life, management vs push, cliff, fresh-tyre peak,
                      a planner that manages and needs 6.5 s per stop) make the rule field race
                      like 2024: stops, strategy sequences, passes, without breaking pace/order?
Hypothesis:           Stops per finisher move from 1.00 everywhere to the real per-track pattern
                      (1.06 Baku ... 2.1 Bahrain/Spain/Hungary ... 3.4 Qatar), on held-out races too.
Data:                 laps_2024_v1 (21 dry races: 16 dry_train, 5 dry_test)
Engine:               lightweight_engine_v0.10.0 (life) vs the same engine without the tyre-life
                      file (the M4 tyre model = v0.9.0 behaviour), identical seeds 1-10 per race
Fit:                  globals kappa 1.2, delta 0.0 by grid search on dry_train only
                      (docs/experiments/fit_tyre_life.json; kappa 0.8-1.45, delta 0-0.5);
                      per-race life/wear inputs are in-sample (like car pace offsets)
Code:                 scripts/fit_tyre_life.py; scripts/experiments/exp010_tyre_life.py
                      -> docs/experiments/exp010.json (420 races, 2 min on 14 processes)
```

| | Real | W1 (life) | v0.9 |
|---|---|---|---|
| Stops per finisher, train (16) | 1.77 | **1.49** | 1.12 |
| Error per race, train (MAE) / correlation | | **0.34 / 0.40** | 0.66 / 0.25 |
| Stops per finisher, test (5, not fitted) | 1.34 | **1.28** | 1.00 |
| Error per race, test (MAE) / correlation | | **0.28 / 0.75** | 0.56 / n.a. (all 1.0) |
| Passes per race, train / test | 52.7 / 56.6 | **47.5 / 50.1** | 43.9 / 44.3 |
| Retirements per race, train / test | 1.62 / 3.0 | 2.38 / 2.44 | 2.12 / 2.20 |
| Finishing order vs real (Spearman), train / test | | 0.631 / 0.599 | 0.653 / 0.618 |
| Winner time error (min), train / test | | +0.49 / -12.9 | +0.33 / -12.95 |

Per race, stops per finisher (real / W1 / v0.9) and the most common sequences:

| Race | Real | W1 | v0.9 | Real sequences | W1 sequences |
|---|---|---|---|---|---|
| Saudi Arabia | 1.06 | 1.00 | 1.00 | M-H, M-S | M-H, S-H |
| Azerbaijan | 1.06 | 1.02 | 1.00 | M-H, H-M | M-H, H-M |
| Singapore | 1.11 | 1.02 | 1.00 | M-H, H-M | M-H, H-M |
| Imola | 1.21 | 1.09 | 1.00 | M-H, H-M | M-H, H-M |
| USA | 1.21 | 1.00 | 1.00 | M-H, H-M, M-M-H | M-H, H-M |
| Zandvoort | 1.30 | 1.56 | 1.00 | M-H, M-H-S | M-H, M-H-H |
| Miami | 1.42 | 1.06 | 1.00 | M-H, M-H-M | M-H, H-M |
| Belgium | 1.79 | 1.87 | 1.00 | M-H-H, M-H-M | M-H-H, M-H |
| Hungary | 2.05 | 2.02 | 1.95 | M-H-M, M-H-H | M-H-H, S-H-H |
| Australia | 2.06 | 2.00 | 1.00 | M-H-H, S-H-H, H-M-H | M-H-H, S-H-H, H-M-H |
| Las Vegas | 2.06 | **1.05** | 1.00 | M-H-H, H-M-H | M-H, H-M |
| Spain | 2.10 | 1.95 | 1.00 | S-M-H, S-M-S | S-H-H, M-H |
| China | 2.12 | 2.21 | 1.00 | M-H-H, M-H | M-H-H, M-H-H-H |
| Bahrain | 2.15 | 2.00 | 1.00 | S-H-H, S-H-S | S-H-H |
| Austria | 2.30 | 2.00 | 1.95 | M-H-M, M-H-H | M-H-H, H-M-H |
| Qatar | 3.40 | **1.00** | 1.00 | M-H-H-H | M-H, H-M |
| *test:* Monaco | 0.44* | 1.00 | 1.00 | M-H, H-M | H-M, M-H |
| *test:* Mexico | 1.24 | 1.02 | 1.00 | M-H, H-M | M-H, H-M |
| *test:* Abu Dhabi | 1.44 | 1.00 | 1.00 | M-H, M-H-H | M-H, H-M |
| *test:* Monza | 1.58 | 1.39 | 1.00 | M-H, M-H-H (47 % / 47 %) | M-H, M-H-H (61 % / 39 %) |
| *test:* Japan | 2.00 | 2.00 | 1.00 | M-M-H, S-S-H-H | M-H-H, S-H-H |

\* Monaco and Japan: tyres changed under a lap-1 red flag (not simulated).

Same-lap regression on race laps (all 21 races; simulated: 3 seeds per race):

| | Soft vs medium | Hard vs medium | Wear S / M / H (s per lap of age) | Fresh-tyre peak |
|---|---|---|---|---|
| Real | -0.28 s | -0.03 s | 0.067 / 0.038 / 0.037 | -0.13 s |
| W1 simulated | +0.01 s | -0.14 s | 0.131 / 0.063 / 0.057 | -0.07 s |

```text
Findings:
  1. STRATEGY NOW DEPENDS ON THE TRACK, AS IN 2024. Stops per finisher error halves (train
     0.66 -> 0.34, test 0.56 -> 0.28) and follows the real races (test correlation 0.75; v0.9
     had none: always one stop). Several races reproduce the real sequences exactly (Australia,
     Belgium, Bahrain, Monza's one/two-stop split).
  2. MISSES: Las Vegas (real 2.06, W1 1.05: graining/cold track not modelled), Qatar (3.40 vs
     1.00: extreme wear and kerb damage beyond what stint lengths show), Miami/Abu Dhabi/USA
     (real some two-stops, W1 one); stop counts overall a little low (1.49 vs 1.77 train).
  3. LESS VARIETY INSIDE A RACE than real: every rule car runs the same planner, so most of a
     race's field picks the same plan (Bahrain 100 % two-stop vs real 85 %). Variety is the job
     of characters / the AI field (M5 slices 4-5).
  4. SIDE EFFECTS: cars push more (PUSH 63 % of laps in the fit), so retirements rose (2.40 vs
     2.14 per race; real 1.95 over the 21 races): PUSH's editorial x1.6 risk. Order vs real
     slightly lower (Spearman -0.02). Winner times unchanged (Japan -13 min is the real race's
     red-flag stoppage, as before).
  5. RACE-LAP PACE BY COMPOUND IS NOT REPRODUCED: real softs are 0.28 s faster than mediums on
     the same lap, simulated ones aren't (they manage their short life), and simulated wear on
     race laps is ~2x the real one (pushing at x1.4). The behaviour fit preferred delta = 0;
     the pace-mode trade-offs are editorial. A joint fit of stops, shares AND same-lap pace is
     the next improvement.
Negative results:     Las Vegas, Qatar; same-lap compound pace; retirements up.
Reproduce:            uv run python scripts/fit_tyre_life.py --report docs/experiments/fit_tyre_life.json
                      uv run python scripts/experiments/exp010_tyre_life.py --seeds 10 \
                        --json docs/experiments/exp010.json
Decision / next:      Keep W1 (engine v0.10.0). Next in the research plan: W4 (race control and
                      random events; refit retirements with the new pace-mode mix) or a W1b
                      joint fit; the AI comes back after the world is closer to 2024.
```

## EXP-011 — Race control and random events (W4) against the 2024 races (2026-10-15)

```text
Experiment ID:        EXP-011
Research question:    Does W4 (debris SC/VSC, red flags, pit crews and reliability per team,
                      penalties, risk renormalised for W1's pace mix) bring retirements,
                      neutralisations, red flags and penalties to 2024's level without breaking
                      W1 (stops) or the finishing order?
Data / fit:           race_events_2024_v1 (all 24 races' timing; risk_scale, red flags per crash
                      and the penalty scale fitted on the simulated dry_train field)
Engine:               v0.11.0 (W4) vs the same engine without the race-events file (v0.10),
                      identical per-race seeds (10 per race), 21 dry races
Code:                 scripts/fit_race_events.py; scripts/experiments/exp011_race_events.py
                      -> docs/experiments/exp011.json (420 races, 1.2 min)
```

| Per race (21 dry races) | Real | W4 | v0.10 |
|---|---|---|---|
| Retirements | 1.95 | 1.49 | 2.24 |
| Safety cars / VSCs | 0.43 / 0.38 | 0.51 / 0.37 | 0.51 / 0.46 |
| Races with an SC / with SC or VSC | 29 % / 48 % | 41 % / 62 % | 41 % / 64 % |
| Red flags | 0.095 | 0.12 | 0 |
| Time penalties | 2.10 | 2.03 | 0 |
| Passes | 53.6 | 52.9 | 53.4 |
| Stops per finisher (error per race) | 1.67 | 1.40 (0.34) | 1.44 (0.32) |
| Finishing order vs real (Spearman) | | **0.675** | 0.616 |

Train (16) / test (5): retirements real 1.63 / 3.0, W4 1.51 / 1.44; penalties real 1.88 / 2.8,
W4 1.89 / 2.46; order 0.679 / 0.662 (v0.10: 0.620 / 0.603).

```text
Findings:
  1. The race-event rates now match 2024: red flags 0.12 vs 0.095, penalties 2.0 vs 2.1, VSCs
     0.37 vs 0.38, SCs 0.51 vs 0.43; retirements back down from W1's 2.24 (train 1.51 vs 1.63).
  2. The finishing order follows the real races more closely (Spearman 0.616 -> 0.675): fewer
     random retirements of fast cars, team reliability.
  3. The test races had more retirements in reality (3.0: Monaco's and Japan's lap-1 crashes)
     than simulated (1.44); single races are noisy and lap-1 pile-ups aren't modelled as such.
  4. Neutralisations are spread over too many races (an SC in 41 % of simulated races vs 29 %):
     real SCs cluster (Qatar 3, China 2); the simulation draws them more independently.
  5. Stops per finisher unchanged within noise (free tyres under red flags lower them slightly).
  6. Method: shared seeds across tracks repeat the same luck (keyed draws without the event);
     per-race seeds fixed the fit and change some earlier numbers (passes, D-042).
Negative results:     SC clustering; lap-1 pile-ups; real red-flag stoppage time not simulated.
Reproduce:            uv run python scripts/fit_race_events.py --seeds 8
                      uv run python scripts/experiments/exp011_race_events.py --seeds 10 \
                        --json docs/experiments/exp011.json
Decision / next:      Keep W4 (engine v0.11.0). Next per the plan: W2 (race pace realism: lap
                      noise, driver race pace vs teammate, development).
```

## BENCH-001 — Batch headless throughput (2026-10-12)

```text
Command:   uv run python scripts/benchmark.py --races 10 --parallel 64
Machine:   Intel 12th-gen laptop CPU (Family 6 Model 154), 16 logical CPUs, Python 3.12.3;
           the machine was under other load (timings varied 0.6-1.6 s/race across the day)
Workload:  full 20-car Monza race (53 laps, ~9,100 ticks) with every M2 feature, data inputs
Result:    sequential 1.50 s/race (min 0.92), ~6,100 ticks/s, ~2,400 races/hour on one core;
           64 races on 16 processes in 38 s -> ~6,000 races/hour (worker start-up included)
Profile:   ~55 % of a race is pydantic model_copy (one CarState copy per car per tick, ~188k per
           race); cost grows with CarState's field count.
Verdict:   ~145,000 races/day >> the plan's 100 races/day floor for M5. No optimisation needed
           now; if RL needs more, a lap-level fast path or a mutable internal car record behind
           the same API (SIMULATOR_DECISION follow-up 3).
```

## BENCH-002 — Training environment throughput (2026-10-13)

```text
Command:   uv run python scripts/env_benchmark.py --races 5 (Monza and Monaco, random legal actions)
Machine:   same laptop as BENCH-001 (16 logical CPUs); a backend server was running alongside
Result:    one AI car vs the rule field: 0.71 s/race, 81 decisions/s, ~5,000 races/hour per core
           all 20 cars AI: 1.12 s/race, ~1,200 decisions/s, ~3,200 races/hour per core
Verdict:   1 M training decisions = ~3.4 core-hours with one AI car (~15 min on 14 workers),
           ~14 core-minutes with a whole AI field. The simulation, not the GPU, will set the
           training speed; the 20-car setting gives 15x more experience per simulated race.
```
