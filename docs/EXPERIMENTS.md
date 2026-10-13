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
