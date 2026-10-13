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
| Quadratic + per-race adjustment (in-sample) | 0.455 [0.42–0.49] | 0.618 [0.39–0.87] |

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

Per-race wear multiplier (in-sample, ridge-shrunk to 1): qatar -0.24, monaco 0.03, saudi_arabian 0.07, japanese 0.28, azerbaijan 0.37, belgian 0.51, miami 0.56, united_states 0.66, las_vegas 0.73, italian 0.83, australian 0.96, mexico_city 1.00, abu_dhabi 1.14, dutch 1.16, chinese 1.22, singapore 1.31, spanish 1.37, bahrain 1.50, emilia_romagna 1.51, hungarian 1.63, austrian 1.73.

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
     and -17 % on dry_test (0.80 -> 0.66); per-race adjustment adds another -10 % / -7 %.
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
     negative quadratic would make tyres faster after ~33 laps if extrapolated. Qatar's fitted
     multiplier is negative (-0.24): not physical (traffic/track evolution not separated).
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
