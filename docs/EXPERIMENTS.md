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
