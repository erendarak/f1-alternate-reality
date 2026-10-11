# Known limitations

Honest list of what does **not** work or is only approximate. Update whenever one is found or fixed.

## Current (M0, step 3)

- **The race model is a synthetic placeholder.** Its coefficients are uncalibrated, and there are
  no pit stops, strategy decisions, weather, safety cars or DNFs.
- **No car performance yet:** pace comes only from driver personas, so team strength is ignored. In
  a 20-car run, a Sauber can finish 10th from P19.
- **No traffic or overtaking model:** cars pass through each other with no time cost, so grid
  position barely matters.
- Every car finishes all 53 laps; nobody is lapped or flagged at the leader's finish.
- There's no pause/save/load/fork yet (steps 4–6) and no lap data or UI yet.
- 12 of 20 drivers use unmodified generic templates, so several drivers share identical personas
  (see `ASSUMPTIONS.md`).
- The roster comes from a single source (Jolpica). It gets cross-checked against FastF1 in step 7.
- Determinism guarantees will cover **CPU mode only**. BLAS/GPU/parallel bitwise equality is out of scope
  (plan §4.5).
