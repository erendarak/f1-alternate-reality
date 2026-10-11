# Known limitations

Honest list of what does **not** work or is only approximate. Update whenever one is found or fixed.

## Current (M0, step 5)

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
- **No car performance yet:** pace comes only from driver personas, so team strength is ignored. In
  a 20-car run, a Sauber can finish 10th from P19.
- **No traffic or overtaking model:** cars pass through each other with no time cost, so grid
  position barely matters.
- Every car finishes all 53 laps; nobody is lapped or flagged at the leader's finish.
- There's no fork yet (step 6) and no lap data or UI yet.
- 12 of 20 drivers use unmodified generic templates, so several drivers share identical personas
  (see `ASSUMPTIONS.md`).
- The roster comes from a single source (Jolpica). It gets cross-checked against FastF1 in step 7.
- Determinism guarantees will cover **CPU mode only**. BLAS/GPU/parallel bitwise equality is out of scope
  (plan §4.5).
