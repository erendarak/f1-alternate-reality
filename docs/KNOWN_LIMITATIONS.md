# Known limitations

Honest list of what does **not** work or is only approximate. Update whenever one is found or fixed.

## Current (M0, step 1)

- There's no simulation, data, profiles or UI yet. Only the repo tooling and the canonical state codec exist.
- Determinism guarantees will cover **CPU mode only**. BLAS/GPU/parallel bitwise equality is out of scope
  (plan §4.5).
