# Simulator decision memo (M0 step 8)

**Date:** 2026-10-12 · **Decision:** **Option C, our own lightweight engine (GO).** TUM race-simulation
as the engine: **NO-GO**. TUM time-discrete simulator: **NO-GO**. Plan §7.2 option C was pre-approved
(D-003); this memo records the evidence.

## Question

Plan §7.2 asks whether an existing open-source race simulator can be wrapped in a **pausable,
step-based, serialisable, forkable** adapter (§7.3 `RaceEngine` port) for 2024 Monza, judged in this
order: (1) pause/save/load/fork fit, (2) reproducibility, (3) 2024 calibration, (4) performance,
(5) licence, (6) maintenance.

## What was tested

- **A. [TUMFTM/race-simulation](https://github.com/TUMFTM/race-simulation)** (lap-discrete Python,
  LGPL-3.0). Cloned into an isolated sandbox outside this repo and run in its own venv (Python 3.12,
  numpy 2.5, pandas 3, scipy 1.18). The probe is `scripts/research/tum_feasibility_probe.py`, which
  documents how to reproduce it. No TUM code is in this repository.
- **B. [heilmeiera/time-discrete-race-simulator](https://github.com/heilmeiera/time-discrete-race-simulator)**
  (Rust, Apache-2.0/MIT). Assessed from its repository and README, not built (time-box).
- **C. Our engine** (`f1ar/simulation/lightweight_engine.py`, steps 3–6), already passing the M0
  persistence gates.

## Evidence: option A (TUM race-simulation)

| # | Probe result (Spielberg 2019, TUM's default example) | Meaning |
|---|---|---|
| 1 | Imports on Python 3.12 **only after stubbing TensorFlow and cvxpy**: `race.py` imports them at module load even when unused. The pinned stack (numpy 1.18, pandas 1.0, TF 2.2) is Python-3.8 era. | Runs on modern Python with workarounds; the dependency hygiene is fragile. |
| 2 | Single race: **54 ms** deterministic; **116 ms** stochastic (≈ 8.6 races/s). | Fast. A lap-level engine. |
| 3 | `random.seed` alone is **not repeatable**: pit standstill times use `scipy.stats.fisk.rvs` without `random_state`, i.e. NumPy's global RNG. With both `random.seed` and `np.random.seed` it is. | Randomness lives in **two interpreter-global generators**. |
| 4 | Stepping lap by lap via the **private** `_Race__simulate_lap()`, then `deepcopy` plus capturing both global RNG states, resumes **exactly**. | In-memory pause/fork is possible, at private-API level. |
| 5 | One extra NumPy draw in a branch (e.g. one more pit stop) **changes every later random outcome for every car**. | **No common random numbers across forks.** Counterfactual comparisons mix decision effects with dice effects (plan §4.5). |
| 6 | State = `Race` (22 slots) + 20 × (`Driver` 15, `Car` 36, `Tireset` 5) + `Track` 19, all name-mangled private; the repo persists via **pickle only**. | Disk save/load needs a hand-written serialiser for ~100 private fields per car stack, or pickle, which the plan forbids loading. |
| 7 | Safety cars, VSCs and retirements are **pre-drawn for the whole race at construction**. | The future sits inside the state: a lookahead hazard for policies (plan §9.4). |
| 8 | Strategy is a **pre-planned list per driver** (`[[0,'A4',0,0.0],[46,'A6',0,0.0]]`). Live decisions exist only through the TensorFlow VSE. | Agent-in-the-loop decisions would need invasive changes. |
| 9 | Monza params exist only for **2014–2019**. **`pars_Monza_2019.ini` as shipped fails to construct**: its two real VSC periods are 1.46 laps apart and the engine requires ≥ 1.5. With them removed, cars **run out of fuel on the last lap** (−0.028 kg), because fuel loads are calibrated to that exact race's events. | It can't represent a real race it ships, and parameters are tied to historical events. That's the opposite of what alternate timelines need. 2024 needs full re-calibration anyway. |
| 10 | Last code change 2021 (repo pushed 2023); LGPL-3.0. | Unmaintained. Modifying it (needed for 4–8) creates LGPL obligations for the modified files. |

## Evidence: option B (time-discrete simulator)

The README describes it as *"more in a proof of concept state"*. It ships a single, incomplete
parameter file (Yas Marina 2017, 6 drivers) and has FCY phases "not implemented yet". It is a Rust
workspace, so using it from Python means a Rust toolchain plus bindings or a subprocess protocol,
with no serialisation story. Licence is fine (Apache/MIT), maturity is not.

## Scoring (plan §7.2 order)

| Criterion | A: TUM lap-discrete | B: TUM time-discrete | **C: ours** |
|---|---|---|---|
| 1. Pause/save/load/fork | In-memory only via private API + global RNG capture; disk = pickle; no CRN across forks | Unknown, unbuilt | **All M0 gates pass**: ≤ 1-tick pause, verified SQLite snapshots, fresh-process resume, forks with CRN (tests) |
| 2. Reproducibility | Two global RNGs; coupling across components | Unknown | **Keyed RNG, no hidden state**; cross-process hash equality tested |
| 3. 2024 calibration | None; 2019 Monza file broken; event-coupled params | None | Placeholder, but **FastF1 2024 data pipeline ready** (step 7) |
| 4. Performance | 54–116 ms/race (lap-level) | Unknown | 73 ms (2 cars) / 478 ms (20 cars) per race at 0.5 s ticks; lap-level fast path possible |
| 5. Licence | LGPL-3.0 (modification obligations) | Apache/MIT | Ours |
| 6. Maintenance | Unmaintained since 2021, Py3.8-era deps | Proof of concept | Ours, typed, 128 tests |

## Decision

- **Engine: option C.** Keep `lightweight_engine` behind the existing API (`create_world`, `start`,
  `pause`, `resume`, `step`, `apply_action`, `standings`) and grow it data-driven, as the plan prescribes.
- **Do not vendor or wrap TUM code.** No TUM code is copied into this repository.
- **Borrow modelling ideas, not code** (re-implemented from first principles, calibrated on FastF1,
  with credit to Heilmeier et al. / TUM FTM in docstrings where an idea is used):
  per-driver pace offset and lap-time sigma (we already have equivalents), **per-driver accident
  probability** (basis for the M2 mistake/DNF model), start performance distribution, fisk-like
  pit standstill distribution, overtaking time-loss model, FCY handling, fuel-mass effect, per-compound
  degradation curve shapes (linear/quadratic/log).
- **Adopt their good idea of pre-drawn exogenous events, safely:** our keyed RNG already gives forks
  identical exogenous luck. Future SC/VSC/failure events will be drawn on demand from keyed streams,
  never stored as visible future in the world state.
- **Racetrack-database (LGPL-3.0 data):** not needed now. The FastF1-derived outline (step 7) is enough
  for the 2D view. Revisit only if track width or pit-lane geometry becomes necessary.

## Follow-ups this creates

1. **M2:** pit stops + tyre choice, real start compounds, lapped cars/flag rule, DNF/mistake model
   (ideas from row 8–9 above), traffic/overtaking cost. Engine version bump each time.
2. **M4:** calibrate lap-model coefficients on FastF1 (2024 Monza first, then held-out races).
3. **M5 (performance):** if RL needs more throughput than ~2 races/s for 20 cars, add a lap-level
   fast path that uses the same lap model and keyed RNG, with a test that it matches the tick engine at
   lap boundaries.
