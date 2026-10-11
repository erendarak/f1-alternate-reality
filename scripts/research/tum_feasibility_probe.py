"""Reproducible feasibility probe for TUMFTM/race-simulation (M0 step 8; SIMULATOR_DECISION.md).

This is NOT part of f1ar and must NOT run in the project environment. Run it against a separate
checkout, in its own throwaway venv, so third-party code never touches ours:

    git clone --depth 1 https://github.com/TUMFTM/race-simulation <SANDBOX>/race-simulation
    uv venv --python 3.12 <SANDBOX>/.venv-tum
    uv pip install --python <SANDBOX>/.venv-tum numpy pandas scipy matplotlib
    <SANDBOX>/.venv-tum/Scripts/python -I scripts/research/tum_feasibility_probe.py \
        <SANDBOX>/race-simulation [pars_Spielberg_2019.ini] [--drop-fcy]

Each check prints one ``RESULT | name | OK/FAIL | detail`` line.
"""

import copy
import io
import random
import sys
import time
import traceback
import types
from contextlib import redirect_stdout
from typing import Any

REPO = sys.argv[1]
RACE = next((a for a in sys.argv[2:] if a.endswith(".ini")), "pars_Spielberg_2019.ini")
DROP_FCY = "--drop-fcy" in sys.argv
sys.path.insert(0, REPO)

# racesim/src/race.py imports TensorFlow (via the ML strategy engineer) and cvxpy (via
# racesim_basic) at module load, even when neither is used. We never enable them: stub them.
STUBBED = []
for mod in ("tensorflow", "tf_agents", "tensorflow_probability", "cvxpy"):
    try:
        __import__(mod)
    except ImportError:
        sys.modules[mod] = types.ModuleType(mod)
        STUBBED.append(mod)


def result(name: str, ok: bool, detail: str) -> None:
    print(f"RESULT | {name} | {'OK' if ok else 'FAIL'} | {detail}", flush=True)


try:
    import numpy as np
    from racesim.src.import_pars import import_pars
    from racesim.src.race import Race
except Exception as exc:
    result("import", False, f"{type(exc).__name__}: {exc}")
    sys.exit(1)
result(
    "import", True, f"Python {sys.version.split()[0]}, numpy {np.__version__}; stubbed {STUBBED}"
)

with redirect_stdout(io.StringIO()):
    PARS, _ = import_pars(
        use_print=False, use_vse=False, race_pars_file=RACE, mcs_pars_file="pars_mcs.ini"
    )
if DROP_FCY:
    PARS["event_pars"]["fcy_data"]["phases"] = []


def new_race(*, prob: bool, events: bool) -> Any:
    p = copy.deepcopy(PARS)
    return Race(
        race_pars=p["race_pars"],
        driver_pars=p["driver_pars"],
        car_pars=p["car_pars"],
        tireset_pars=p["tireset_pars"],
        track_pars=p["track_pars"],
        vse_pars=p["vse_pars"],
        vse_paths=None,
        use_prob_infl=prob,
        create_rand_events=events,
        monte_carlo_pars=p["monte_carlo_pars"],
        event_pars=p["event_pars"],
    )


def step_to(race: Any, lap: int) -> None:
    while race.cur_lap < lap:
        race._Race__simulate_lap()  # private, name-mangled: the only lap-wise step


def same(a: Any, b: Any) -> bool:
    return bool(np.allclose(a.racetimes, b.racetimes, equal_nan=True))


def seed(s: int, *, numpy_too: bool = True) -> None:
    random.seed(s)
    if numpy_too:
        np.random.seed(s)


try:
    laps = PARS["race_pars"]["tot_no_laps"]
    result("pars", True, f"{RACE}: {laps} laps, {len(PARS['driver_pars'])} drivers")

    t0 = time.perf_counter()
    r = new_race(prob=False, events=False)
    with redirect_stdout(io.StringIO()):
        r.simulate_race()
    result(
        "single_race_deterministic",
        r.result_status == 0,
        f"{(time.perf_counter() - t0) * 1000:.0f} ms, status {r.result_status}",
    )

    seed(1)
    t0 = time.perf_counter()
    with redirect_stdout(io.StringIO()):
        for _ in range(50):
            new_race(prob=True, events=True).simulate_race()
    per = (time.perf_counter() - t0) / 50
    result("throughput_stochastic", True, f"{per * 1000:.0f} ms per race ({1 / per:.1f} races/s)")

    def full(s: int, numpy_too: bool) -> Any:
        seed(s, numpy_too=numpy_too)
        race = new_race(prob=True, events=True)
        step_to(race, laps)
        return race

    result(
        "repeat_python_seed_only",
        same(full(7, False), full(7, False)),
        "random.seed alone: pit times use scipy.stats (NumPy global RNG) -> not repeatable",
    )
    result(
        "repeat_both_global_seeds",
        same(full(7, True), full(7, True)),
        "random.seed + np.random.seed",
    )

    seed(7)
    ref = new_race(prob=True, events=True)
    step_to(ref, laps)
    seed(7)
    run = new_race(prob=True, events=True)
    step_to(run, 20)
    snap, rng, nprng = copy.deepcopy(run), random.getstate(), np.random.get_state()
    random.random()
    np.random.random()
    random.setstate(rng)
    np.random.set_state(nprng)
    step_to(snap, laps)
    result(
        "deepcopy_fork_resume",
        same(ref, snap),
        "step to lap 20, deepcopy + capture BOTH global RNG states, resume == uninterrupted",
    )

    seed(7)
    a = new_race(prob=True, events=True)
    step_to(a, 20)
    b = copy.deepcopy(a)
    rng, nprng = random.getstate(), np.random.get_state()
    step_to(a, laps)
    random.setstate(rng)
    np.random.set_state(nprng)
    np.random.random()  # e.g. one extra pit stop's standstill draw in a fork
    step_to(b, laps)
    result(
        "common_random_numbers_across_forks",
        same(a, b),
        "one extra NumPy draw in a branch shifts all later random outcomes of every car",
    )

    drv = r.drivers_list[0]

    def slots(cls: type) -> int:
        return sum(len(getattr(c, "__slots__", ())) for c in cls.__mro__)

    result(
        "state_surface",
        True,
        f"__slots__: Race {slots(type(r))}, Driver {slots(type(drv))}, "
        f"Car {slots(type(drv.car))}, Tireset {slots(type(drv.car.tireset))}, "
        f"Track {slots(type(r.track))}; persistence in repo = pickle only",
    )
    pre = new_race(prob=True, events=True)
    result(
        "events_predrawn_at_init",
        True,
        f"FCY {len(pre.fcy_data['phases'])} phase(s), "
        f"{sum(x is not None for x in pre.retire_data['retirements'])} retirement(s) fixed "
        "before lap 1",
    )
    result(
        "driver_model",
        True,
        f"{drv.initials}: t_driver {drv.t_driver}, lap sigma {drv.t_lap_var_sigma}, "
        f"p_accident {drv.p_accident}; strategy_info {drv.strategy_info}",
    )
except Exception as exc:
    result("probe", False, f"{type(exc).__name__}: {exc}")
    traceback.print_exc()
