"""The same race metrics as ``reference_metrics``, measured on a simulated race (EXP-002 onwards).

Overtakes are counted exactly as for a real race (positions at the line, pit laps excluded) and
put in the same causes; a lap is "neutral" if any car completes it under SC/VSC (and the lap
after: the restart).
"""

from collections import Counter

from f1ar.calibration.reference_metrics import (
    RaceMetrics,
    count_overtakes,
    pass_causes,
    sequence_of,
)
from f1ar.domain.race_state import CarStatus, RacePhase, RaceSetup, WorldState, WorldStatus
from f1ar.simulation import lightweight_engine as engine


def sim_metrics(setup: RaceSetup, seed: int, *, until_lap: int | None = None) -> RaceMetrics:
    """Run one race (or until every running car has completed ``until_lap`` laps) and measure
    it. With ``until_lap`` only the overtake counts are meaningful."""
    state: WorldState = engine.start(engine.create_world(setup, seed=seed, world_id="metrics"))
    code = {d.driver_id: d.code for d in setup.drivers}
    crossings: dict[int, dict[str, float]] = {}
    stopped: dict[str, set[int]] = {}
    neutral: set[int] = set()
    tyre_age: dict[tuple[str, int], int] = {}
    compounds: dict[str, list[str]] = {
        code[d.driver_id]: [d.start_compound.value] for d in setup.drivers
    }
    counts: Counter[str] = Counter()
    while state.world_status is WorldStatus.RUNNING:
        before = state
        result = engine.step(state)
        state = result.state
        for e in result.events:
            counts[e.type] += 1
            if e.driver_id is None:
                continue
            c = code[e.driver_id]
            if e.type == "LAP_COMPLETED":
                lap = int(e.payload["lap"])
                crossings.setdefault(lap, {})[c] = float(e.payload["crossing_time_s"])
                tyre_age[(c, lap)] = before.car(e.driver_id).tyre_age_laps
                phases = (before.race_control.phase, state.race_control.phase)
                if any(p is not RacePhase.GREEN for p in phases):
                    neutral.update((lap, lap + 1))
            elif e.type == "PIT_STOP":
                stopped.setdefault(c, set()).add(int(e.payload["lap"]))
                compounds[c].append(str(e.payload["to_compound"]))
        if until_lap is not None and all(
            c.laps_completed >= until_lap for c in state.cars if c.status is CarStatus.RUNNING
        ):
            break
    order = [code[d] for d in engine.standings(state)]
    finished = {code[c.driver_id] for c in state.cars if c.status is CarStatus.FINISHED}
    winner_car = state.car(engine.standings(state)[0])
    grid = [d.code for d in sorted(setup.drivers, key=lambda d: d.grid_position)]
    if until_lap is not None:
        crossings = {lap: t for lap, t in crossings.items() if lap <= until_lap}
    return RaceMetrics(
        order=order,
        finishers=finished,
        winner=order[0],
        winner_time_s=winner_car.finish_time_s or 0.0,
        stops={code[c.driver_id]: c.pit_stops for c in state.cars if code[c.driver_id] in finished},
        sequences={d: sequence_of(compounds[d]) for d in finished},
        overtakes=count_overtakes(crossings, grid, stopped),
        retirements=sum(1 for c in state.cars if c.status is CarStatus.RETIRED),
        safety_cars=state.race_control.sc_count,
        vscs=state.race_control.vsc_count,
        lapped=sum(
            1
            for c in state.cars
            if c.status is CarStatus.FINISHED and c.laps_completed < winner_car.laps_completed
        ),
        extra={"pass_causes": dict(pass_causes(crossings, grid, stopped, neutral, tyre_age))},
    )
