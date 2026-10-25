"""Rule teams that react to rivals (A1b, D-062): the undercut and covering it.

Our planner optimises each car's own race time on an empty track, so it never tries to jump a
car it is stuck behind, and never answers a rival's stop. Real teams do both: EXP-020 counted
4.6 undercut attempts a race in 2024 against 1.5 in our rule field. Two rules (shape
editorial, windows fitted to the real attempt rate, EXP-031):

- **undercut**: crossing the line within ``attack_gap_s`` of the car ahead on ``attack_laps``
  laps in a row, with the planned stop at most ``attack_window_laps`` laps away: stop now (the
  planned compound; the planner re-plans the rest after the stop);
- **cover**: the car right behind, within ``cover_gap_s`` at the last line, has just pitted, and
  the planned stop is at most ``cover_window_laps`` away: stop now.

Gaps are read from line-crossing times (``lap_started_at_s``) of cars that already crossed: what
a pit wall sees on the timing screen.
"""

from f1ar.domain.race_state import (
    CarState,
    CarStatus,
    Compound,
    StrategyController,
    WorldState,
)
from f1ar.simulation.weather import is_slick


def gap_ahead_s(state: WorldState, car: CarState, crossing_time: float) -> float | None:
    """Time to the car that crossed the line just before ``car`` on the same lap (cars in the
    pit lane aren't on the road ahead)."""
    lap = car.laps_completed + 1
    best = None
    for o in state.cars:
        if o.driver_id == car.driver_id or o.status is CarStatus.RETIRED or o.pit_hold_s > 0.0:
            continue
        if o.laps_completed == lap and o.lap_started_at_s <= crossing_time:
            gap = crossing_time - o.lap_started_at_s
            best = gap if best is None or gap < best else best
    return best


def rival_just_pitted_behind(state: WorldState, car: CarState, within_s: float) -> str | None:
    """The car that was right behind ``car`` at the last line (within ``within_s``) and pitted
    there: it is on its out-lap now."""
    for o in state.cars:
        if o.driver_id == car.driver_id or o.status is not CarStatus.RUNNING:
            continue
        behind = o.lap_started_at_s - car.lap_started_at_s
        if (
            o.laps_completed == car.laps_completed
            and o.pit_hold_s > 0.0
            and o.tyre_age_laps == 0
            and 0.0 < behind <= within_s
        ):
            return o.driver_id
    return None


def rival_stop(
    state: WorldState, car: CarState, crossing_time: float
) -> tuple[Compound | None, int, str]:
    """``car`` crossing the line (state still at its lap's start): a stop the rival rules call
    now, the updated stuck-lap count, and the reason."""
    rules = state.setup.rival_rules
    if rules is None:
        return None, 0, ""
    gap = gap_ahead_s(state, car, crossing_time)
    stuck = car.stuck_laps + 1 if gap is not None and gap < rules.attack_gap_s else 0
    if car.controller is not StrategyController.BASELINE or not car.planned_stops:
        return None, stuck, ""
    nxt = car.planned_stops[0]
    until = nxt.lap - (car.laps_completed + 1)
    if until < 1 or not is_slick(nxt.compound):
        return None, stuck, ""
    if stuck >= rules.attack_laps and until <= rules.attack_window_laps:
        return nxt.compound, 0, f"undercut ({gap:.1f} s behind for {stuck} laps)"
    rival = rival_just_pitted_behind(state, car, rules.cover_gap_s)
    if rival is not None and until <= rules.cover_window_laps:
        return nxt.compound, stuck, f"cover {rival}"
    return None, stuck, ""
