"""Car-to-car interaction: dirty air, held up, overtakes, blue flags (M2 slice 2).

The engine samples a whole lap at once and moves each car at constant speed through it, so
interaction is resolved when a car *joins the track*: at its line crossing, or when it leaves
the pit lane after a stop. At that moment every car ahead has already fixed its lap (it joined
earlier), so the resolution is a deterministic chain with no circular dependencies:

1. **Dirty air:** the nearest car ahead on the road is less than ``dirty_air_window_s`` away
   (time since it left the line) -> the lap costs ``dirty_air_loss_s``.
2. For every car ahead on the road, nearest first: if this car's planned arrival at the line is
   less than ``min_gap_s`` behind that car's arrival, they would meet during the lap:

   - **Lapped car ahead -> blue flag.** It yields. The attacker loses
     ``blue_flag_attacker_loss_s``; the lapped car loses ``blue_flag_defender_loss_s`` on its
     next lap (its current lap is already fixed).
   - **Otherwise -> overtake attempt.** ``margin`` = time the attacker would lose by sitting
     ``min_gap_s`` behind. Below ``overtake_threshold_s`` there's no attempt. Above it, the pass
     succeeds with probability ``1 - exp(-(margin - threshold) / scale)``, drawn from the keyed
     stream ``("overtake", attacker, defender, lap)``, so forks share it. A pass costs the
     attacker ``attacker_loss_s`` but must reach the line at least ``pass_margin_s`` ahead.
     When the free lap isn't quite enough, the car arrives exactly that far ahead; this stands in
     for DRS/slipstream. The defender loses ``defender_loss_s`` on its next lap.
   - **Failed or no attempt -> held up:** the car arrives exactly ``min_gap_s`` behind.

Because both cars move linearly through the rest of the lap, a held car never visually passes
the car it's stuck behind, and a passing car crosses it exactly once.

Idea credited to TUM FTM race-simulation (Heilmeier et al.); re-implemented with keyed RNG and
lap-start resolution. All parameters are editorial until calibrated (EXP-002, M4).
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

from f1ar.domain.race_state import CarState, RaceSetup
from f1ar.simulation.rng_streams import uniform

PASS_MARGIN_S = 0.1
"""A completed pass leads at the line by at least this much."""


@dataclass(frozen=True)
class Ahead:
    """A car ahead on the road: when its current lap started (racing clock) and how far ahead
    it is now, in seconds of its own driving."""

    car: CarState
    lap_start_s: float
    gap_s: float


@dataclass
class TrafficOutcome:
    lap_target_s: float
    events: list[tuple[str, dict[str, float | int | str]]] = field(default_factory=list)
    carried_losses: dict[str, float] = field(default_factory=dict)
    """``driver_id -> seconds`` added to that car's next lap (defending, yielding)."""


def resolve(
    setup: RaceSetup,
    seed: int,
    me: CarState,
    my_lap_start_s: float,
    ahead: Sequence[Ahead],
    *,
    context: str = "",
) -> TrafficOutcome:
    """New lap target for ``me``; ``ahead`` is nearest first.

    ``context`` separates the overtake draws of a second resolution in the same lap (the
    restart after a safety car) from the first.
    """
    p = setup.traffic
    out = TrafficOutcome(lap_target_s=me.lap_target_s)
    if not p.enabled or not ahead:
        return out
    lap = me.laps_completed + 1
    salt = (context,) if context else ()

    if ahead[0].gap_s < p.dirty_air_window_s:
        out.lap_target_s += p.dirty_air_loss_s

    passed: list[tuple[str, float]] = []  # (rival, their arrival) for passes made so far
    for other in ahead:
        their_end = other.lap_start_s + other.car.lap_target_s
        my_end = my_lap_start_s + out.lap_target_s
        if my_end >= their_end + p.min_gap_s:
            continue  # they don't meet before the line
        rival = other.car.driver_id

        margin = their_end + p.min_gap_s - my_end
        chance = (
            1.0 - math.exp(-(margin - p.overtake_threshold_s) / p.overtake_scale_s)
            if margin > p.overtake_threshold_s
            else 0.0
        )
        if other.car.laps_completed < me.laps_completed:
            out.lap_target_s += p.blue_flag_attacker_loss_s
            _carry(out, rival, p.blue_flag_defender_loss_s)
            out.events.append(("BLUE_FLAG", {"yielded": rival, "lap": lap}))
        elif chance > 0.0 and uniform(seed, "overtake", me.driver_id, rival, lap, *salt) < chance:
            new_end = min(my_end + p.attacker_loss_s, their_end - PASS_MARGIN_S)
            out.lap_target_s = new_end - my_lap_start_s
            _carry(out, rival, p.defender_loss_s)
            out.events.append(
                (
                    "OVERTAKE",
                    {"passed": rival, "lap": lap, "margin_s": margin, "probability": chance},
                )
            )
            passed.append((rival, their_end))
        else:
            # Stuck behind: arrive exactly min_gap behind.
            out.lap_target_s = their_end + p.min_gap_s - my_lap_start_s
        # Any of the above can make us arrive later: a hold, or the time lost passing or
        # lapping a car further up the road. That may put us level with or behind a car we
        # "passed" earlier in this loop; then that pass never happened and we're held behind
        # that car instead. Repeat until every remaining pass really ends ahead.
        _undo_passes_not_completed(out, passed, my_lap_start_s, p.min_gap_s, p.defender_loss_s)
    return out


def _undo_passes_not_completed(
    out: TrafficOutcome,
    passed: list[tuple[str, float]],
    my_lap_start_s: float,
    min_gap_s: float,
    defender_loss_s: float,
) -> None:
    changed = True
    while changed:
        changed = False
        for rival, their_end in list(passed):
            if my_lap_start_s + out.lap_target_s > their_end - PASS_MARGIN_S:
                passed.remove((rival, their_end))
                out.events = [
                    (k, pl)
                    for k, pl in out.events
                    if not (k == "OVERTAKE" and pl["passed"] == rival)
                ]
                _carry(out, rival, -defender_loss_s)
                if out.carried_losses[rival] <= 1e-12:
                    del out.carried_losses[rival]
                out.lap_target_s = max(out.lap_target_s, their_end + min_gap_s - my_lap_start_s)
                changed = True


def _carry(out: TrafficOutcome, driver_id: str, seconds: float) -> None:
    out.carried_losses[driver_id] = out.carried_losses.get(driver_id, 0.0) + seconds
