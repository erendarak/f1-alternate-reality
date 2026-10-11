"""Turn a fetched race into engine inputs: real start tyres and team car-pace offsets.

**Start compounds** (``historical_verified``): the compound each car ran on lap 1.

**Car pace offsets** (``estimated_from_data``), plan §5.2 "car vs driver kept separate":

    driver median green-flag lap      m_d     (representative laps, see data.quality)
    our model's driver contribution   s_d  =  skill_gap * (1 - race_pace_skill_d)
    car-only pace                     c_d  =  m_d - s_d
    team car pace                     C_t  =  mean of c_d over the team's drivers
    offset                            o_t  =  C_t - min over teams of C_t      (fastest car = 0)

Removing ``s_d`` means the engine doesn't count a driver's speed twice: once through the
persona and again through the car. This is a deliberately simple first estimate. It mixes
fuel loads, tyre compounds and traffic across the race and uses one race only; proper fitting
is M4.
"""

from collections import defaultdict
from dataclasses import dataclass
from statistics import fmean

import pandas as pd

from f1ar.data.quality import representative_laps
from f1ar.domain.driver_profiles import ProfileSet
from f1ar.domain.identities import RaceRoster
from f1ar.domain.race_state import Compound, LapModelParams

MIN_LAPS = 10
"""A driver needs this many representative laps to contribute to their team's estimate."""


@dataclass(frozen=True)
class CarPaceEstimate:
    offsets_s: dict[str, float]
    """``team_id -> seconds per lap slower than the fastest car``."""
    car_pace_s: dict[str, float]
    driver_median_s: dict[str, float]
    excluded_drivers: tuple[str, ...]
    laps_used: int
    note: str


def start_compounds(laps: pd.DataFrame, roster: RaceRoster) -> dict[str, Compound]:
    """``driver_id -> compound on lap 1``. Only dry compounds the engine models are returned."""
    by_number = {e.car_number: e.driver_id for e in roster.entries}
    first = laps[laps["lap_number"] == 1]
    out: dict[str, Compound] = {}
    for number, compound in zip(first["driver_number"], first["compound"], strict=True):
        driver_id = by_number.get(int(number))
        if driver_id is not None and not pd.isna(compound) and compound in Compound.__members__:
            out[driver_id] = Compound(compound)
    return out


def estimate_car_offsets(
    laps: pd.DataFrame,
    roster: RaceRoster,
    profiles: ProfileSet,
    lap_model: LapModelParams,
) -> CarPaceEstimate:
    reps = representative_laps(laps)
    by_number = {e.car_number: e for e in roster.entries}
    team_values: dict[str, list[float]] = defaultdict(list)
    medians: dict[str, float] = {}
    excluded: list[str] = []
    for _, group in reps.groupby("driver_number"):
        entry = by_number.get(int(group["driver_number"].iloc[0]))
        if entry is None:
            continue
        if len(group) < MIN_LAPS:
            excluded.append(entry.driver_id)
            continue
        median = float(group["lap_time_s"].median())
        skill = profiles.get(entry.driver_id).archetype.race_pace_skill
        medians[entry.driver_id] = median
        team_values[entry.team_id].append(median - lap_model.skill_gap_s * (1.0 - skill))
    if not team_values:
        raise ValueError("no driver has enough representative laps")
    car_pace = {team: fmean(values) for team, values in team_values.items()}
    fastest = min(car_pace.values())
    offsets = {team: round(pace - fastest, 3) for team, pace in car_pace.items()}
    for entry in roster.entries:  # a team with no usable laps gets the field median offset
        if entry.team_id not in offsets:
            offsets[entry.team_id] = round(float(pd.Series(list(offsets.values())).median()), 3)
    return CarPaceEstimate(
        offsets_s=offsets,
        car_pace_s={t: round(v, 3) for t, v in car_pace.items()},
        driver_median_s={d: round(v, 3) for d, v in medians.items()},
        excluded_drivers=tuple(sorted(excluded)),
        laps_used=len(reps),
        note=(
            f"estimated_from_data: median of {len(reps)} representative laps per driver minus the "
            f"persona skill term, averaged per team, relative to the fastest car; drivers with "
            f"<{MIN_LAPS} laps excluded: {sorted(excluded) or 'none'}"
        ),
    )
