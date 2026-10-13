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


@dataclass(frozen=True)
class BaseLapEstimate:
    base_lap_s: float
    laps_used: int
    note: str


def estimate_base_lap(
    laps: pd.DataFrame,
    roster: RaceRoster,
    profiles: ProfileSet,
    lap_model: LapModelParams,
    car_estimate: "CarPaceEstimate",
    total_laps: int,
) -> BaseLapEstimate:
    """``base_lap_s`` such that the lap model's mean prediction for the fastest team's real
    representative laps matches them (median residual).

    For each such lap: ``lap_time - skill term - compound offset - wear so far - fuel term``,
    where wear so far is the model's own wear after ``tyre life - 1`` laps (BALANCED mode, the
    engine's ``degradation_s``) and fuel = ``fuel_s_per_lap_remaining x laps remaining``.
    It anchors the overall level of lap times to the track for whatever ``lap_model`` is given
    (since M4 the calibrated one, D-033).
    """
    from f1ar.domain.race_state import PaceMode
    from f1ar.simulation.lap_model import degradation_s

    fastest = min(car_estimate.car_pace_s, key=lambda team: car_estimate.car_pace_s[team])
    by_number = {e.car_number: e for e in roster.entries}
    residuals: list[float] = []
    for row in representative_laps(laps).to_dict("records"):
        entry = by_number.get(int(row["driver_number"]))
        compound = row["compound"]
        if entry is None or entry.team_id != fastest or compound not in Compound.__members__:
            continue
        driver = profiles.get(entry.driver_id).archetype
        c = Compound(compound)
        skill = lap_model.skill_gap_s * (1.0 - driver.race_pace_skill)
        life = row.get("tyre_life_laps")  # missing -> treat as a fresh tyre (no wear term)
        tyre_life = 1 if life is None or pd.isna(life) else int(life)
        wear = degradation_s(
            lap_model,
            driver,
            compound=c,
            tyre_age_laps=max(0, tyre_life - 1),
            pace_mode=PaceMode.BALANCED,
        )
        fuel = lap_model.fuel_s_per_lap_remaining * (total_laps - int(row["lap_number"]) + 1)
        residuals.append(
            float(row["lap_time_s"]) - skill - lap_model.compounds[c].offset_s - wear - fuel
        )
    if not residuals:
        raise ValueError("no representative laps of the fastest team")
    base = round(float(pd.Series(residuals).median()), 3)
    return BaseLapEstimate(
        base_lap_s=base,
        laps_used=len(residuals),
        note=(
            f"estimated_from_data: base lap {base} s = median residual of {len(residuals)} "
            f"representative laps of the fastest team ({fastest}) after the model's skill, "
            "compound, wear and fuel terms"
        ),
    )


@dataclass(frozen=True)
class PitLossEstimate:
    total_loss_s: float
    """Median of (in-lap + out-lap - 2 x the driver's median representative lap)."""
    stops_used: int
    note: str


def estimate_pit_loss(laps: pd.DataFrame) -> PitLossEstimate:
    """Total time a stop costs on this track, from real in-laps and out-laps.

    It includes standstill, pit-lane driving, cold tyres on the out-lap and whatever traffic
    the stop caused, so it's a whole-stop figure, not a pure pit-lane length.
    """
    medians = representative_laps(laps).groupby("driver_number")["lap_time_s"].median()
    losses: list[float] = []
    for number, group in laps.groupby("driver_number"):
        if number not in medians.index:
            continue
        rows = group.sort_values("lap_number").to_dict("records")
        for in_lap, out_lap in zip(rows, rows[1:], strict=False):
            if (
                pd.isna(in_lap["pit_in_time_s"])
                or pd.isna(out_lap["pit_out_time_s"])
                or pd.isna(in_lap["lap_time_s"])
                or pd.isna(out_lap["lap_time_s"])
            ):
                continue
            losses.append(
                float(in_lap["lap_time_s"]) + float(out_lap["lap_time_s"]) - 2 * medians[number]
            )
    if not losses:
        raise ValueError("no complete in-lap/out-lap pairs")
    total = round(float(pd.Series(losses).median()), 2)
    return PitLossEstimate(
        total_loss_s=total,
        stops_used=len(losses),
        note=(
            f"estimated_from_data: median whole-stop loss {total} s over {len(losses)} stops "
            f"(in-lap + out-lap - 2 x driver median representative lap)"
        ),
    )


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
