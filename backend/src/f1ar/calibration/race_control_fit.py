"""Race-control rates from the 2024 season (M4 slice 5, D-035, EXP-006).

Method of moments on the dry training races (D-031 split), checked on the dry test races:

- **Retirement hazard**: retirements on lap 1 per starter, and retirements per car-lap after
  lap 1 (exposure = laps each car started). The engine's crash rate is per driver (skill and
  pace mode, D-024); its failure rate isn't. Timing data says "Retired", not why, so the split
  between crashes and failures is **editorial** (``CRASH_SHARE`` of later retirements; lap-1
  retirements are all crashes).
- **Neutralisations per retirement**: safety cars and VSCs (track-status deployments) divided
  by retirements, the same for crashes and failures. Some real neutralisations have other
  causes (debris; Qatar 2024 lap 35: a mirror) and some retirements cause none (car into the
  pits): this is an average, not a cause model. A VSC turned into an SC counts as both.
- **Safety-car length**: deployment to green, in laps of the race's median green lap time (an SC
  that turned into a red flag is left out).
"""

import hashlib
import json
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

from f1ar.calibration.reference_metrics import red_flag_periods
from f1ar.data.status import is_finisher
from f1ar.domain.provenance import SourceType
from f1ar.domain.race_state import RaceControlParams

CRASH_SHARE = 0.6
"""Editorial: share of retirements after lap 1 that are crashes (the rest are failures)."""
RACE_CONTROL_FILE = Path("calibration") / "race_control_2024_v1.json"


@dataclass
class RaceCounts:
    event_id: str
    starters: int
    car_laps_after_lap1: int
    lap1_retirements: int
    later_retirements: int
    safety_cars: int
    vscs: int
    red_flags: int
    sc_laps: list[float] = field(default_factory=list)


def race_counts(data_dir: Path) -> RaceCounts:
    laps = pd.read_parquet(data_dir / "laps.parquet")
    results = pd.read_parquet(data_dir / "results.parquet")
    status = pd.read_parquet(data_dir / "track_status.parquet")
    started = results[results["status"] != "Did not start"]
    last = laps.groupby("driver_code")["lap_number"].max()
    total = int(laps["lap_number"].max())
    lap1 = later = exposure = 0
    for code, st in zip(started["driver_code"], started["status"], strict=True):
        n = int(last.get(code, 0))
        retired = not is_finisher(st)
        laps_started = n if retired else min(n, total)
        exposure += max(0, laps_started - 1)
        if retired:
            if n <= 1:
                lap1 += 1
            else:
                later += 1
    codes = [str(c) for c in status["status"]]
    times = [float(t) for t in status["session_time_s"]]

    def deployments(code: str) -> list[int]:
        return [i for i, c in enumerate(codes) if c == code and (i == 0 or codes[i - 1] != code)]

    green = laps[(laps["track_status"].astype(str) == "1") & laps["lap_time_s"].notna()]
    lap_s = float(green["lap_time_s"].median()) if len(green) else 90.0
    sc_laps = []
    for i in deployments("4"):
        j = next((j for j in range(i + 1, len(codes)) if codes[j] in ("1", "5")), None)
        if j is not None and codes[j] == "1":  # an SC that became a red flag has no length
            sc_laps.append(round((times[j] - times[i]) / lap_s, 2))
    return RaceCounts(
        event_id=data_dir.name,
        starters=len(started),
        car_laps_after_lap1=exposure,
        lap1_retirements=lap1,
        later_retirements=later,
        safety_cars=len(deployments("4")),
        vscs=len(deployments("6")),
        red_flags=len(red_flag_periods(status)),
        sc_laps=sc_laps,
    )


@dataclass(frozen=True)
class RaceControlFit:
    lap1_retirement_per_starter: float
    retirement_per_car_lap: float
    crash_per_lap: float
    failure_per_lap: float
    first_lap_crash_multiplier: float
    sc_per_retirement: float
    vsc_per_retirement_without_sc: float
    sc_laps_median: float
    sc_laps_p25: float
    sc_laps_p75: float
    races: int
    retirements: int
    safety_cars: int
    vscs: int
    fitted_on: tuple[str, ...]
    note: str = ""

    def to_json(self) -> str:
        d = asdict(self)
        d["fitted_on"] = list(self.fitted_on)
        return json.dumps(d, indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "RaceControlFit":
        d = json.loads(text)
        d["fitted_on"] = tuple(d.get("fitted_on", ()))
        return cls(**d)


def fit_race_control(counts: list[RaceCounts], *, note: str = "") -> RaceControlFit:
    starters = sum(c.starters for c in counts)
    exposure = sum(c.car_laps_after_lap1 for c in counts)
    lap1 = sum(c.lap1_retirements for c in counts)
    later = sum(c.later_retirements for c in counts)
    retirements = lap1 + later
    sc = sum(c.safety_cars for c in counts)
    vsc = sum(c.vscs for c in counts)
    lap1_rate = lap1 / starters
    later_rate = later / exposure
    crash = later_rate * CRASH_SHARE
    failure = later_rate - crash
    # lap 1: all retirements are crashes; the engine multiplies the crash rate on lap 1
    multiplier = max(1.0, (lap1_rate - failure) / crash) if crash > 0 else 1.0
    sc_lengths = sorted(x for c in counts for x in c.sc_laps)
    q = statistics.quantiles(sc_lengths, n=4) if len(sc_lengths) >= 2 else [3.0, 4.0, 5.0]
    return RaceControlFit(
        lap1_retirement_per_starter=round(lap1_rate, 5),
        retirement_per_car_lap=round(later_rate, 6),
        crash_per_lap=round(crash, 6),
        failure_per_lap=round(failure, 6),
        first_lap_crash_multiplier=round(multiplier, 2),
        sc_per_retirement=round(sc / retirements, 4) if retirements else 0.0,
        vsc_per_retirement_without_sc=round(vsc / max(1, retirements - sc), 4),
        sc_laps_median=round(statistics.median(sc_lengths), 2) if sc_lengths else 4.0,
        sc_laps_p25=round(q[0], 2),
        sc_laps_p75=round(q[2], 2),
        races=len(counts),
        retirements=retirements,
        safety_cars=sc,
        vscs=vsc,
        fitted_on=tuple(c.event_id for c in counts),
        note=note,
    )


def load_race_control(configs_dir: Path) -> tuple[RaceControlFit, str]:
    text = (configs_dir / RACE_CONTROL_FILE).read_text(encoding="utf-8")
    return RaceControlFit.from_json(text), "sha256:" + hashlib.sha256(text.encode()).hexdigest()


def race_control_params(
    fit: RaceControlFit, sha: str, base: RaceControlParams, mean_skill_factor: float
) -> RaceControlParams:
    """The engine's race-control rates from the season fit. The engine multiplies the crash rate
    by each driver's skill factor (editorial personas), so the base rate is divided by the
    field's mean factor: the average driver crashes at the fitted rate. Mistakes, pace-mode
    risk factors, VSC length and the SC mechanics stay editorial."""
    low = max(1, round(fit.sc_laps_p25))
    # validated: a fitted value outside the schema must fail here, not when a save is loaded
    return RaceControlParams.model_validate(
        base.model_dump()
        | {
            "source_type": SourceType.ESTIMATED_FROM_DATA,
            "note": (
                f"estimated_from_data: retirements {fit.retirement_per_car_lap:.5f}/car-lap after "
                f"lap 1, {fit.lap1_retirement_per_starter:.4f}/starter on lap 1, SC "
                f"{fit.sc_per_retirement:.2f} and VSC {fit.vsc_per_retirement_without_sc:.2f} per "
                f"retirement, SC {low}-{max(low, round(fit.sc_laps_p75))} laps, from "
                f"{fit.races} dry 2024 races (EXP-006, {sha[:15]}); crash/failure split "
                f"editorial; field mean skill factor {mean_skill_factor:.2f}"
            ),
            "crash_per_lap": fit.crash_per_lap / mean_skill_factor,
            "failure_per_lap": fit.failure_per_lap,
            "first_lap_crash_multiplier": max(1.0, fit.first_lap_crash_multiplier),
            "crash_sc_probability": fit.sc_per_retirement,
            "failure_sc_probability": fit.sc_per_retirement,
            "crash_vsc_probability": fit.vsc_per_retirement_without_sc,
            "failure_vsc_probability": fit.vsc_per_retirement_without_sc,
            "sc_min_laps": low,
            "sc_max_laps": max(low, round(fit.sc_laps_p75)),
        }
    )
