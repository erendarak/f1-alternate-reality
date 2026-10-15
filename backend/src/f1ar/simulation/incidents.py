"""Incidents: driver mistakes, crashes and mechanical failures, and the neutralisation they cause.

Everything here is a pure function of (seed, driver, lap, profile, pace mode): nothing is drawn
in advance and stored in the world state, so the future isn't sitting in a save file. Forks share
the draws (common random numbers), so a branch only differs where its decisions differ (e.g. a
car that pushes is more likely to crash on the laps it pushes).

Per lap, for a car that starts the lap:

- **Mistake** (time loss, the car continues): probability ``mistake_per_lap`` x skill factor x
  pace-mode factor; the loss is uniform in [min, max] and it happens at a keyed point in the lap.
- **Crash** (retirement): ``crash_per_lap`` x skill factor x pace-mode factor, x
  ``first_lap_crash_multiplier`` on lap 1.
- **Mechanical failure** (retirement): ``failure_per_lap``, the same for every car.

Skill factor = ``(1.5 - consistency) x (1.5 - composure)`` (1.0 for a neutral 0.5/0.5 driver,
0.25 for a 1.0/1.0 driver, 2.25 for 0/0). ``risk_appetite`` deliberately isn't used: it's a
strategy preference (D-010), not a crash rate.

After a retirement, the race control decision (SC / VSC / nothing) and its duration are keyed by
(driver, lap) too. W4 (engine v0.11.0, D-042): a crash can stop the race (red flag); a
neutralisation can also come without a retirement (debris), drawn per race lap; failure rates
differ by team (``DriverEntry.failure_multiplier``) and crash/mistake rates are renormalised
for the W1 pace-mode mix (``risk_scale``). Time penalties are drawn here too
(``penalty_draw``). Ideas credited to TUM FTM race-simulation (per-driver accident probability,
first-lap risk, FCY phases); all rates are editorial (D-024).
"""

from dataclasses import dataclass
from enum import StrEnum

from f1ar.domain.race_state import CarState, PaceMode, RacePhase, RaceSetup
from f1ar.simulation.rng_streams import uniform


class IncidentKind(StrEnum):
    MISTAKE = "MISTAKE"
    CRASH = "CRASH"
    FAILURE = "FAILURE"


@dataclass(frozen=True)
class Incident:
    kind: IncidentKind
    lap_fraction: float
    """Where in the lap it happens, in [0.05, 0.95] of the driven part."""
    time_loss_s: float = 0.0
    """MISTAKE only."""


@dataclass(frozen=True)
class Neutralisation:
    phase: RacePhase
    vsc_s: float = 0.0
    sc_laps: int = 0
    red_flag: bool = False
    """W4: the race is stopped first (free tyre change for everyone), then restarted behind the
    safety car (``phase`` SC)."""


def _mode_factor(setup: RaceSetup, mode: PaceMode) -> float:
    p = setup.race_control
    if mode is PaceMode.PUSH:
        return p.push_risk_multiplier
    if mode is PaceMode.CONSERVE:
        return p.conserve_risk_multiplier
    return 1.0


def skill_factor(setup: RaceSetup, driver_id: str) -> float:
    a = setup.driver(driver_id).profile.archetype
    return (1.5 - a.consistency) * (1.5 - a.composure)


def lap_incident(
    setup: RaceSetup, seed: int, car: CarState, lap: int, wetness: float = 0.0
) -> Incident | None:
    """The incident (if any) of ``car`` on ``lap``, driven in ``car.lap_pace_mode``; a wet
    track multiplies crash and mistake rates by ``1 + wet_incident_factor * wetness``."""
    p = setup.race_control
    if not p.enabled:
        return None
    risk = skill_factor(setup, car.driver_id) * _mode_factor(setup, car.lap_pace_mode)
    risk *= p.risk_scale
    risk *= 1.0 + setup.weather.wet_incident_factor * wetness
    crash = p.crash_per_lap * risk * (p.first_lap_crash_multiplier if lap == 1 else 1.0)
    failure = p.failure_per_lap * setup.driver(car.driver_id).failure_multiplier
    mistake = p.mistake_per_lap * risk
    where = 0.05 + 0.9 * uniform(seed, "incident_where", car.driver_id, lap)
    u = uniform(seed, "incident", car.driver_id, lap)
    if u < crash:
        return Incident(IncidentKind.CRASH, where)
    if u < crash + failure:
        return Incident(IncidentKind.FAILURE, where)
    if u < crash + failure + mistake:
        return Incident(
            IncidentKind.MISTAKE, where, mistake_loss_s(setup, seed, car.driver_id, lap)
        )
    return None


def mistake_loss_s(setup: RaceSetup, seed: int, driver_id: str, lap: int) -> float:
    """Time lost by a mistake on ``lap``: its own keyed draw (doesn't depend on the rates, so
    reporting it later in the lap gives the same value whatever the conditions are then)."""
    p = setup.race_control
    size = uniform(seed, "mistake_loss", driver_id, lap)
    return p.mistake_loss_min_s + (p.mistake_loss_max_s - p.mistake_loss_min_s) * size


def neutralisation_after(
    setup: RaceSetup, seed: int, driver_id: str, lap: int, kind: IncidentKind
) -> Neutralisation | None:
    """Race control's response to a retirement: safety car, VSC or nothing."""
    p = setup.race_control
    sc, vsc = (
        (p.crash_sc_probability, p.crash_vsc_probability)
        if kind is IncidentKind.CRASH
        else (p.failure_sc_probability, p.failure_vsc_probability)
    )
    u = uniform(seed, "race_control", driver_id, lap)
    length = uniform(seed, "race_control_length", driver_id, lap)
    if kind is IncidentKind.CRASH and uniform(seed, "red_flag", driver_id, lap) < (
        p.red_flag_per_crash
    ):
        return Neutralisation(RacePhase.SC, sc_laps=p.red_flag_sc_laps, red_flag=True)
    if u < sc:
        laps = p.sc_min_laps + min(
            int(length * (p.sc_max_laps - p.sc_min_laps + 1)), p.sc_max_laps - p.sc_min_laps
        )
        return Neutralisation(RacePhase.SC, sc_laps=laps)
    if u < sc + (1.0 - sc) * vsc:
        return Neutralisation(
            RacePhase.VSC, vsc_s=p.vsc_min_s + (p.vsc_max_s - p.vsc_min_s) * length
        )
    return None


def debris_neutralisation(setup: RaceSetup, seed: int, lap: int) -> Neutralisation | None:
    """A neutralisation without a retirement (debris, a car stopped and recovered), drawn when
    the leader starts ``lap`` (W4). Keyed by the lap: every fork shares it."""
    p = setup.race_control
    if not p.enabled or p.debris_per_lap <= 0.0:
        return None
    if uniform(seed, "debris", lap) >= p.debris_per_lap:
        return None
    length = uniform(seed, "debris_length", lap)
    if uniform(seed, "debris_kind", lap) < p.debris_sc_share:
        laps = p.sc_min_laps + min(
            int(length * (p.sc_max_laps - p.sc_min_laps + 1)), p.sc_max_laps - p.sc_min_laps
        )
        return Neutralisation(RacePhase.SC, sc_laps=laps)
    return Neutralisation(RacePhase.VSC, vsc_s=p.vsc_min_s + (p.vsc_max_s - p.vsc_min_s) * length)


def penalty_factor(setup: RaceSetup, driver_id: str) -> float:
    """How penalty-prone a driver is relative to the field: ``1.5 - composure`` over the
    field's mean (editorial: it decides *who* is penalised; the rates set how often)."""
    field = [1.5 - d.profile.archetype.composure for d in setup.drivers]
    mean = sum(field) / len(field)
    return (1.5 - setup.driver(driver_id).profile.archetype.composure) / mean


def penalty_draw(
    setup: RaceSetup, seed: int, kind: str, driver_id: str, *key: str | int, rate: float
) -> bool:
    """Whether ``driver_id`` gets a ``kind`` penalty for the event keyed by ``key`` (W4)."""
    if not setup.race_control.enabled or rate <= 0.0:
        return False
    return uniform(seed, "penalty", kind, driver_id, *key) < rate * penalty_factor(setup, driver_id)
