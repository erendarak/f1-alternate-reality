"""The field's in-race tyre estimate (A1, D-061): what teams learn from the timing screen.

Every team sees every car's lap times. A clean lap (green, dry, no traffic, no pit, no incident,
not lap 1) becomes a ``TimingObservation``: its lap time minus what the race-day prior expects
for that car (its tyres as the prior predicts them, its fuel, its known pace). The residual is

    r = (k - 1) x prior wear + trend x lap + race form of the car + unexpected cliff + noise

with ``k`` = true / prior wear, the race's hidden track trend, and each car's hidden form
(a constant per car). The estimate, recomputed once per race lap:

1. **Wear and trend**: Bayesian least squares on laps without a cliff (under the prior and the
   current life estimate), one free constant per car; priors ``N(0, wear_sd)`` on ``k - 1`` and
   ``N(0, trend_sd)`` on the trend. Cars on different tyre ages on the same lap, and the drop
   in lap time at a stop, separate wear from the trend.
2. **Life per compound**: a grid posterior on ``u = log(true / prior life)``. Under each ``u``
   every lap predicts its cliff (``cliff_s``); laps without the predicted cliff rule short lives
   out, cars falling off rule long ones out. Compounds share a factor ``s ~ N(0,
   life_shared_sd)`` and differ by ``N(0, life_compound_sd)`` (the measured surprise sizes,
   H3), so a soft that fails early also says something about the medium.
3. Wear and trend once more on the laps the life estimate says are cliff-free.

A team's tool, not the truth: the estimate is wrong early and noisy throughout; it is what a
reasonable average team could compute (editorial design; the prior spreads are measured).
"""

import math

import numpy as np

from f1ar.domain.race_state import (
    DRY_COMPOUNDS,
    CarState,
    LapModelParams,
    RaceSetup,
    TimingObservation,
    TyreEstimate,
    TyreKnowledge,
)
from f1ar.simulation.lap_model import expected_lap_time_s, noise_sd_s
from f1ar.simulation.weather import driver_factor, penalty_s

GRID_POINTS = 41
GRID_SPAN_SD = 4.0
MIN_COMPOUND_SD = 0.02
"""Before 2018 the compounds share one measured life (no per-compound surprise): a tiny
spread keeps the grid algebra well defined."""
Z_CAP = 4.0
"""Residuals beyond 4 sd count as 4 sd (a lap with an unseen hold-up or a VSC part)."""
DRY_PENALTY_S = 0.05
"""A lap counts as dry while the wet penalty on slicks is below this (the track's wetness is
rarely exactly 0); the small penalty is taken off the lap time."""
MATERIAL_CHANGE = 0.03
"""A new estimate is published to the teams (who then re-plan) only when a life or wear factor
moved by more than this share, or the trend by more than 0.005 s/lap (editorial)."""
NO_CLIFF_BELOW = 0.95
"""Life used under which a lap counts as cliff-free for the wear and trend fit."""


def observe(
    setup: RaceSetup,
    car: CarState,
    lap_time_s: float,
    *,
    neutralised: bool,
    wetness: float,
    phase_since_s: float = 0.0,
) -> TimingObservation | None:
    """The lap ``car`` just completed (state still at the lap's start) as the timing screen
    shows it, or None when it isn't a clean lap (or nobody is learning). ``phase_since_s``:
    when race control's current phase began (a lap that saw an SC/VSC end isn't clean)."""
    k = setup.knowledge
    if k is None or not k.learning:
        return None
    if car.compound not in DRY_COMPOUNDS:
        return None
    wet_s = penalty_s(setup.weather, car.compound, wetness) * driver_factor(
        setup.weather, setup.driver(car.driver_id).profile.archetype.wet_skill
    )
    if (
        neutralised
        or phase_since_s > car.lap_started_at_s
        or wet_s > DRY_PENALTY_S
        or car.laps_completed < 1  # lap 1: the launch
        or car.pit_hold_s > 0.0
        or car.traffic_loss_s > 0.0
        or car.incident_kind is not None
    ):
        return None
    entry = setup.driver(car.driver_id)
    driver = entry.profile.archetype
    prior = k.lap_model
    true_life = setup.lap_model.compounds[car.compound].life_laps
    prior_life = prior.compounds[car.compound].life_laps
    life_used = car.tyre_life_used
    if true_life is not None and prior_life is not None:
        life_used *= true_life / prior_life
    wear = car.tyre_wear_s * k.wear_scale
    laps_remaining = (
        setup.track.total_laps - car.laps_completed
        if car.fuel_laps is None
        else max(0, round(car.fuel_laps))
    )
    expected = expected_lap_time_s(
        prior,
        driver,
        compound=car.compound,
        tyre_wear_s=wear,
        pace_mode=car.lap_pace_mode,
        laps_remaining=laps_remaining,
        car_offset_s=entry.car_pace_offset_s + entry.driver_pace_offset_s,
        tyre_life_used=life_used,
        fresh_lap=car.tyre_age_laps == 1 and car.pit_stops > 0,
    )
    sd = entry.lap_noise_s if entry.lap_noise_s is not None else noise_sd_s(prior, driver)
    return TimingObservation(
        driver_id=car.driver_id,
        lap=car.laps_completed + 1,
        compound=car.compound,
        wear_s=wear,
        life_used=life_used,
        residual_s=lap_time_s - wet_s - expected,
        noise_sd_s=max(sd, 0.05),
    )


def _pad(form: np.ndarray, n: int) -> np.ndarray:
    return np.concatenate([form, np.zeros(n - len(form))]) if len(form) < n else form


def _cliff(per_lap: float, life_used: np.ndarray, life_laps: np.ndarray) -> np.ndarray:
    return np.asarray(per_lap * np.clip(life_used - 1.0, 0.0, None) * life_laps, dtype=float)


def _wear_and_trend(
    k: TyreKnowledge,
    car: np.ndarray,
    x: np.ndarray,
    lap: np.ndarray,
    r: np.ndarray,
    w: np.ndarray,
) -> tuple[float, float, float, np.ndarray]:
    """Bayesian ``r = beta x + tau lap + a_car``: (beta, sd of beta, tau, a per car)."""
    n_cars = int(car.max()) + 1 if len(car) else 0
    sw = np.bincount(car, w, n_cars)
    safe = np.where(sw > 0, sw, 1.0)
    xc = x - (np.bincount(car, w * x, n_cars) / safe)[car]
    lc = lap - (np.bincount(car, w * lap, n_cars) / safe)[car]
    rc = r - (np.bincount(car, w * r, n_cars) / safe)[car]
    a = np.array(
        [
            [np.sum(w * xc * xc) + 1.0 / k.wear_sd**2, np.sum(w * xc * lc)],
            [np.sum(w * xc * lc), np.sum(w * lc * lc) + 1.0 / max(k.trend_sd, 1e-4) ** 2],
        ]
    )
    b = np.array([np.sum(w * xc * rc), np.sum(w * lc * rc)])
    cov = np.linalg.inv(a)
    beta, tau = cov @ b
    form = np.bincount(car, w * (r - beta * x - tau * lap), n_cars) / safe
    return float(beta), float(math.sqrt(cov[0, 0])), float(tau), form


def estimate(setup: RaceSetup, timing: tuple[TimingObservation, ...]) -> TyreEstimate:
    """The posterior from every clean lap so far (pure; the prior when there are none)."""
    k = setup.knowledge
    assert k is not None
    prior = k.lap_model
    if not timing:
        return TyreEstimate(
            observations=0,
            wear_factor=1.0,
            wear_factor_sd=k.wear_sd,
            trend_s_per_lap=0.0,
            life_factor=dict.fromkeys(DRY_COMPOUNDS, 1.0),
            life_factor_sd=dict.fromkeys(DRY_COMPOUNDS, k.life_shared_sd),
        )
    ids = {d.driver_id: i for i, d in enumerate(setup.drivers)}
    car = np.array([ids[o.driver_id] for o in timing])
    x = np.array([o.wear_s for o in timing])
    lu = np.array([o.life_used for o in timing])
    lap = np.array([float(o.lap) for o in timing])
    r = np.array([o.residual_s for o in timing])
    w = 1.0 / np.array([o.noise_sd_s for o in timing]) ** 2
    comp = np.array([DRY_COMPOUNDS.index(o.compound) for o in timing])
    lives = np.array([prior.compounds[c].life_laps or 0.0 for c in DRY_COMPOUNDS], dtype=float)
    has_life = lives > 0
    sd_c = max(k.life_compound_sd, MIN_COMPOUND_SD)
    sd_s = max(k.life_shared_sd, MIN_COMPOUND_SD)
    span = GRID_SPAN_SD * math.hypot(sd_s, sd_c)
    u = np.linspace(-span, span, GRID_POINTS)
    u_hat = np.zeros(len(DRY_COMPOUNDS))
    life_mean = np.ones(len(DRY_COMPOUNDS))
    life_sd = np.full(len(DRY_COMPOUNDS), sd_s)
    beta, beta_sd, tau = 0.0, k.wear_sd, 0.0
    for _ in range(2):
        clean = (lu < NO_CLIFF_BELOW) & (lu * np.exp(-u_hat[comp]) < NO_CLIFF_BELOW)
        if clean.sum() >= 3:
            beta, beta_sd, tau, form = _wear_and_trend(
                k, car[clean], x[clean], lap[clean], r[clean], w[clean]
            )
            # robust: refit without laps beyond Z_CAP sd (an unseen hold-up)
            fit = beta * x + tau * lap + _pad(form, len(setup.drivers))[car]
            clean &= np.abs(r - fit) * np.sqrt(w) < Z_CAP
            if clean.sum() >= 3:
                beta, beta_sd, tau, form = _wear_and_trend(
                    k, car[clean], x[clean], lap[clean], r[clean], w[clean]
                )
        else:
            form = np.zeros(len(setup.drivers))
        form = _pad(form, len(setup.drivers))
        res = r - beta * x - tau * lap - form[car]
        prior_cliff = _cliff(prior.cliff_s_per_lap, lu, lives[comp])
        loglik = np.zeros((len(DRY_COMPOUNDS), GRID_POINTS))
        for ci in range(len(DRY_COMPOUNDS)):
            m = comp == ci
            if not has_life[ci] or not m.any():
                continue
            hyp = _cliff(
                prior.cliff_s_per_lap,
                lu[m][:, None] * np.exp(-u)[None, :],
                lives[ci] * np.exp(u)[None, :],
            )
            z = (res[m][:, None] - (hyp - prior_cliff[m][:, None])) * np.sqrt(w[m])[:, None]
            loglik[ci] = -0.5 * np.sum(np.minimum(z * z, Z_CAP**2), axis=0)
        # shared factor s on the same grid; compound c: u_c = s + d, d ~ N(0, sd_c)
        trans = -0.5 * ((u[None, :] - u[:, None]) / sd_c) ** 2  # [s, u]
        joint = trans[None, :, :] + loglik[:, None, :]  # [c, s, u]
        top = joint.max(axis=2, keepdims=True)
        per_s = top[..., 0] + np.log(np.exp(joint - top).sum(axis=2))  # [c, s] log L_c(s)
        per_s -= per_s.max(axis=1, keepdims=True)
        log_ps = -0.5 * (u / sd_s) ** 2 + per_s.sum(axis=0)
        for ci in range(len(DRY_COMPOUNDS)):
            # p(u_c) = sum_s p(s | others) p(u_c | s, own laps)
            others = log_ps - per_s[ci]
            p_s = np.exp(others - others.max())
            p_s /= p_s.sum()
            cond = joint[ci] - joint[ci].max(axis=1, keepdims=True)
            cond = np.exp(cond)
            cond /= cond.sum(axis=1, keepdims=True)
            p_u = p_s @ cond
            f = np.exp(u)
            mean = float(p_u @ f)
            life_mean[ci] = mean
            life_sd[ci] = float(math.sqrt(max(0.0, p_u @ (f - mean) ** 2)))
        u_hat = np.log(life_mean)
    return TyreEstimate(
        observations=len(timing),
        wear_factor=round(max(0.1, 1.0 + beta), 4),
        wear_factor_sd=round(beta_sd, 4),
        trend_s_per_lap=round(tau, 5),
        life_factor={c: round(float(life_mean[i]), 4) for i, c in enumerate(DRY_COMPOUNDS)},
        life_factor_sd={c: round(float(life_sd[i]), 4) for i, c in enumerate(DRY_COMPOUNDS)},
    )


def material(old: TyreEstimate | None, new: TyreEstimate) -> bool:
    """``new`` differs enough from the published ``old`` to publish it."""
    if old is None:
        return True
    pairs = [(old.wear_factor, new.wear_factor)] + [
        (old.life_factor[c], new.life_factor[c]) for c in new.life_factor
    ]
    moved = any(abs(b / a - 1.0) > MATERIAL_CHANGE for a, b in pairs)
    return moved or abs(new.trend_s_per_lap - old.trend_s_per_lap) > 0.005


def belief_model(k: TyreKnowledge, est: TyreEstimate) -> LapModelParams:
    """The prior's lap model with the estimate's life and wear factors (what teams plan with)."""
    from f1ar.simulation.race_day import scaled

    life = {c.value: est.life_factor.get(c, 1.0) for c in DRY_COMPOUNDS}
    return scaled(k.lap_model, life, est.wear_factor)
