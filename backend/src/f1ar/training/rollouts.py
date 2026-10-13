"""Imagined futures: what a strategist may simulate at decision time (M5 slice 2, D-039).

A strategist that looks ahead (the MPC baseline B3, the enumerator's re-ranking B2) may use the
engine, **but not the real future**:

- **Other random draws:** the imagined world gets a new seed (lap noise, incidents, safety cars
  differ from the real race). Same imagined seed for every option compared = common random
  numbers, so the comparison isn't drowned in noise.
- **No real future weather:** a race replaying real weather would hand the strategist the
  actual rain timeline. Each imagined future instead draws its own rain timeline from what the
  team knows now: the forecast (US-06: noisy, 10-minute horizon) and the current rain.
  Not raining: with the forecast's probability, rain starts at a random moment within the
  horizon. Raining: it keeps raining for a random time. Every shower lasts an exponential time
  (editorial mean, the random-rain stop rate). **Beyond the horizon the conditions persist**
  (no new rain): a naive persistence forecast, since the team has no longer forecast. The
  track's wetness then follows the fitted wetting/drying dynamics from its current value.
  (First version, found by a test before EXP-008 finished: the 10-minute forecast became a
  constant rain rate for the whole rest of the race, so dry races imagined showers in almost
  every future.)
- **Known residue:** the lap in progress keeps the time the engine already drew for it (and
  any incident drawn at its start); imagined draws start from the next lap.

After the option is applied, the car keeps it until its next line crossing, then the rule
planner drives it to the end (a one-step look-ahead with the baseline as the base policy).
"""

import math
from collections.abc import Mapping

from f1ar.domain.race_state import (
    CarStatus,
    SetStrategyController,
    StrategyController,
    WeatherMode,
    WeatherParams,
    WorldState,
    WorldStatus,
)
from f1ar.simulation import lightweight_engine as engine
from f1ar.simulation.rng_streams import RngSpec, uniform
from f1ar.simulation.weather import forecast_rain_probability
from f1ar.training.actions import agent_commands
from f1ar.training.objectives import Scoring, race_results
from f1ar.training.session import MAX_TICKS

RANDOM_RAIN_STOP_PER_MIN = 0.08
"""Editorial: a shower lasts 1/0.08 = 12.5 min on average (the app's "random showers" rate)."""


def imagined_seed(real_seed: int, driver_id: str, lap: int, k: int) -> int:
    """A seed for imagined future ``k`` (never the real one)."""
    seed = int(uniform(real_seed, "imagined_future", driver_id, lap, k) * (2**31 - 2)) + 1
    return seed if seed != real_seed else seed + 1


def imagined_weather(state: WorldState, seed: int) -> WeatherParams:
    """A rain timeline for one imagined future, from the forecast and the rain now only."""
    weather = state.setup.weather
    now = state.sim_clock_s
    raining = state.weather.raining
    forecast = forecast_rain_probability(weather, state.rng.seed, now, raining)
    mean_s = 60.0 / RANDOM_RAIN_STOP_PER_MIN

    def u(name: str) -> float:
        return uniform(seed, "imagined_rain", name)

    timeline: list[tuple[float, bool]] = [(now - 1.0, raining)]
    if raining:
        timeline.append((now - math.log(1.0 - u("stop")) * mean_s, False))
    elif u("starts") < forecast:
        start = now + u("start_at") * weather.forecast_horizon_s
        timeline += [(start, True), (start - math.log(1.0 - u("stop")) * mean_s, False)]
    return weather.model_copy(
        update={
            "mode": WeatherMode.HISTORICAL,
            "rain_timeline": tuple(timeline),
            "wetness_timeline": (),  # wetness follows the dynamics from its value now
        }
    )


def imagined_world(state: WorldState, seed: int) -> WorldState:
    imagined = imagined_weather(state, seed)
    return state.model_copy(
        update={
            "rng": RngSpec(seed=seed),
            "setup": state.setup.model_copy(update={"weather": imagined}),
        }
    )


def rollout_reward(
    state: WorldState,
    driver_id: str,
    action: int,
    seed: int,
    scoring: Mapping[str, Scoring],
    actor_id: str,
) -> float:
    """``driver_id``'s score if it takes ``action`` now and the rule planner drives it (and every
    other AI car) from its next line crossing on, in the imagined future ``seed``."""
    world = imagined_world(state, seed)
    for car in world.cars:
        if car.controller is StrategyController.AGENT and car.driver_id != driver_id:
            world = engine.apply_action(
                world,
                SetStrategyController(
                    driver_id=car.driver_id,
                    controller=StrategyController.BASELINE,
                    actor_id=actor_id,
                ),
            ).state
    for command in agent_commands(world.car(driver_id), action, actor_id):
        world = engine.apply_action(world, command).state
    # the option holds until the car's next line crossing; then the rule planner takes over
    lap = world.car(driver_id).laps_completed
    ticks = 0
    while world.world_status is WorldStatus.RUNNING:
        world = engine.step(world).state
        ticks += 1
        if ticks > MAX_TICKS:
            raise RuntimeError("imagined race did not finish (tick safety net)")
        me = world.car(driver_id)
        if (
            me.controller is StrategyController.AGENT
            and me.status is CarStatus.RUNNING
            and me.laps_completed > lap
        ):
            world = engine.apply_action(
                world,
                SetStrategyController(
                    driver_id=driver_id,
                    controller=StrategyController.BASELINE,
                    actor_id=actor_id,
                ),
            ).state
    results = race_results(world)
    value = scoring[driver_id].reward(driver_id, results, len(world.setup.drivers))
    return value if math.isfinite(value) else -1.0
