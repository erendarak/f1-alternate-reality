"""Refuelling-era constants (H1b, D-057). EDITORIAL ASSUMPTIONS, one value for every track:

- ``FUEL_KG_PER_LAP`` 2.75 kg: a 2000s V8/V10 burned roughly 2.5-3 kg a lap (rule of thumb; not
  in our data, which has no pit-stop or fuel records before 2011);
- ``FUEL_S_PER_KG`` 0.03 s: the commonly quoted ~0.3 s per 10 kg of fuel (rule of thumb);
- ``REFUEL_KG_PER_S`` 9.0 kg/s: the FIA's 12.1 l/s flow cap x ~0.745 kg/l fuel density
  (formula1-dictionary.net/f1-refuelling), giving the era's 6-12 s stops for 20-35 laps of fuel.

Fuel is counted in **laps of fuel**: the lap model's fuel term is ``FUEL_S_PER_LAP x laps of
fuel on board``, a stop puts in ``REFUEL_S_PER_LAP`` seconds per lap of fuel.
"""

FUEL_KG_PER_LAP = 2.75
FUEL_S_PER_KG = 0.03
REFUEL_KG_PER_S = 12.1 * 0.745

FUEL_S_PER_LAP = FUEL_S_PER_KG * FUEL_KG_PER_LAP
"""Lap time per lap of fuel on board (0.0825 s)."""
REFUEL_S_PER_LAP = FUEL_KG_PER_LAP / REFUEL_KG_PER_S
"""Stationary seconds per lap of fuel put in (0.305 s)."""
LAST_REFUELLING_SEASON = 2009
"""Refuelling was banned from 2010 (configs/regulations/seasons.json)."""


def fuel_on_board(stint_end_lap: int, lap_number: int) -> int:
    """Laps of fuel when a lap starts: a stop puts in exactly the stint's fuel (H1b)."""
    return max(1, stint_end_lap - lap_number + 1)
