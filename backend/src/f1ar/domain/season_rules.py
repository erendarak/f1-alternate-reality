"""The sporting rules of one season (H1, D-053, ``docs/PLAN_HIDDEN_INFO_ERAS.md`` §4).

A race setup carries its season's rules; the engine and the scoring read them instead of 2024's.
The catalogue (``configs/regulations/seasons.json``) lists every rule with its source;
``SEASON_2024`` is the default so worlds built before H1 keep their behaviour.

What each rule changes today:

- ``vsc``: without it (before 2015) a situation race control would answer with a VSC gets local
  double waved yellows, i.e. no neutralisation (the FIA designed the VSC for exactly the cases
  that needed double yellows but not a safety car); the safety-car share is unchanged (H2 refits
  the rates per era); a race director can't deploy one either.
- ``min_distinct_dry_compounds``: the dry-compound rule the engine and the scoring enforce.
- ``points``, ``fastest_lap_point``, ``double_points_final_round``: championship points.
- Stored for later slices, no effect yet: ``drs`` (H2 fits overtaking per era),
  ``top10_start_tyres`` (start tyres come from the real lap 1 where compounds are known),
  ``shortened_race_points`` (the engine doesn't shorten races), ``refuelling`` and
  ``tyre_changes_allowed`` (H1b, refuelling era), ``tyre_supplier`` (H2/H3 priors).
"""

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

CATALOGUE_FILE = Path("regulations") / "seasons.json"
"""Relative to ``configs/``."""


class SeasonRules(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    season: int = Field(ge=1950)
    points: tuple[int, ...]
    """Points for P1, P2, ... (a full-distance grand prix)."""
    fastest_lap_point: bool = False
    """One point for the fastest lap, if that driver is classified in the points (2019-2024)."""
    double_points_final_round: bool = False
    shortened_race_points: Literal["half_below_75pct", "sliding_scale_2022"] = "sliding_scale_2022"
    refuelling: bool = False
    tyre_changes_allowed: bool = True
    min_distinct_dry_compounds: int = Field(default=2, ge=0, le=3)
    top10_start_tyres: Literal["free", "q3_tyres", "q2_tyres"] = "free"
    tyre_supplier: str = "Pirelli"
    drs: bool = True
    vsc: bool = True
    race_fuel_from_qualifying: Literal["none", "all", "top10"] = "none"
    """H1b: who starts the race on the fuel fixed in qualifying (no fuel added before the race):
    everyone 2003-2005, the Q3 top ten 2006-2009."""
    pit_lane_closed_under_sc: bool = False
    """2007-2008: no stop until the field has queued behind the safety car (H1b)."""
    sources: tuple[str, ...] = ()

    def points_for(
        self, position: int, *, fastest_lap: bool = False, final_round: bool = False
    ) -> int:
        """Grand prix points of a classified finisher (``position`` 1-based)."""
        in_points = 1 <= position <= len(self.points)
        points = self.points[position - 1] if in_points else 0
        if fastest_lap and in_points and self.fastest_lap_point:
            points += 1
        if final_round and self.double_points_final_round:
            points *= 2
        return points


POINTS_2010 = (25, 18, 15, 12, 10, 8, 6, 4, 2, 1)

SEASON_2024 = SeasonRules(
    season=2024,
    points=POINTS_2010,
    fastest_lap_point=True,
    sources=("built-in default: the 2024 rules the engine was calibrated on",),
)


def load_catalogue(configs_dir: Path) -> dict[int, SeasonRules]:
    """Every season in ``configs/regulations/seasons.json``, by year."""
    data = json.loads((configs_dir / CATALOGUE_FILE).read_text(encoding="utf-8"))
    common = data.get("defaults", {})
    source = (f"configs/{CATALOGUE_FILE.as_posix()}",)
    out = {}
    for season, rules in data["seasons"].items():
        out[int(season)] = SeasonRules.model_validate(
            {**common, **rules, "season": int(season), "sources": source}
        )
    return out


def rules_for(configs_dir: Path, season: int) -> SeasonRules:
    catalogue = load_catalogue(configs_dir)
    if season not in catalogue:
        raise ValueError(f"no sporting rules for {season} in configs/{CATALOGUE_FILE.as_posix()}")
    return catalogue[season]
