"""Races the API can simulate: roster, personas, setup builder and drawable assets per event.

A race is listed when it has driver personas (its own seed or its season's, M4) *and* its
fetched data (outline, corners, colours, laps) is present locally. Setups are built lazily and
cached, since the data-derived inputs (car offsets, base lap, pit loss, tyre model) read the
lap table.
"""

import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from f1ar.domain.driver_profiles import ProfileSet
from f1ar.domain.identities import RaceRoster
from f1ar.domain.race_state import RaceSetup, TrackSpec
from f1ar.presentation.event_assets import EventAssets, load_event_assets, missing_assets
from f1ar.simulation.scenarios import (
    event_data_dir,
    has_personas,
    load_event,
    setup_from_codes,
    track_for,
)

SetupBuilder = Callable[[tuple[str, ...] | None, bool, bool], RaceSetup]
"""``(driver codes or None for the full grid, synthetic, random rain) -> RaceSetup``."""


class UnknownRaceError(KeyError):
    pass


def race_name(event_id: str) -> str:
    """``2024_italian_gp`` -> ``2024 Italian GP``."""
    season, *words = event_id.split("_")
    return " ".join([season, *("GP" if w == "gp" else w.capitalize() for w in words)])


@dataclass
class RaceSpec:
    race_id: str
    roster: RaceRoster
    profiles: ProfileSet
    assets: EventAssets
    build: SetupBuilder
    has_data: bool
    """Data-derived inputs available (else only synthetic setups)."""
    track: TrackSpec | None = None
    _cache: dict[tuple[tuple[str, ...] | None, bool, bool], RaceSetup] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def name(self) -> str:
        return race_name(self.race_id)

    def setup(
        self,
        codes: Sequence[str] | None = None,
        *,
        synthetic: bool = False,
        random_rain: bool = False,
    ) -> RaceSetup:
        key = (None if codes is None else tuple(c.upper() for c in codes), synthetic, random_rain)
        with self._lock:
            if key not in self._cache:
                self._cache[key] = self.build(key[0], key[1] or not self.has_data, key[2])
            return self._cache[key]

    def summary(self) -> dict[str, Any]:
        track = self.track
        if track is None:  # built without data: the published figures, if any
            try:
                track = track_for(self.race_id)
            except ValueError:
                track = None
        return {
            "race_id": self.race_id,
            "name": self.name,
            "season": self.roster.season,
            "round": self.roster.round,
            "circuit_id": self.roster.circuit_id,
            "race_date": self.roster.race_date,
            "total_laps": None if track is None else track.total_laps,
            "length_m": None if track is None else track.length_m,
            "driver_count": len(self.roster.entries),
            "has_data": self.has_data,
        }

    def drivers(self) -> list[dict[str, Any]]:
        """The verified entry list (historical_verified), grid order."""
        colors = self.assets.team_colors
        return [
            {
                "driver_id": e.driver_id,
                "code": e.code,
                "display_name": e.display_name,
                "car_number": e.car_number,
                "team_id": e.team_id,
                "team_name": e.team_name,
                "grid_position": e.grid_position,
                "team_color": colors.get(e.driver_id),
                "source_type": self.roster.source_type.value,
            }
            for e in sorted(self.roster.entries, key=lambda e: (e.grid_position or 99, e.code))
        ]


class RaceCatalog:
    def __init__(self, races: Sequence[RaceSpec]) -> None:
        self._races = {r.race_id: r for r in races}

    def get(self, race_id: str) -> RaceSpec:
        try:
            return self._races[race_id]
        except KeyError:
            raise UnknownRaceError(race_id) from None

    def __contains__(self, race_id: object) -> bool:
        return race_id in self._races

    def all(self) -> list[RaceSpec]:
        return list(self._races.values())


def load_catalog(root: Path) -> tuple[RaceCatalog, dict[str, str]]:
    """Every race with personas whose data (with the track outline) has been fetched under
    ``root``, in round order; plus why the others are missing (event id -> reason)."""
    races: list[RaceSpec] = []
    skipped: dict[str, str] = {}
    configs = root / "configs"
    events = sorted(p.stem for p in (configs / "rosters").glob("*.json"))
    for event_id in events:
        if not has_personas(configs, event_id):
            continue
        data = root / "data" / "processed" / event_id
        missing = missing_assets(data)
        if missing:
            skipped[event_id] = f"not fetched with telemetry (missing {', '.join(missing)})"
            continue
        roster, profiles = load_event(configs, event_id)
        track = track_for(event_id, data)
        assets = load_event_assets(roster, data, engine_length_m=track.length_m)
        data_dir = event_data_dir(root, event_id)

        def build(
            codes: tuple[str, ...] | None,
            synthetic: bool,
            random_rain: bool,
            roster: RaceRoster = roster,
            profiles: ProfileSet = profiles,
            data_dir: Path | None = data_dir,
        ) -> RaceSetup:
            return setup_from_codes(
                roster,
                profiles,
                codes,
                data_dir=None if synthetic else data_dir,
                random_rain=random_rain and not synthetic,
            )

        races.append(
            RaceSpec(event_id, roster, profiles, assets, build, data_dir is not None, track=track)
        )
    races.sort(key=lambda r: (r.roster.season, r.roster.round))
    return RaceCatalog(races), skipped
