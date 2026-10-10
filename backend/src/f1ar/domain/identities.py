"""Race-specific roster snapshots (plan §3.3, §5.2, §12.3).

A roster belongs to one event, not to a season: drivers changed seats during 2024
(e.g. Colapinto debuted at Monza), so a Monza roster must never be reused for another
race automatically. Identity data here is ``historical_verified`` and carries its source.
Persona/skill numbers do NOT live here; see ``driver_profiles``.
"""

import json
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from f1ar.domain.provenance import SourceType


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class RosterSource(_Frozen):
    name: str
    url: str
    retrieved_utc: str
    license_review_status: str


class RosterEntry(_Frozen):
    driver_id: str = Field(pattern=r"^[a-z0-9_]+_\d{4}$")
    """Season-specific entry id, e.g. ``max_verstappen_2024``."""
    person_id: str = Field(pattern=r"^[a-z0-9_]+$")
    """Stable personal identity across seasons, e.g. ``max_verstappen``."""
    code: str = Field(pattern=r"^[A-Z]{3}$")
    car_number: int = Field(ge=0, le=99)
    given_name: str
    family_name: str
    team_id: str
    team_name: str
    grid_position: int = Field(ge=0)
    """Official starting grid slot; 0 means pit-lane start."""
    external_ids: dict[str, str]

    @property
    def display_name(self) -> str:
        return f"{self.given_name} {self.family_name}"


class RaceRoster(_Frozen):
    schema_version: Literal[1]
    event_id: str
    season: int
    round: int
    circuit_id: str
    race_date: str
    source_type: Literal[SourceType.HISTORICAL_VERIFIED]
    source: RosterSource
    entries: tuple[RosterEntry, ...]

    @model_validator(mode="after")
    def _check_consistency(self) -> Self:
        ids = [e.driver_id for e in self.entries]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate driver_id in roster")
        numbers = [e.car_number for e in self.entries]
        if len(set(numbers)) != len(numbers):
            raise ValueError("duplicate car_number in roster")
        for e in self.entries:
            if not e.driver_id.endswith(f"_{self.season}"):
                raise ValueError(f"{e.driver_id} does not belong to season {self.season}")
        grid = sorted(e.grid_position for e in self.entries if e.grid_position > 0)
        if len(set(grid)) != len(grid):
            raise ValueError("two cars share a grid slot")
        return self

    def driver_ids(self) -> tuple[str, ...]:
        return tuple(e.driver_id for e in self.entries)

    def entry(self, driver_id: str) -> RosterEntry:
        for e in self.entries:
            if e.driver_id == driver_id:
                return e
        raise KeyError(driver_id)


def load_roster(path: Path, *, expected_event_id: str) -> RaceRoster:
    """Load a roster file, refusing one that belongs to a different event."""
    roster = RaceRoster.model_validate(json.loads(path.read_text(encoding="utf-8")))
    if roster.event_id != expected_event_id:
        raise ValueError(
            f"roster {path.name} is for {roster.event_id!r}, not {expected_event_id!r}; "
            "rosters are race-specific and are never reused across events"
        )
    return roster
