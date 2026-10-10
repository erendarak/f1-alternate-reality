"""Print the 2024 Monza starting grid with each driver's Stage 1 persona values.

Usage:
    uv run python scripts/show_profiles.py
"""

from pathlib import Path

from f1ar.domain.driver_profiles import STAGE1_FIELDS, load_profiles
from f1ar.domain.identities import load_roster

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
EVENT_ID = "2024_italian_gp"
SHORT = {
    "race_pace_skill": "pace",
    "consistency": "cons",
    "risk_appetite": "risk",
    "tire_management": "tyre",
    "wet_skill": "wet",
    "adaptability": "adapt",
}


def main() -> None:
    roster = load_roster(CONFIGS / "rosters" / f"{EVENT_ID}.json", expected_event_id=EVENT_ID)
    profiles = load_profiles(
        CONFIGS / "drivers" / "seed_2024_monza_drivers.yaml",
        CONFIGS / "drivers" / "archetypes.yaml",
        roster,
    )

    print(f"{roster.season} round {roster.round} - {roster.circuit_id} ({roster.race_date})")
    print(f"roster: {roster.source.name}, retrieved {roster.source.retrieved_utc}")
    print(f"profiles: {profiles.manifest_hash[:23]}...  (persona values = editorial assumptions)\n")

    header = "grid  #   code team              " + " ".join(f"{SHORT[f]:>5}" for f in STAGE1_FIELDS)
    print(header + "  conf  template          persona")
    print("-" * (len(header) + 48))
    for entry in roster.entries:
        p = profiles.get(entry.driver_id)
        values = " ".join(f"{getattr(p.archetype, f):>5.2f}" for f in STAGE1_FIELDS)
        grid = f"P{entry.grid_position:<3}" if entry.grid_position else "PIT "
        print(
            f"{grid}  {entry.car_number:<3} {entry.code}  {entry.team_name:<17} {values}"
            f"  {p.metadata.confidence:.2f}  {p.metadata.base_template:<17} "
            f"{p.metadata.persona_label or '-'}"
        )


if __name__ == "__main__":
    main()
