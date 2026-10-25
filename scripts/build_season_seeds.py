"""Write a persona seed per season (H2a, D-054): configs/drivers/seed_<season>_season_drivers.yaml.

Every value is an EDITORIAL ASSUMPTION, like the 2024 seed. A driver who also raced in 2024
keeps the persona of the 2024 seed (a persona belongs to the person; it is not refitted for the
season here); any other driver gets a template: ``generic_rookie`` without a start in any earlier
season we have (rosters, Jolpica history 2000+, Jolpica driver lists 1990-1999), else
``generic_veteran``. The W5 refit then replaces the fields a season's data can measure, once
that season is fitted.

Usage:
    uv run python scripts/build_season_seeds.py --seasons 2018 2019 2020 2021 2022 2023 2025
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
HISTORY = ROOT / "data" / "processed" / "history"
RAW = ROOT / "data" / "raw" / "jolpica"
FIRST_DRIVER_LIST = 1990
"""Drivers who raced in 1990-1999 aren't rookies in 2000 (season driver lists, not results)."""
REFERENCE = CONFIGS / "drivers" / "seed_2024_season_drivers.yaml"


def starts_by_season() -> dict[int, set[str]]:
    """Jolpica driver ids with a race start, per season (rosters and history results)."""
    out: dict[int, set[str]] = defaultdict(set)
    for path in (CONFIGS / "rosters").glob("*.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        for e in data["entries"]:
            out[int(data["season"])].add(e["external_ids"]["jolpica_driver_id"])
    for folder in HISTORY.glob("*_*"):
        results = folder / "results.parquet"
        if results.exists():
            out[int(folder.name.split("_")[0])] |= set(pd.read_parquet(results)["driver"])
    # before our race data (2000): Jolpica's season driver lists, cached (10 requests once)
    from f1ar.data.jolpica_adapter import BASE_URL
    from f1ar.data.jolpica_history import Throttle, cached, get_json

    throttle = Throttle(9.0)
    for year in range(FIRST_DRIVER_LIST, 2000):
        url = f"{BASE_URL}/{year}/drivers.json?limit=100"

        def fetch(u: str = url) -> Any:
            return get_json(u, throttle)

        page = cached(RAW / str(year) / "drivers.json", fetch)
        out[year] |= {d["driverId"] for d in page["MRData"]["DriverTable"]["Drivers"]}
    return out


def season_drivers(season: int) -> dict[str, dict[str, Any]]:
    """driver_id -> roster entry, over every race of the season."""
    out: dict[str, dict[str, Any]] = {}
    for path in sorted((CONFIGS / "rosters").glob(f"{season}_*.json")):
        for e in json.loads(path.read_text(encoding="utf-8"))["entries"]:
            out.setdefault(e["driver_id"], e)
    return out


def build(season: int, reference: dict[str, Any], starts: dict[int, set[str]]) -> str:
    by_person = {d.rsplit("_", 1)[0]: v for d, v in reference["drivers"].items()}
    earlier = set().union(*(ids for s, ids in starts.items() if s < season))
    lines = [
        f"# Driver persona seed for the {season} season (H2a, D-054), written by",
        "# scripts/build_season_seeds.py. Identity comes from each race's roster (Jolpica).",
        "# THE NUMBERS ARE EDITORIAL ASSUMPTIONS: illustrative simulation personas, not verified",
        "# psychometrics. Drivers who raced in 2024 keep their 2024 seed persona (not refitted for",
        f"# {season}); others are template-only (rookie = no start in an earlier season we have).",
        "",
    ]
    drivers: dict[str, Any] = {}
    for driver_id, entry in season_drivers(season).items():
        person = entry["person_id"]
        if person in by_person:
            src = dict(by_person[person])
            src["origin"] = f"copied from the 2024 seed ({src.get('origin', 'template only')})"
            src["confidence"] = min(float(src.get("confidence", 0.15)), 0.15)
            drivers[driver_id] = src
        else:
            rookie = entry["external_ids"]["jolpica_driver_id"] not in earlier
            drivers[driver_id] = {
                "template": "generic_rookie" if rookie else "generic_veteran",
                "confidence": 0.1,
                "origin": "template only"
                + (" (no start before this season in our data)" if rookie else ""),
            }
    seed = {
        "schema_version": 1,
        "season": season,
        "event_id": f"{season}_season",
        "explanation": "Illustrative simulation persona; not verified driver psychometrics.",
        "profile_revision": "v1",
        "policy": reference["policy"],
        "drivers": dict(sorted(drivers.items())),
    }
    return "\n".join(lines) + yaml.safe_dump(seed, sort_keys=False, allow_unicode=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seasons", type=int, nargs="+", required=True)
    args = parser.parse_args()
    reference = yaml.safe_load(REFERENCE.read_text(encoding="utf-8"))
    starts = starts_by_season()
    for season in args.seasons:
        if season == 2024:
            print("2024: kept (the reference seed)")
            continue
        out = CONFIGS / "drivers" / f"seed_{season}_season_drivers.yaml"
        text = build(season, reference, starts)
        out.write_text(text, encoding="utf-8", newline="\n")
        seed = yaml.safe_load(text)["drivers"]
        copied = sum("copied" in v["origin"] for v in seed.values())
        rookies = sum(v.get("template") == "generic_rookie" for v in seed.values())
        print(
            f"{season}: {len(seed)} drivers ({copied} from 2024, {rookies} rookies) -> {out.name}"
        )


if __name__ == "__main__":
    main()
