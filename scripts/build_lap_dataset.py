"""Build the season lap dataset, its splits and the season reference metrics (M4 slice 1).

Usage:
    uv run python scripts/fetch_season.py --season 2024        # once
    uv run python scripts/build_lap_dataset.py --season 2024

Writes (data is gitignored, manifests and reports are tracked):
    data/processed/datasets/laps_<season>_v<N>.parquet     one row per lap (lap_dataset.py)
    data/manifests/dataset_laps_<season>_v<N>.yaml           hashes, schema, splits, sources
    docs/data/season_<season>_reference.md / .json          real-race metrics per race
"""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from f1ar.calibration.lap_dataset import (
    DATASET_VERSION,
    FEATURE_SCHEMA_VERSION,
    FEATURES,
    LAP_FACTS,
    TARGET,
    file_sha256,
    is_wet_race,
    make_splits,
    race_from_dir,
    season_events,
)
from f1ar.calibration.reference_metrics import real_metrics

ROOT = Path(__file__).resolve().parents[1]


def content_sha256(df: pd.DataFrame) -> str:
    """Hash of the values (stable across parquet writer versions, unlike the file bytes)."""
    row_hashes = pd.util.hash_pandas_object(df, index=False).to_numpy()
    return "sha256:" + hashlib.sha256(row_hashes.tobytes()).hexdigest()


def quality_failures(event_id: str) -> list[str]:
    """The FAIL checks of a race's quality report (``docs/data/<event>_quality.md``)."""
    report = ROOT / "docs" / "data" / f"{event_id}_quality.md"
    if not report.exists():
        return []
    rows = report.read_text(encoding="utf-8").splitlines()
    return [r.split("|")[2].strip() for r in rows if r.startswith("| FAIL |")]


def race_reference(data_dir: Path, race: pd.DataFrame, wet: bool) -> dict[str, Any]:
    m = real_metrics(data_dir)
    status = pd.read_parquet(data_dir / "track_status.parquet")["status"].astype(str).to_list()
    red = sum(1 for i, c in enumerate(status) if c == "5" and (i == 0 or status[i - 1] != "5"))
    stops = list(m.stops.values())
    compounds = race["compound"].dropna().astype(str)
    return {
        "laps": int(race["scheduled_laps"].iloc[0]),
        "wet": wet,
        "finishers": len(m.finishers),
        "stops_per_finisher": round(sum(stops) / len(stops), 2) if stops else None,
        "one_stop_share": round(sum(s == 1 for s in stops) / len(stops), 2) if stops else None,
        "two_plus_stop_share": round(sum(s >= 2 for s in stops) / len(stops), 2) if stops else None,
        # ties broken alphabetically: Counter order would follow set iteration (not stable)
        "top_sequences": dict(
            sorted(Counter(m.sequences.values()).items(), key=lambda kv: (-kv[1], kv[0]))[:3]
        ),
        "overtakes": m.overtakes,
        "retirements": m.retirements,
        "safety_cars": m.safety_cars,
        "vscs": m.vscs,
        "red_flags": red,
        "lapped_finishers": m.lapped,
        "winner": m.winner,
        "winner_time_min": round(m.winner_time_s / 60, 2),
        "wet_tyre_lap_share": round(float(compounds.isin(["INTERMEDIATE", "WET"]).mean()), 3),
        "rain_at_lap_start_share": round(
            float(race["rainfall_start"].fillna(False).astype(bool).mean()), 3
        ),
        "representative_laps": int(race["representative"].sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--season", type=int, default=2024)
    args = parser.parse_args()

    events = season_events(ROOT, args.season)
    if not events:
        parser.error(f"no fetched {args.season} races; run scripts/fetch_season.py first")
    races: list[pd.DataFrame] = []
    rounds: dict[str, int] = {}
    wet: set[str] = set()
    sources: dict[str, str] = {}
    reference: dict[str, dict[str, Any]] = {}
    excluded: dict[str, str] = {}
    for event_id, data_dir, roster_path in events:
        failed = quality_failures(event_id)
        if failed:  # H2a (D-054): e.g. results declared before the laps run, feed gaps
            excluded[event_id] = "; ".join(failed)
            print(f"  excluded {event_id:<28} quality FAIL: {excluded[event_id][:80]}")
            continue
        race = race_from_dir(data_dir, roster_path)
        rounds[event_id] = int(race["round"].iloc[0])
        if is_wet_race(race):
            wet.add(event_id)
        source_manifest = ROOT / "data" / "manifests" / f"{event_id}_fastf1.yaml"
        sources[event_id] = file_sha256(source_manifest)
        reference[event_id] = race_reference(data_dir, race, event_id in wet)
        races.append(race)
        print(
            f"  round {rounds[event_id]:>2} {event_id:<28} {len(race):>5} laps"
            f"{'  WET' if event_id in wet else ''}"
        )

    splits = make_splits(rounds, wet)
    data = pd.concat(races, ignore_index=True)
    data["group"] = data["event_id"].map(splits.group_of)

    name = f"laps_{args.season}_v{DATASET_VERSION}"
    out = ROOT / "data" / "processed" / "datasets" / f"{name}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    data.to_parquet(out, index=False)

    by_event = data.groupby("event_id")
    manifest = {
        "dataset_id": name,
        "dataset_version": DATASET_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "season": args.season,
        "license_review_status": "research_only (F1 timing data via FastF1; not redistributed)",
        "built_by": "scripts/build_lap_dataset.py",
        "file": {
            "relative_path": out.relative_to(ROOT).as_posix(),
            "sha256": file_sha256(out),
            "content_sha256": content_sha256(data),
            "rows": len(data),
            "representative_rows": int(data["representative"].sum()),
        },
        "target": TARGET,
        "features": [
            {
                "name": f.name,
                "available": f.available,
                "source_type": f.source_type.value,
                "description": f.description,
            }
            for f in FEATURES
        ],
        "lap_facts_never_features": list(LAP_FACTS),
        "leakage_audit": (
            "tests/unit/test_lap_dataset.py::test_no_feature_uses_anything_after_the_lap_starts"
        ),
        "excluded_quality_fail": excluded,
        "splits": {
            "rule": splits.rule,
            "wet": list(splits.wet),
            "dry_test": list(splits.dry_test),
            "dry_train": list(splits.dry_train),
        },
        "events": {
            str(e): {
                "round": rounds[str(e)],
                "group": splits.group_of(str(e)),
                "rows": int(len(g)),
                "representative_rows": int(g["representative"].sum()),
                "source_manifest_sha256": sources[str(e)],
            }
            for e, g in by_event
        },
    }
    manifest_path = ROOT / "data" / "manifests" / f"dataset_{name}.yaml"
    manifest_path.write_text(
        yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True, width=100),
        encoding="utf-8",
        newline="\n",
    )

    ref_json = ROOT / "docs" / "data" / f"season_{args.season}_reference.json"
    ref_json.write_text(
        json.dumps(reference, indent=2, sort_keys=False) + "\n", encoding="utf-8", newline="\n"
    )
    ref_md = ROOT / "docs" / "data" / f"season_{args.season}_reference.md"
    ref_md.write_text(
        reference_markdown(args.season, reference, rounds, splits.group_of),
        encoding="utf-8",
        newline="\n",
    )

    print(
        f"\n{len(data)} laps ({int(data['representative'].sum())} representative) from "
        f"{len(events)} races -> {out.relative_to(ROOT)}"
    )
    print(
        f"splits: wet {list(splits.wet)}\n        dry_test {list(splits.dry_test)}\n"
        f"        dry_train {len(splits.dry_train)} races"
    )
    print(f"manifest -> {manifest_path.relative_to(ROOT)}\nreference -> {ref_md.relative_to(ROOT)}")


def reference_markdown(
    season: int, ref: dict[str, dict[str, Any]], rounds: dict[str, int], group_of: Any
) -> str:
    lines = [
        f"# {season} season reference metrics (real races)",
        "",
        "Generated by `scripts/build_lap_dataset.py` from F1 timing data via FastF1 (derived",
        "counts, not raw data). Overtakes, stops and SC/VSC are counted exactly as EXP-002",
        "counts them in simulated races (`f1ar/calibration/reference_metrics.py`), so real and",
        "simulated values are comparable. These are the targets of M4's calibration.",
        "",
        "| Rd | Race | Group | Laps | Stops/fin | 1-stop | 2+ stops | Top sequences | Passes |"
        " Ret | SC | VSC | Red | Wet laps | Winner |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for e in sorted(ref, key=lambda x: rounds[x]):
        r = ref[e]
        seqs = ", ".join(f"{k} {v}" for k, v in r["top_sequences"].items())
        lines.append(
            f"| {rounds[e]} | {e} | {group_of(e)} | {r['laps']} | {r['stops_per_finisher']} | "
            f"{r['one_stop_share']} | {r['two_plus_stop_share']} | {seqs} | {r['overtakes']} | "
            f"{r['retirements']} | {r['safety_cars']} | {r['vscs']} | {r['red_flags']} | "
            f"{r['wet_tyre_lap_share']:.0%} | {r['winner']} |"
        )
    dry = [r for r in ref.values() if not r["wet"]]
    if dry:

        def mean(key: str) -> float:
            vals = [float(r[key]) for r in dry if r[key] is not None]
            return round(sum(vals) / len(vals), 2)

        lines += [
            "",
            f"Dry-race means ({len(dry)} races): stops/finisher {mean('stops_per_finisher')}, "
            f"one-stop share {mean('one_stop_share')}, passes {mean('overtakes')}, "
            f"retirements {mean('retirements')}, safety cars {mean('safety_cars')}, "
            f"VSCs {mean('vscs')}.",
        ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
