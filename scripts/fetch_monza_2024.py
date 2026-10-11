"""Fetch the 2024 Italian GP race via FastF1; write Parquet, a manifest and a quality report.

Usage:
    uv run python scripts/fetch_monza_2024.py                 # cached after the first run
    uv run python scripts/fetch_monza_2024.py --no-telemetry  # skip ~600k position samples
    uv run python scripts/fetch_monza_2024.py --verify        # only check files vs the manifest

Outputs:
    data/processed/2024_italian_gp/*.parquet   (gitignored: source data, licence-restricted)
    data/manifests/2024_italian_gp_fastf1.yaml  (tracked: hashes, rows, units, provenance)
    docs/data/2024_italian_gp_quality.md        (tracked: generated data-quality report)
"""

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

import fastf1
import yaml

from f1ar.data import jolpica_adapter
from f1ar.data.fastf1_adapter import load_race_session, tables_from_session, track_outline
from f1ar.data.manifests import DatasetManifest, verify_files, write_table
from f1ar.data.quality import build_quality_report, render_markdown
from f1ar.domain.provenance import SourceType
from f1ar.simulation.scenarios import MONZA_2024, load_event, monza_2024_track

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".cache" / "fastf1"
OUT = ROOT / "data" / "processed" / MONZA_2024
MANIFEST = ROOT / "data" / "manifests" / f"{MONZA_2024}_fastf1.yaml"
REPORT = ROOT / "docs" / "data" / f"{MONZA_2024}_quality.md"
OUTLINE_LAP = 30  # a mid-race green lap of the winner, for the 2D track outline


def _verify() -> int:
    manifest = DatasetManifest.model_validate(yaml.safe_load(MANIFEST.read_text(encoding="utf-8")))
    results = verify_files(manifest, ROOT)
    for path, ok in results.items():
        print(f"  {'OK ' if ok else 'BAD'} {path}")
    return 0 if all(results.values()) else 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch + validate Monza 2024 race data")
    parser.add_argument("--no-telemetry", action="store_true")
    parser.add_argument("--skip-jolpica", action="store_true", help="no independent cross-check")
    parser.add_argument("--verify", action="store_true", help="only verify files vs manifest")
    args = parser.parse_args()
    if args.verify:
        sys.exit(_verify())

    roster, _ = load_event(ROOT / "configs")
    retrieved = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    session = load_race_session(
        roster.season, "Italian Grand Prix", CACHE, telemetry=not args.no_telemetry
    )
    tables = tables_from_session(session)

    reference = None
    if not args.skip_jolpica:
        url = jolpica_adapter.results_url(roster.season, roster.round)
        reference = jolpica_adapter.result_records(jolpica_adapter.fetch_json(url))

    hist = SourceType.HISTORICAL_VERIFIED
    files = [
        write_table(
            tables.laps,
            OUT / "laps.parquet",
            root=ROOT,
            source_type=hist,
            description="one row per car per lap: times, sectors, tyres, stint, position",
        ),
        write_table(
            tables.results,
            OUT / "results.parquet",
            root=ROOT,
            source_type=hist,
            description="official classification incl. grid, status, laps",
        ),
        write_table(
            tables.weather,
            OUT / "weather.parquet",
            root=ROOT,
            source_type=hist,
            description="trackside weather samples (~1/min)",
        ),
        write_table(
            tables.track_status,
            OUT / "track_status.parquet",
            root=ROOT,
            source_type=hist,
            description="track status changes (green/SC/VSC/red)",
        ),
        write_table(
            tables.race_control,
            OUT / "race_control.parquet",
            root=ROOT,
            source_type=hist,
            description="race control messages",
        ),
    ]
    if not tables.positions.empty:
        winner = int(tables.results.iloc[0]["driver_number"])
        outline = track_outline(
            tables.positions, tables.laps, driver_number=winner, lap_number=OUTLINE_LAP
        )
        files += [
            write_table(
                tables.positions,
                OUT / "positions.parquet",
                root=ROOT,
                source_type=hist,
                description="car X/Y/Z location samples in metres (~4 Hz)",
            ),
            write_table(
                outline,
                OUT / "track_outline.parquet",
                root=ROOT,
                source_type=SourceType.ESTIMATED_FROM_DATA,
                description=f"derived: car {winner} lap {OUTLINE_LAP} path as track outline",
            ),
        ]

    report = build_quality_report(
        tables,
        roster,
        source={
            "source": "FastF1",
            "source_version": fastf1.__version__,
            "retrieved_utc": retrieved,
            "reference": "Jolpica-F1 results" if reference is not None else "none",
        },
        expected_total_laps=monza_2024_track().total_laps,
        reference_results=reference,
    )
    manifest = DatasetManifest(
        dataset_id=f"{MONZA_2024}_fastf1_v1",
        event=MONZA_2024,
        source="fastf1",
        source_api_version=fastf1.__version__,
        retrieved_utc=retrieved,
        session_key=f"{roster.season}/round {roster.round}/R",
        license_review_status="research_only",
        files=files,
        unit_conventions={
            "time": "seconds (float); *_time_s columns are session time",
            "distance": "metres (positions converted from FastF1 1/10 m)",
            "temperature": "celsius",
            "speed": "km/h (speed_trap_kph), m/s (wind_speed_ms)",
        },
        missingness_notes=[
            f"lap table `{k}` missing in {v} rows" for k, v in report.lap_missingness.items()
        ],
    )
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(manifest.to_yaml(), encoding="utf-8", newline="\n")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(render_markdown(report), encoding="utf-8", newline="\n")

    print(f"{tables.event['event_name']} {roster.season} | FastF1 {fastf1.__version__}")
    for f in files:
        print(f"  {f.relative_path:<55} {f.rows:>7} rows  {f.sha256[:19]}...")
    print(f"\nquality: {report.worst}")
    for c in report.checks:
        print(f"  [{c.status}] {c.title}: {c.detail}")
    print(f"\nmanifest -> {MANIFEST.relative_to(ROOT)}\nreport   -> {REPORT.relative_to(ROOT)}")
    if report.worst == "FAIL":
        sys.exit(1)


if __name__ == "__main__":
    main()
