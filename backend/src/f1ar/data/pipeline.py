"""Fetch + validate any race into the project's data layout (plan §6.4; generalises step 7).

For ``(season, round)``:

1. Jolpica-F1 results: the event identity, a race-specific roster (written once to
   ``configs/rosters/<event_id>.json``), and the independent cross-check reference.
2. FastF1 race session: normalised tables written to ``data/processed/<event_id>/``.
3. Track outline from the winner's most *typical* clean green lap (median representative lap).
4. Manifest ``data/manifests/<event_id>_fastf1.yaml``, report ``docs/data/<event_id>_quality.md``.

The only network access is FastF1 (with its cache) and one Jolpica request.
"""

import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import fastf1
import pandas as pd
import yaml

from f1ar.data import jolpica_adapter
from f1ar.data.fastf1_adapter import load_race_session, tables_from_session, track_outline
from f1ar.data.manifests import DatasetFile, DatasetManifest, verify_files, write_table
from f1ar.data.quality import (
    QualityReport,
    build_quality_report,
    render_markdown,
    representative_laps,
)
from f1ar.domain.identities import RaceRoster, load_roster
from f1ar.domain.provenance import SourceType


def event_id_for(season: int, race_name: str) -> str:
    """``(2024, "Italian Grand Prix") -> "2024_italian_gp"``."""
    name = jolpica_adapter.slug(race_name)
    return f"{season}_{re.sub(r'grand_prix$', 'gp', name)}"


@dataclass(frozen=True)
class EventPaths:
    root: Path
    event_id: str

    @property
    def processed(self) -> Path:
        return self.root / "data" / "processed" / self.event_id

    @property
    def manifest(self) -> Path:
        return self.root / "data" / "manifests" / f"{self.event_id}_fastf1.yaml"

    @property
    def report(self) -> Path:
        return self.root / "docs" / "data" / f"{self.event_id}_quality.md"

    @property
    def roster(self) -> Path:
        return self.root / "configs" / "rosters" / f"{self.event_id}.json"


@dataclass(frozen=True)
class FetchResult:
    paths: EventPaths
    event_name: str
    files: list[DatasetFile]
    report: QualityReport
    outline_lap: int | None


def typical_lap(laps: pd.DataFrame, driver_number: int) -> int:
    """The representative lap of a car whose time is closest to that car's median."""
    reps = representative_laps(laps[laps["driver_number"] == driver_number])
    if reps.empty:
        raise ValueError(f"car {driver_number} has no representative laps")
    median = reps["lap_time_s"].median()
    ranked = reps.assign(deviation=(reps["lap_time_s"] - median).abs()).sort_values(
        ["deviation", "lap_number"], kind="stable"
    )  # ties -> earliest lap, so the choice is deterministic
    return int(ranked["lap_number"].iloc[0])


def fetch_event(
    season: int,
    round_: int,
    root: Path,
    *,
    telemetry: bool = True,
    refresh_roster: bool = False,
) -> FetchResult:
    url = jolpica_adapter.results_url(season, round_)
    payload = jolpica_adapter.fetch_json(url)
    race_name = payload["MRData"]["RaceTable"]["Races"][0]["raceName"]
    paths = EventPaths(root, event_id_for(season, race_name))
    retrieved = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    roster: RaceRoster
    if paths.roster.exists() and not refresh_roster:
        roster = load_roster(paths.roster, expected_event_id=paths.event_id)
    else:
        roster = jolpica_adapter.roster_from_results(
            payload, event_id=paths.event_id, source_url=url, retrieved_utc=retrieved
        )
        paths.roster.parent.mkdir(parents=True, exist_ok=True)
        paths.roster.write_text(
            json.dumps(roster.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    reference = jolpica_adapter.result_records(payload)

    session = load_race_session(season, round_, root / ".cache" / "fastf1", telemetry=telemetry)
    tables = tables_from_session(session)

    hist, out = SourceType.HISTORICAL_VERIFIED, paths.processed
    files = [
        write_table(
            tables.laps,
            out / "laps.parquet",
            root=root,
            source_type=hist,
            description="one row per car per lap: times, sectors, tyres, stint, position",
        ),
        write_table(
            tables.results,
            out / "results.parquet",
            root=root,
            source_type=hist,
            description="official classification incl. grid, status, laps, team colour",
        ),
        write_table(
            tables.weather,
            out / "weather.parquet",
            root=root,
            source_type=hist,
            description="trackside weather samples (~1/min)",
        ),
        write_table(
            tables.track_status,
            out / "track_status.parquet",
            root=root,
            source_type=hist,
            description="track status changes (green/SC/VSC/red)",
        ),
        write_table(
            tables.race_control,
            out / "race_control.parquet",
            root=root,
            source_type=hist,
            description="race control messages",
        ),
        write_table(
            tables.corners,
            out / "corners.parquet",
            root=root,
            source_type=hist,
            description="corner numbers/positions (metres) + suggested display rotation",
        ),
    ]
    outline_lap = None
    if not tables.positions.empty:
        winner = int(tables.results.iloc[0]["driver_number"])
        outline_lap = typical_lap(tables.laps, winner)
        outline = track_outline(
            tables.positions, tables.laps, driver_number=winner, lap_number=outline_lap
        )
        files += [
            write_table(
                tables.positions,
                out / "positions.parquet",
                root=root,
                source_type=hist,
                description="car X/Y/Z location samples in metres (~4 Hz)",
            ),
            write_table(
                outline,
                out / "track_outline.parquet",
                root=root,
                source_type=SourceType.ESTIMATED_FROM_DATA,
                description=f"derived: car {winner} lap {outline_lap} path as outline",
            ),
        ]

    report = build_quality_report(
        tables,
        roster,
        source={
            "source": "FastF1",
            "source_version": fastf1.__version__,
            "retrieved_utc": retrieved,
            "reference": "Jolpica-F1 results",
        },
        expected_total_laps=max(r["laps"] for r in reference),
        reference_results=reference,
    )
    manifest = DatasetManifest(
        dataset_id=f"{paths.event_id}_fastf1_v1",
        event=paths.event_id,
        source="fastf1",
        source_api_version=fastf1.__version__,
        retrieved_utc=retrieved,
        session_key=f"{season}/round {round_}/R",
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
    paths.manifest.parent.mkdir(parents=True, exist_ok=True)
    paths.manifest.write_text(manifest.to_yaml(), encoding="utf-8", newline="\n")
    paths.report.parent.mkdir(parents=True, exist_ok=True)
    paths.report.write_text(render_markdown(report), encoding="utf-8", newline="\n")
    return FetchResult(paths, str(tables.event["event_name"]), files, report, outline_lap)


def verify_event(root: Path, event_id: str) -> dict[str, bool]:
    paths = EventPaths(root, event_id)
    manifest = DatasetManifest.model_validate(
        yaml.safe_load(paths.manifest.read_text(encoding="utf-8"))
    )
    return verify_files(manifest, root)


def format_result(result: FetchResult, root: Path) -> list[str]:
    """Human-readable summary lines for CLIs."""
    lines = [f"{result.event_name} -> {result.paths.event_id} | FastF1 {fastf1.__version__}"]
    lines += [f"  {f.relative_path:<58} {f.rows:>7} rows  {f.sha256[:19]}..." for f in result.files]
    if result.outline_lap is not None:
        lines.append(f"  track outline from the winner's typical lap {result.outline_lap}")
    lines += ["", f"quality: {result.report.worst}"]
    lines += [f"  [{c.status}] {c.title}: {c.detail}" for c in result.report.checks]
    lines += ["", f"roster   -> {result.paths.roster.relative_to(root)}"]
    lines.append(f"manifest -> {result.paths.manifest.relative_to(root)}")
    lines.append(f"report   -> {result.paths.report.relative_to(root)}")
    return lines
