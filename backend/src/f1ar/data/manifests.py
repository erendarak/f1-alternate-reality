"""Dataset manifests (plan §6.2): what was fetched, from where, when, and the exact file hashes.

Data files under ``data/processed/`` are NOT committed (licensing); the small manifest YAML
under ``data/manifests/`` is, so anyone can verify they have byte-identical data.
"""

import hashlib
from pathlib import Path

import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict

from f1ar.domain.provenance import SourceType


class DatasetFile(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    relative_path: str
    sha256: str
    rows: int
    columns: list[str]
    source_type: SourceType
    description: str


class DatasetManifest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    dataset_id: str
    event: str
    source: str
    source_api_version: str
    retrieved_utc: str
    session_key: str
    license_review_status: str
    files: list[DatasetFile]
    unit_conventions: dict[str, str]
    missingness_notes: list[str]

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False, allow_unicode=True)


def file_sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def write_table(
    df: pd.DataFrame,
    path: Path,
    *,
    root: Path,
    source_type: SourceType,
    description: str,
) -> DatasetFile:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return DatasetFile(
        relative_path=path.relative_to(root).as_posix(),
        sha256=file_sha256(path),
        rows=len(df),
        columns=[str(c) for c in df.columns],
        source_type=source_type,
        description=description,
    )


def verify_files(manifest: DatasetManifest, root: Path) -> dict[str, bool]:
    """``relative_path -> matches`` for every file in the manifest (False if missing)."""
    out = {}
    for f in manifest.files:
        path = root / f.relative_path
        out[f.relative_path] = path.exists() and file_sha256(path) == f.sha256
    return out
