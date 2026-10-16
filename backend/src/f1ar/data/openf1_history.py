"""Tyres, pit stops, race control and weather for 2023+ from OpenF1 (H0, D-052).

Since 2026-10 F1's live-timing archive (FastF1's source) answers HTTP 401 without an F1 TV
sign-in, for every season (2024 included, which we had fetched before). OpenF1 (``openf1.org``)
serves the same timing data openly from 2023: per race session the **stints** (compound and tyre
age at the start), **pit** (lap, lane time), **race_control** (flags, SC/VSC, penalties),
**weather** (incl. track temperature, rainfall) and **laps**. Free plan: 3 requests/s and
30/minute; data CC BY-NC-SA 4.0, for education / non-commercial research
(docs/LEGAL_AND_LICENSES.md).
Every response is cached under ``data/raw/openf1/`` (gitignored); reruns fetch only what's missing.
"""

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from f1ar.data.jolpica_adapter import USER_AGENT

BASE = "https://api.openf1.org/v1"
TABLES = ("stints", "pit", "race_control", "weather", "laps")


@dataclass
class Throttle:
    """30 requests/minute: one per ``interval_s``."""

    interval_s: float = 2.2
    last: float = field(default=0.0)
    requests: int = 0

    def wait(self) -> None:
        delay = self.last + self.interval_s - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        self.last = time.monotonic()
        self.requests += 1


def get(url: str, throttle: Throttle, retries: int = 6) -> Any:
    backoff = 20.0
    for attempt in range(retries):
        throttle.wait()
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code == 404:  # OpenF1's answer to a query without rows
                return []
            if exc.code not in (429, 500, 502, 503, 504) or attempt == retries - 1:
                raise
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt == retries - 1:
                raise
        time.sleep(backoff)
        backoff = min(backoff * 2, 600.0)
    raise RuntimeError("unreachable")


def _cached(path: Path, url: str, throttle: Throttle) -> Any:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    data = get(url, throttle)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data), encoding="utf-8")
    tmp.replace(path)
    return data


def race_sessions(year: int, raw: Path, throttle: Throttle) -> list[dict[str, Any]]:
    """The year's race sessions (sprints excluded), in date order."""
    data = _cached(
        raw / str(year) / "sessions.json",
        f"{BASE}/sessions?year={year}&session_name=Race",
        throttle,
    )
    rows = [s for s in data if s.get("session_type") == "Race" and not s.get("is_cancelled")]
    return sorted(rows, key=lambda s: str(s["date_start"]))


def fetch_session(year: int, session_key: int, raw: Path, throttle: Throttle) -> dict[str, int]:
    out = {}
    for table in TABLES:
        data = _cached(
            raw / str(year) / str(session_key) / f"{table}.json",
            f"{BASE}/{table}?session_key={session_key}",
            throttle,
        )
        out[table] = len(data)
    return out


def session_tables(year: int, session_key: int, raw: Path) -> dict[str, pd.DataFrame]:
    folder = raw / str(year) / str(session_key)
    out = {}
    for table in TABLES:
        path = folder / f"{table}.json"
        data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
        df = pd.DataFrame(data)
        if table == "laps" and not df.empty:  # the per-segment arrays aren't needed
            df = df.drop(columns=[c for c in df.columns if c.startswith("segments_")])
        out[table] = df
    return out
