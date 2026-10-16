"""Historical races 2000-2025 from Jolpica-F1 (H0, D-052): raw cache and tidy tables.

Jolpica (Ergast successor) has results since 1950, **lap times since 1996** and **pit stops
since 2011**; a page holds at most 100 rows. Its unauthenticated limits are 4 requests/s burst
and 500/hour (docs/LEGAL_AND_LICENSES.md), so fetching is throttled (``Throttle``) and every page
is cached under ``data/raw/jolpica/<season>/<round>/`` (gitignored): a rerun only fetches what is
missing. Data licence CC BY-NC-SA 4.0 (non-commercial); raw and processed tables stay local.

Tables per race (``race_tables``): ``laps`` (driver, lap, position, lap time), ``results`` (grid,
finish, status, points, team), ``pitstops`` (2011+: lap, stop number, duration), ``qualifying``.
"""

import functools
import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from f1ar.data.jolpica_adapter import BASE_URL, USER_AGENT

PAGE = 100
PITSTOPS_FROM = 2011
LAPS_FROM = 1996


@dataclass
class Throttle:
    """At most one request per ``interval_s`` (500/hour = 7.2 s; default leaves headroom)."""

    interval_s: float = 9.0
    last: float = field(default=0.0)
    requests: int = 0

    def wait(self) -> None:
        delay = self.last + self.interval_s - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        self.last = time.monotonic()
        self.requests += 1


def get_json(url: str, throttle: Throttle, retries: int = 6) -> Any:
    """GET with the project's User-Agent; backs off on 429 / 5xx / network errors."""
    backoff = 30.0
    for attempt in range(retries):
        throttle.wait()
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code not in (429, 500, 502, 503, 504) or attempt == retries - 1:
                raise
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt == retries - 1:
                raise
        time.sleep(backoff)
        backoff = min(backoff * 2, 900.0)
    raise RuntimeError("unreachable")


def cached(path: Path, fetch: Callable[[], Any]) -> Any:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    data = fetch()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data), encoding="utf-8")
    tmp.replace(path)
    return data


def season_schedule(season: int, raw: Path, throttle: Throttle) -> list[dict[str, Any]]:
    data = cached(
        raw / str(season) / "schedule.json",
        lambda: get_json(f"{BASE_URL}/{season}.json?limit={PAGE}", throttle),
    )
    races: list[dict[str, Any]] = data["MRData"]["RaceTable"]["Races"]
    return races


def _pages(
    season: int, round_: int, kind: str, raw: Path, throttle: Throttle
) -> list[dict[str, Any]]:
    """Every page of ``/<season>/<round>/<kind>.json`` (cached per page)."""
    folder = raw / str(season) / f"{round_:02d}"
    pages = []
    offset = 0
    while True:
        url = f"{BASE_URL}/{season}/{round_}/{kind}.json?limit={PAGE}&offset={offset}"
        data = cached(
            folder / f"{kind}_{offset:05d}.json", functools.partial(get_json, url, throttle)
        )
        pages.append(data)
        total = int(data["MRData"]["total"])
        offset += PAGE
        if offset >= total:
            return pages


def fetch_race(season: int, round_: int, raw: Path, throttle: Throttle) -> dict[str, int]:
    """Cache every table of one race; returns the row totals per table."""
    out = {}
    kinds = ["results", "qualifying"]
    if season >= LAPS_FROM:
        kinds.append("laps")
    if season >= PITSTOPS_FROM:
        kinds.append("pitstops")
    for kind in kinds:
        pages = _pages(season, round_, kind, raw, throttle)
        out[kind] = int(pages[0]["MRData"]["total"])
    return out


def race_done(season: int, round_: int, raw: Path) -> bool:
    """Whether every table of the race is cached (all pages)."""
    folder = raw / str(season) / f"{round_:02d}"
    kinds = ["results", "qualifying"] + (["laps"] if season >= LAPS_FROM else [])
    kinds += ["pitstops"] if season >= PITSTOPS_FROM else []
    for kind in kinds:
        first = folder / f"{kind}_00000.json"
        if not first.exists():
            return False
        total = int(json.loads(first.read_text(encoding="utf-8"))["MRData"]["total"])
        if any(
            not (folder / f"{kind}_{o:05d}.json").exists() for o in range(0, max(total, 1), PAGE)
        ):
            return False
    return True


# --- tidy tables (pure: from the cached pages) -------------------------------------------------


def _races(folder: Path, kind: str) -> list[dict[str, Any]]:
    out = []
    for p in sorted(folder.glob(f"{kind}_*.json")):
        out += json.loads(p.read_text(encoding="utf-8"))["MRData"]["RaceTable"]["Races"]
    return out


def _seconds(text: str | None) -> float | None:
    if not text:
        return None
    parts = text.split(":")
    try:
        return float(sum(float(x) * 60**i for i, x in enumerate(reversed(parts))))
    except ValueError:
        return None


def race_tables(season: int, round_: int, raw: Path) -> dict[str, pd.DataFrame]:
    folder = raw / str(season) / f"{round_:02d}"
    laps = [
        {
            "lap": int(lap["number"]),
            "driver": t["driverId"],
            "position": int(t["position"]),
            "lap_time_s": _seconds(t.get("time")),
        }
        for race in _races(folder, "laps")
        for lap in race.get("Laps", [])
        for t in lap["Timings"]
    ]
    results = [
        {
            "driver": r["Driver"]["driverId"],
            "code": r["Driver"].get("code"),
            "number": r.get("number"),
            "team": r["Constructor"]["constructorId"],
            "grid": int(r["grid"]),
            "position": int(r["position"]),
            "classified": r.get("positionText", "").isdigit(),
            "laps": int(r["laps"]),
            "status": r["status"],
            "points": float(r["points"]),
            "time_s": (
                _seconds(r["Time"]["time"]) if "Time" in r and ":" in r["Time"]["time"] else None
            ),
        }
        for race in _races(folder, "results")
        for r in race.get("Results", [])
    ]
    pitstops = [
        {
            "driver": s["driverId"],
            "lap": int(s["lap"]),
            "stop": int(s["stop"]),
            "duration_s": _seconds(s.get("duration")),
        }
        for race in _races(folder, "pitstops")
        for s in race.get("PitStops", [])
    ]
    qualifying = [
        {
            "driver": q["Driver"]["driverId"],
            "position": int(q["position"]),
            "q1_s": _seconds(q.get("Q1")),
            "q2_s": _seconds(q.get("Q2")),
            "q3_s": _seconds(q.get("Q3")),
        }
        for race in _races(folder, "qualifying")
        for q in race.get("QualifyingResults", [])
    ]
    return {
        "laps": pd.DataFrame(laps, columns=["lap", "driver", "position", "lap_time_s"]),
        "results": pd.DataFrame(results),
        "pitstops": pd.DataFrame(pitstops, columns=["driver", "lap", "stop", "duration_s"]),
        "qualifying": pd.DataFrame(qualifying),
    }
