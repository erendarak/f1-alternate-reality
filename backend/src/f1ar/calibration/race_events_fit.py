"""Race events from the 2024 season: pit crews and reliability per team, neutralisations
without a retirement, red flags, time penalties (W4, engine v0.11.0, D-042).

Measured on the fetched races (``data/processed/<event>``):

- **Pit crews**: pit-lane time of every green-flag stop (pit exit - pit entry) relative to its
  race's median: per team the median (the crew's offset, shrunk toward 0 with
  ``CREW_PRIOR_STOPS``) and the share of stops 3+ s slow (shrunk toward the field's); the field's
  slow share and the size of the slow part (mean excess over 3 s) give the slow-stop tail.
- **Reliability**: retirements (and non-starts) per team per start, shrunk toward the field's
  rate with ``RELIABILITY_PRIOR`` starts. The data has no cause, so all-cause retirements stand
  in for mechanical failures (labelled; crashes stay on the drivers' side).
- **Neutralisations without a retirement** (debris, a stopped car that rejoined): SC/VSC
  deployments with no retirement within a lap, per race lap, on the dry races; their SC share.
- **Red flags** (track status 5, end-of-session messages excluded) and **time penalties** by
  type from the race-control messages: collisions (per pass, ``reference_metrics`` passes),
  false starts (per starter), pit-lane speeding (per stop), everything else (per car and lap).

``risk_scale`` and ``red_flag_per_crash`` need the simulated field (crash counts depend on the
engine) and are fitted by ``scripts/fit_race_events.py``.
"""

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from f1ar.calibration.season_files import REFERENCE_SEASON, own_season
from f1ar.data.status import is_finisher
from f1ar.domain.identities import load_roster

CREW_PRIOR_STOPS = 20.0
SLOW_PRIOR_STOPS = 30.0
RELIABILITY_PRIOR = 24.0
"""Starts of shrinkage toward the field: all-cause retirements are a weak proxy for failures."""
SLOW_THRESHOLD_S = 3.0
EVENTS_FILE = Path("calibration") / "race_events_2024_v1.json"


def _team_of(configs: Path, event_id: str) -> dict[int, str]:
    roster = load_roster(configs / "rosters" / f"{event_id}.json", expected_event_id=event_id)
    return {e.car_number: e.team_id for e in roster.entries}


def pit_lane_times(data_dir: Path, team_of: dict[int, str]) -> pd.DataFrame:
    laps = pd.read_parquet(data_dir / "laps.parquet").sort_values(["driver_number", "lap_number"])
    rows = []
    for number, g in laps.groupby("driver_number"):
        recs = g.to_dict("records")
        for a, b in zip(recs, recs[1:], strict=False):
            if pd.isna(a["pit_in_time_s"]) or pd.isna(b["pit_out_time_s"]):
                continue
            if str(a.get("track_status", "1")) != "1":
                continue
            lane = float(b["pit_out_time_s"]) - float(a["pit_in_time_s"])
            if 10.0 < lane < 120.0:
                rows.append({"team": team_of.get(int(str(number)), "?"), "lane": lane})
    out = pd.DataFrame(rows)
    if len(out):
        out["rel"] = out["lane"] - out["lane"].median()
    return out


@dataclass
class EventCounts:
    event_id: str
    laps: int
    starters: int
    retirements_by_team: dict[str, int]
    starts_by_team: dict[str, int]
    neutralisations: list[tuple[int, str, bool]] = field(default_factory=list)
    """(lap, SC/VSC, a retirement within a lap)."""
    red_flags: int = 0
    penalties: dict[str, int] = field(default_factory=dict)
    stops: int = 0


PENALTY_KINDS = (
    ("collision", r"CAUSING A COLLISION|FORCING ANOTHER (DRIVER|CAR) OFF"),
    ("false_start", r"FALSE START"),
    ("pit_speeding", r"SPEEDING IN THE PIT"),
)


def penalty_kind(message: str) -> str | None:
    """A time / drive-through / stop-go penalty's kind. H2a (D-054): before 2022 the messages
    have no "FIA STEWARDS:" prefix, and stop-go penalties were missed (3 of 50 in 2024)."""
    if "PENALTY FOR" not in message:
        return None
    if not any(k in message for k in ("DRIVE THROUGH", "SECOND TIME PENALTY", "STOP/GO")):
        return None
    for kind, pattern in PENALTY_KINDS:
        if re.search(pattern, message):
            return kind
    return "other"


def event_counts(data_dir: Path, configs: Path, event_id: str) -> EventCounts:
    team_of = _team_of(configs, event_id)
    res = pd.read_parquet(data_dir / "results.parquet")
    rc = pd.read_parquet(data_dir / "race_control.parquet")
    ts = pd.read_parquet(data_dir / "track_status.parquet")
    laps = pd.read_parquet(data_dir / "laps.parquet")
    total = int(res["laps"].max())
    # H2a (D-054): 2024 says "Retired"; earlier seasons give the cause ("Engine", ...)
    ret_laps = [
        int(n) + 1 for n, s in zip(res["laps"], res["status"], strict=True) if not is_finisher(s)
    ]
    by_team_ret: dict[str, int] = {}
    by_team_start: dict[str, int] = {}
    for number, status in zip(res["driver_number"], res["status"], strict=True):
        team = team_of.get(int(str(number)), "?")
        by_team_start[team] = by_team_start.get(team, 0) + 1
        if not is_finisher(status):
            by_team_ret[team] = by_team_ret.get(team, 0) + 1
    neutral = []
    for _, m in rc[rc["message"].astype(str).str.contains("SAFETY CAR DEPLOYED")].iterrows():
        lap = int(m["lap"]) if pd.notna(m["lap"]) else 0
        kind = "VSC" if "VIRTUAL" in str(m["message"]) else "SC"
        neutral.append((lap, kind, any(abs(r - lap) <= 1 for r in ret_laps)))
    penalties: dict[str, int] = {}
    for msg in rc["message"].astype(str):
        k = penalty_kind(msg)
        if k is not None:
            penalties[k] = penalties.get(k, 0) + 1
    stops = int(laps["pit_in_time_s"].notna().sum())
    return EventCounts(
        event_id=event_id,
        laps=total,
        starters=int((res["status"] != "Did not start").sum()),
        retirements_by_team=by_team_ret,
        starts_by_team=by_team_start,
        neutralisations=neutral,
        red_flags=int((ts["status"].astype(str) == "5").sum()),
        penalties=penalties,
        stops=stops,
    )


def fit(
    root: Path,
    events: Sequence[str],
    dry: Sequence[str],
    passes_by_event: dict[str, int],
    season: int = REFERENCE_SEASON,
) -> dict[str, Any]:
    """Everything measurable from data; ``risk_scale`` / ``red_flag_per_crash`` start at 1 / 0."""
    configs = root / "configs"
    processed = root / "data" / "processed"
    lanes = pd.concat(
        [pit_lane_times(processed / e, _team_of(configs, e)) for e in events], ignore_index=True
    )
    slow = lanes["rel"] > SLOW_THRESHOLD_S
    field_slow = float(slow.mean())
    slow_excess = float((lanes.loc[slow, "rel"] - SLOW_THRESHOLD_S).mean())
    crews = {}
    for team, g in lanes.groupby("team"):
        n = len(g)
        offset = float(g["rel"].median()) * n / (n + CREW_PRIOR_STOPS)
        share = float((g["rel"] > SLOW_THRESHOLD_S).mean())
        mult = (share * n + field_slow * SLOW_PRIOR_STOPS) / (n + SLOW_PRIOR_STOPS) / field_slow
        crews[str(team)] = {
            "pit_crew_offset_s": round(offset, 3),
            "pit_slow_multiplier": round(mult, 3),
            "stops": n,
            "raw_median_rel_s": round(float(g["rel"].median()), 3),
            "raw_slow_share": round(share, 3),
        }
    counts = [event_counts(processed / e, configs, e) for e in events]
    ret: dict[str, int] = {}
    starts: dict[str, int] = {}
    for c in counts:
        for t, n in c.starts_by_team.items():
            starts[t] = starts.get(t, 0) + n
        for t, n in c.retirements_by_team.items():
            ret[t] = ret.get(t, 0) + n
    q = sum(ret.values()) / sum(starts.values())
    reliability = {
        t: {
            "failure_multiplier": round(
                (ret.get(t, 0) + RELIABILITY_PRIOR * q) / (n * q + RELIABILITY_PRIOR * q), 3
            ),
            "retirements": ret.get(t, 0),
            "starts": n,
        }
        for t, n in sorted(starts.items())
    }
    dry_counts = [c for c in counts if c.event_id in dry]
    race_laps = sum(c.laps for c in dry_counts)
    free = [(lap, k) for c in dry_counts for lap, k, with_ret in c.neutralisations if not with_ret]
    pens: dict[str, int] = {}
    for c in counts:
        for k, n in c.penalties.items():
            pens[k] = pens.get(k, 0) + n
    car_laps = sum(c.laps * c.starters for c in counts)
    passes = sum(passes_by_event.get(c.event_id, 0) for c in counts)
    return {
        "label": f"{season} race events (W4, D-042)",
        "events": list(events),
        "dry_events": list(dry),
        "pit": {
            "slow_threshold_s": SLOW_THRESHOLD_S,
            "slow_stop_probability": round(field_slow, 4),
            "slow_stop_scale_s": round(slow_excess, 2),
            "stops": len(lanes),
            "note": "pit-lane time vs the race median, green-flag stops",
        },
        "crews": crews,
        "reliability": {
            "field_retirements_per_start": round(q, 4),
            "teams": reliability,
            "note": "all-cause retirements and non-starts as a proxy (no cause in the data)",
        },
        "debris": {
            "per_lap": round(len(free) / race_laps, 5),
            "sc_share": round(sum(1 for _, k in free if k == "SC") / len(free), 3) if free else 0.4,
            "count": len(free),
            "race_laps": race_laps,
            "note": "SC/VSC deployments without a retirement within a lap, dry races",
        },
        "red_flags": {
            "count": sum(c.red_flags for c in counts),
            "races": len(counts),
            "per_race": round(sum(c.red_flags for c in counts) / len(counts), 4),
        },
        "penalties": {
            "counts": pens,
            "collision_per_pass": round(pens.get("collision", 0) / max(1, passes), 5),
            "false_start_per_starter": round(
                pens.get("false_start", 0) / sum(c.starters for c in counts), 5
            ),
            "pit_speeding_per_stop": round(
                pens.get("pit_speeding", 0) / max(1, sum(c.stops for c in counts)), 5
            ),
            "other_per_car_lap": round(pens.get("other", 0) / car_laps, 6),
            "passes": passes,
        },
        "fitted": {"risk_scale": 1.0, "red_flag_per_crash": 0.0},
    }


def load_events(configs: Path, season: int = REFERENCE_SEASON) -> dict[str, Any] | None:
    """This season's fit, or None: keyed by driver/team, never borrowed (H2a, D-054)."""
    path = own_season(configs, EVENTS_FILE, season)
    if path is None:
        return None
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data
