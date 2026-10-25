"""Race-day tyre priors from history, and how far the truth lands from them (H3, D-058).

Every data-backed race has fitted tyre values (W1, D-041; per season since H2a/H2b/H1b): the
life of each dry compound in laps of the real stints (``life_raw``) and how fast the tyres wear
(``wear_multiplier``). In the world these become the race's **hidden truth**. A team only has
what history said *before* that race: the **prior**.

**Prior** (one quantity at a time: ``log life_raw`` per compound, ``log wear_multiplier``), fitted
for each target race on the races strictly before it (earlier seasons, and earlier rounds of
the target's season; never the race itself or anything later):

    y = season level + track effect + noise

- the season level follows a random walk: each season is shrunk toward the previous one with
  weight ``season_shrink`` (races' worth); a season with no race yet takes the last level;
- the track effect is the recency-weighted mean of the track's residuals (weight
  ``recency ** years_ago``), shrunk toward 0 with weight ``track_shrink``; a new track gets 0.

The two are fitted by backfitting. The three hyper-parameters are chosen on the tuning seasons
(``TUNING_SEASONS``, the older half) and reported on all.

**Surprise size** (``estimated_from_data``): the residual truth - prior over every race with
history, per era; split into a part shared by the compounds (the race-mean residual), a
per-compound part (2018+ only: before 2018 the compounds are one measured stint length times
2018's ratios, H2b) and the wear residual with its correlation to the shared part. Only
compounds measured in the race count (``life_source`` "q75 of ..."); filled ones are the
season's ratios, not a measurement.

Caveats (labelled in the output): the season-level calibration (``kappa``, the wear curve's
shape, the compound ratios) is fitted on the whole season and treated as known calibration, not
race-day information; ``life_raw`` is a stint-length measurement, so before 2010 it also carries
the fuel strategy (refuelling era).
"""

import math
import statistics
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from f1ar.calibration.lap_model_fit import LIFE_FILE
from f1ar.calibration.season_files import season_path

PRIOR_FILE = Path("calibration") / "race_day_prior_v1.json"
COMPOUNDS = ("SOFT", "MEDIUM", "HARD")
QUANTITIES = (*COMPOUNDS, "WEAR")
TUNING_SEASONS = range(2000, 2013)
"""Hyper-parameters are chosen on targets in these seasons (rule fixed before fitting)."""
ERAS = {
    "2000-2009": range(2000, 2010),
    "2010-2017": range(2010, 2018),
    "2018-2025": range(2018, 2026),
}
FIRST_COMPOUND_SEASON = 2018
BACKFIT_ITERATIONS = 8


@dataclass(frozen=True)
class RaceRecord:
    """One race's fitted tyre values (the truth) and where it sits in history."""

    event_id: str
    season: int
    round: int
    circuit_id: str
    life_raw: Mapping[str, float]
    measured: Mapping[str, bool]
    wear: float
    split: str
    """``dry_train`` (in the season's tyre fit) or ``dry_test`` (not used by the fitted globals)."""

    @property
    def key(self) -> tuple[int, int]:
        return (self.season, self.round)

    def value(self, quantity: str) -> float:
        return math.log(self.wear if quantity == "WEAR" else self.life_raw[quantity])


@dataclass(frozen=True)
class PriorParams:
    season_shrink: float = 3.0
    track_shrink: float = 1.0
    recency: float = 0.8


@dataclass(frozen=True)
class Prior:
    values: Mapping[str, float]
    """Predicted ``log`` value per quantity."""
    track_years: int
    """Earlier seasons with a race at this circuit."""
    season_races: int
    """Earlier rounds of this season in the data."""


def load_records(configs: Path, seasons: Iterable[int]) -> list[RaceRecord]:
    """Every race with a tyre-life fit, in time order."""
    import json

    out = []
    for season in seasons:
        path = configs / season_path(LIFE_FILE, season)
        if not path.exists():
            continue
        cfg = json.loads(path.read_text(encoding="utf-8"))
        fitted = set(cfg.get("fitted_on", []))
        for event_id, ev in cfg["events"].items():
            roster = json.loads(
                (configs / "rosters" / f"{event_id}.json").read_text(encoding="utf-8")
            )
            out.append(
                RaceRecord(
                    event_id=event_id,
                    season=season,
                    round=int(roster["round"]),
                    circuit_id=str(roster["circuit_id"]),
                    life_raw={c: float(ev["life_raw"][c]) for c in COMPOUNDS},
                    measured={
                        c: str(ev["life_source"].get(c, "")).startswith("q75") for c in COMPOUNDS
                    },
                    wear=float(ev["wear_multiplier"]),
                    split="dry_train" if event_id in fitted else "dry_test",
                )
            )
    return sorted(out, key=lambda r: r.key)


def _fit_one(
    rows: Sequence[tuple[int, str, float]], target_season: int, params: PriorParams
) -> tuple[dict[int, float], dict[str, float]]:
    """Backfit season levels and track effects on ``(season, circuit, y)`` rows."""
    seasons = sorted({s for s, _, _ in rows})
    grand = statistics.fmean(y for _, _, y in rows)
    level = {s: statistics.fmean(y for s2, _, y in rows if s2 == s) for s in seasons}
    track: dict[str, float] = defaultdict(float)
    by_season: dict[int, list[tuple[str, float]]] = defaultdict(list)
    for s, t, y in rows:
        by_season[s].append((t, y))
    for _ in range(BACKFIT_ITERATIONS):
        num: dict[str, float] = defaultdict(float)
        den: dict[str, float] = defaultdict(float)
        for s, t, y in rows:
            w = params.recency ** (target_season - s)
            num[t] += w * (y - level[s])
            den[t] += w
        track = defaultdict(float, {t: num[t] / (den[t] + params.track_shrink) for t in num})
        previous = grand
        for s in seasons:
            resid = [y - track[t] for t, y in by_season[s]]
            level[s] = (sum(resid) + params.season_shrink * previous) / (
                len(resid) + params.season_shrink
            )
            previous = level[s]
    return level, dict(track)


def prior_for(
    target: RaceRecord, history: Sequence[RaceRecord], params: PriorParams
) -> Prior | None:
    """The prior for ``target`` from ``history`` races strictly before it (None: no history)."""
    past = [r for r in history if r.key < target.key]
    if not past:
        return None
    values = {}
    for q in QUANTITIES:
        rows = [(r.season, r.circuit_id, r.value(q)) for r in past]
        level, track = _fit_one(rows, target.season, params)
        last = max(level)  # the target's season if it has earlier rounds, else the latest
        values[q] = level[last] + track.get(target.circuit_id, 0.0)
    return Prior(
        values=values,
        track_years=len({r.season for r in past if r.circuit_id == target.circuit_id}),
        season_races=sum(r.season == target.season for r in past),
    )


@dataclass(frozen=True)
class Residual:
    record: RaceRecord
    prior: Prior

    def life(self, compound: str) -> float:
        return self.record.value(compound) - self.prior.values[compound]

    @property
    def measured(self) -> list[str]:
        return [c for c in COMPOUNDS if self.record.measured[c]]

    @property
    def shared(self) -> float:
        """Race-mean life residual over its measured compounds (all compounds if none)."""
        cs = self.measured or list(COMPOUNDS)
        return statistics.fmean(self.life(c) for c in cs)

    @property
    def wear(self) -> float:
        return self.record.value("WEAR") - self.prior.values["WEAR"]


def residuals(records: Sequence[RaceRecord], params: PriorParams) -> list[Residual]:
    out = []
    for r in records:
        p = prior_for(r, records, params)
        if p is not None:
            out.append(Residual(r, p))
    return out


def rmse(values: Sequence[float]) -> float:
    return math.sqrt(statistics.fmean(v * v for v in values)) if values else float("nan")


def score(res: Sequence[Residual]) -> float:
    """Tuning loss: RMS of the shared life residual plus the wear residual."""
    return rmse([x.shared for x in res]) + rmse([x.wear for x in res])


def surprise_size(res: Sequence[Residual]) -> dict[str, float | int]:
    """Spread of truth - prior: shared life part, per-compound part (2018+), wear, correlation."""
    shared = [x.shared for x in res]
    wear = [x.wear for x in res]
    idio: list[float] = []
    for x in res:
        cs = x.measured
        if x.record.season >= FIRST_COMPOUND_SEASON and len(cs) >= 2:
            m = statistics.fmean(x.life(c) for c in cs)
            k = len(cs)
            idio.extend((x.life(c) - m) * math.sqrt(k / (k - 1)) for c in cs)
    varies = len(set(shared)) > 1 and len(set(wear)) > 1
    rho = statistics.correlation(shared, wear) if len(shared) > 2 and varies else 0.0
    return {
        "races": len(res),
        "life_shared_sd": round(rmse(shared), 4),
        "life_shared_bias": round(statistics.fmean(shared), 4) if shared else 0.0,
        "life_compound_sd": round(rmse(idio), 4) if idio else 0.0,
        "life_compound_n": len(idio),
        "wear_sd": round(rmse(wear), 4),
        "wear_bias": round(statistics.fmean(wear), 4) if wear else 0.0,
        "life_wear_correlation": round(rho, 3),
    }


def era_of(season: int) -> str:
    for name, years in ERAS.items():
        if season in years:
            return name
    raise ValueError(f"no era for {season}")
