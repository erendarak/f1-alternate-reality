"""What old races lack, inferred from lap times (H0, D-052), validated where the truth is known.

Jolpica has lap times since 1996 but pit stops only since 2011, and no track status before
FastF1 (2018). From the lap times of the whole field:

- **Neutralised laps** (SC / VSC / red flag): laps whose field median is more than
  ``NEUTRAL_RATIO`` times the race's typical green-lap median (the whole field slows down).
- **Pit stops:** a car's lap more than ``PIT_EXCESS_S`` slower than the field median of that lap
  (the pit-lane loss lands in the lap that contains the stop); under a neutralisation the excess
  is measured against the neutralised field and ``PIT_EXCESS_NEUTRAL_S`` applies. The first lap
  (standing start) and the car's last lap (retirement) are skipped.

Both are ``estimated_from_data``. Thresholds tuned on the 2018-2021 FastF1 races (real pit laps and
track status) and checked on 2022-2025 (EXP-021): pit stops precision 0.88, recall 0.73 (misses
are mostly stops under a safety car); neutralised laps 83 % found, ~1.4 false laps per race.
"""

import pandas as pd

NEUTRAL_RATIO = 1.08
PIT_EXCESS_S = 10.0
PIT_EXCESS_NEUTRAL_S = 8.0


def neutralised_laps(laps: pd.DataFrame) -> set[int]:
    """Laps on which the whole field ran slow (likely SC / VSC / red flag)."""
    med = _lap_medians(laps)
    racing = [m for lap, m in med.items() if lap > 1]
    if not racing:
        return set()
    typical = float(pd.Series(racing).quantile(0.3))
    return {lap for lap, m in med.items() if lap > 1 and m > NEUTRAL_RATIO * typical}


def _lap_medians(laps: pd.DataFrame) -> dict[int, float]:
    valid = laps.dropna(subset=["lap_time_s"])
    rows = valid.groupby("lap")["lap_time_s"].median().reset_index().to_dict("records")
    return {int(r["lap"]): float(r["lap_time_s"]) for r in rows}


def inferred_pit_laps(laps: pd.DataFrame) -> pd.DataFrame:
    """One row per inferred stop: driver, lap (the lap whose time contains the stop)."""
    valid = laps.dropna(subset=["lap_time_s"])
    med = _lap_medians(laps)
    neutral = neutralised_laps(laps)
    last_rows = valid.groupby("driver")["lap"].max().reset_index().to_dict("records")
    last = {str(r["driver"]): int(r["lap"]) for r in last_rows}
    found: list[tuple[str, int, float]] = []
    for r in sorted(valid.to_dict("records"), key=lambda x: (str(x["driver"]), int(x["lap"]))):
        driver, lap = str(r["driver"]), int(r["lap"])
        if lap <= 1 or lap == last[driver]:
            continue
        excess = float(r["lap_time_s"]) - med[lap]
        if excess <= (PIT_EXCESS_NEUTRAL_S if lap in neutral else PIT_EXCESS_S):
            continue
        # one stop can slow two consecutive laps (in-lap and out-lap): keep the slower one
        if found and found[-1][0] == driver and found[-1][1] == lap - 1:
            if excess > found[-1][2]:
                found[-1] = (driver, lap, excess)
            continue
        found.append((driver, lap, excess))
    return pd.DataFrame(
        [(d, lap, round(e, 2)) for d, lap, e in found], columns=["driver", "lap", "excess_s"]
    )


def match(real: pd.DataFrame, inferred: pd.DataFrame, tolerance: int = 1) -> dict[str, float]:
    """Precision / recall of inferred stops against real ones (same driver, lap within
    ``tolerance``; Jolpica's stop lap is the in-lap)."""
    candidates = [(str(x["driver"]), int(x["lap"])) for x in inferred.to_dict("records")]
    used: set[int] = set()
    hits = 0
    for r in real.to_dict("records"):
        for i, (driver, lap) in enumerate(candidates):
            if (
                i not in used
                and driver == str(r["driver"])
                and abs(lap - int(r["lap"])) <= tolerance
            ):
                used.add(i)
                hits += 1
                break
    return {
        "real": len(real),
        "inferred": len(inferred),
        "hits": hits,
        "precision": round(hits / max(1, len(inferred)), 3),
        "recall": round(hits / max(1, len(real)), 3),
    }
