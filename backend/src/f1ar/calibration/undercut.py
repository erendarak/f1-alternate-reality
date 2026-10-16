"""Undercut attempts, measured the same way in real and simulated races (W7, D-051).

Input: one row per car and lap with ``driver``, ``lap``, ``end_s`` (line-crossing time),
``pit_in`` (the car boxed at the end of this lap) and ``green`` (the lap was raced under green).

An **attempt**: at the end of lap ``L - 1`` car A crosses the line at most ``MAX_GAP_S`` behind
car B on the same lap; A boxes at the end of lap ``L``; B boxes at the end of lap ``L + k``,
``k`` in 1..``MAX_OFFSET``; neither stops otherwise in between, and laps ``L - 1 .. L + k + 1``
are green for both. **Gain** = gap before (A behind B, s) minus gap after both stopped (end of lap
``L + k + 1``): positive = A gained time by stopping first; ``passed`` = A ahead after.
"""

from dataclasses import dataclass

import pandas as pd

MAX_GAP_S = 3.0
MAX_OFFSET = 3


@dataclass(frozen=True)
class Attempt:
    attacker: str
    defender: str
    lap: int
    offset: int
    gap_before_s: float
    gap_after_s: float

    @property
    def gain_s(self) -> float:
        return self.gap_before_s - self.gap_after_s

    @property
    def passed(self) -> bool:
        return self.gap_after_s < 0


def undercut_attempts(laps: pd.DataFrame) -> list[Attempt]:
    by = {(str(r["driver"]), int(r["lap"])): r for r in laps.to_dict("records")}
    pits: dict[str, list[int]] = {}
    for (d, lap), r in by.items():
        if bool(r["pit_in"]):
            pits.setdefault(d, []).append(lap)

    def green(d: str, first: int, last: int) -> bool:
        return all((d, lap) in by and bool(by[(d, lap)]["green"]) for lap in range(first, last + 1))

    out = []
    for a, a_pits in pits.items():
        for lap in a_pits:
            before = by.get((a, lap - 1))
            if before is None:
                continue
            for b, b_pits in pits.items():
                if b == a or (b, lap - 1) not in by:
                    continue
                gap0 = float(before["end_s"]) - float(by[(b, lap - 1)]["end_s"])
                if not 0.0 < gap0 <= MAX_GAP_S:
                    continue
                later = [p for p in b_pits if lap < p <= lap + MAX_OFFSET]
                if not later or any(lap - 1 <= p < lap for p in b_pits):
                    continue
                k = later[0] - lap
                end = lap + k + 1
                if any(lap < p <= end for p in a_pits if p != lap):
                    continue
                if not (green(a, lap - 1, end) and green(b, lap - 1, end)):
                    continue
                if (a, end) not in by or (b, end) not in by:
                    continue
                gap1 = float(by[(a, end)]["end_s"]) - float(by[(b, end)]["end_s"])
                out.append(Attempt(a, b, lap, k, gap0, gap1))
    return out


def real_laps(laps: pd.DataFrame) -> pd.DataFrame:
    """The lap dataset (``build_lap_dataset``) of one race in this module's format."""
    status = laps["track_status_start"].astype(str)
    return pd.DataFrame(
        {
            "driver": laps["driver_code"].astype(str),
            "lap": laps["lap_number"].astype(int),
            "end_s": laps["lap_end_time_s"].astype(float),
            "pit_in": laps["pit_in_lap"].fillna(False).astype(bool),
            "green": ~status.str.contains("4|5|6|7"),
        }
    ).dropna(subset=["end_s"])


def summary(attempts: list[Attempt]) -> dict[str, float | int]:
    if not attempts:
        return {"attempts": 0}
    df = pd.DataFrame(
        [{"gain": a.gain_s, "passed": a.passed, "offset": a.offset} for a in attempts]
    )
    out: dict[str, float | int] = {
        "attempts": len(df),
        "gain_median_s": round(float(df["gain"].median()), 3),
        "gain_mean_s": round(float(df["gain"].mean()), 3),
        "passed_share": round(float(df["passed"].mean()), 3),
    }
    for k in range(1, MAX_OFFSET + 1):
        g = df[df["offset"] == k]["gain"]
        out[f"gain_median_offset{k}_s"] = round(float(g.median()), 3) if len(g) else float("nan")
        out[f"n_offset{k}"] = len(g)
    return out
