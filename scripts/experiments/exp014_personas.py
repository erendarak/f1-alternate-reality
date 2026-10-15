"""EXP-014: which driver traits can 2024 data measure, and do the editorial personas agree?

1. **Reliability** of each measurable trait: per-driver values from odd vs even dry races
   (split-half correlation; Spearman-Brown season reliability ``2r / (1 + r)``): race pace vs
   teammate and lap noise (W2 regression), tyre wear vs teammate (W5), and the big-loss
   ("mistake"?) rate: laps > 2.5 s slower than the fit with no car within 1.5 s ahead (blue-flag
   laps of lapped cars are NOT removed: that rate mostly measures being lapped).
2. **Persona vs data:** Spearman rank correlation between the editorial persona values
   (``configs/drivers``) and the measured traits: consistency vs lap noise, tire_management vs
   tyre wear, overtake/defense skill vs W3 attack/defend, race_pace_skill vs pace vs teammate
   (within-team differences).
3. **Race-level sanity:** 21 dry races x ``--seeds`` per-race seeds with and without the W5
   persona file: stops, passes, retirements, finishing order vs real.

Usage:
    uv run python scripts/experiments/exp014_personas.py --seeds 5 \
        --json docs/experiments/exp014.json
"""

import argparse
import functools
import hashlib
import json
import shutil
import statistics
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr

from f1ar.calibration.driver_pace_fit import race_fit
from f1ar.calibration.persona_fit import PERSONA_FILE, _slopes, load_personas
from f1ar.calibration.racecraft_fit import load_racecraft
from f1ar.calibration.reference_metrics import real_metrics
from f1ar.calibration.simulated_metrics import sim_metrics, spearman
from f1ar.calibration.tyre_life_fit import green_laps
from f1ar.simulation.scenarios import load_event

print = functools.partial(print, flush=True)  # noqa: A001 (progress through pipes)
ROOT = Path(__file__).resolve().parents[2]
SPLITS = ROOT / "data" / "manifests" / "dataset_laps_2024_v1.yaml"
DATASET = ROOT / "data" / "processed" / "datasets" / "laps_2024_v1.parquet"


def event_seed(event: str, k: int) -> int:
    return int(hashlib.sha256(event.encode()).hexdigest()[:6], 16) * 100 + k


def per_race_traits(laps: pd.DataFrame, events: list[str]) -> pd.DataFrame:
    """One row per (race, driver): pace vs teammate, lap noise, wear vs teammate, big-loss rate."""
    rows = []
    for i, ev in enumerate(events):
        r = laps[laps["event_id"] == ev]
        g = green_laps(r)
        if len(g) < 200:
            continue
        terms, res = race_fit(g)
        wear = _slopes(g, "driver_code")
        counts = g.groupby("driver_code").size()
        teams = g.groupby("driver_code")["team_id"].first()
        clean = res[res["gap_ahead_s"].fillna(99.0) > 2.5]
        noise = clean[clean["residual"].abs() < 2.0].groupby("driver_code")["residual"].std()
        big = (
            res[res["gap_ahead_s"].fillna(99.0) > 1.5]
            .groupby("driver_code")["residual"]
            .apply(lambda s: float((s > 2.5).mean()))
        )
        for _, members in teams.groupby(teams):
            pair = [str(d) for d in members.index if counts.get(d, 0) >= 15]
            if len(pair) != 2:
                continue
            pm, wm = (terms[pair[0]] + terms[pair[1]]) / 2, (wear[pair[0]] + wear[pair[1]]) / 2
            for d in pair:
                rows.append(
                    {
                        "event": ev,
                        "half": i % 2,
                        "driver": d,
                        "pace": terms[d] - pm,
                        "noise": float(noise.get(d, np.nan)),
                        "wear": wear[d] - wm,
                        "big": float(big.get(d, np.nan)),
                    }
                )
    return pd.DataFrame(rows)


def reliability(df: pd.DataFrame, col: str) -> dict[str, float]:
    m = df.groupby(["driver", "half"])[col].mean().unstack().dropna()
    r = float(np.corrcoef(m[0], m[1])[0, 1]) if len(m) > 3 else float("nan")
    return {
        "split_half_r": round(r, 3),
        "season_reliability": round(max(0.0, 2 * r / (1 + r)), 3) if r > 0 else 0.0,
        "drivers": len(m),
    }


def personas() -> dict[str, Any]:
    """Season persona values per driver code (editorial seed)."""
    out: dict[str, Any] = {}
    for ev in yaml.safe_load(SPLITS.read_text(encoding="utf-8"))["splits"]["dry_train"]:
        roster, profiles = load_event(ROOT / "configs", ev)
        code = {e.driver_id: e.code for e in roster.entries}
        team = {e.driver_id: e.team_id for e in roster.entries}
        for p in profiles.profiles:
            out[code[p.driver_id]] = {**p.archetype.model_dump(), "team": team[p.driver_id]}
    return out


def agree(a: dict[str, float], b: dict[str, float]) -> dict[str, Any]:
    keys = sorted(set(a) & set(b))
    rho = spearmanr([a[k] for k in keys], [b[k] for k in keys]).statistic
    return {"spearman": round(float(rho), 3), "drivers": len(keys)}


@functools.cache
def _setup(event: str, configs: str) -> Any:
    from f1ar.simulation.scenarios import event_data_dir, setup_from_codes

    roster, profiles = load_event(ROOT / "configs", event)
    return setup_from_codes(
        roster, profiles, None, data_dir=event_data_dir(ROOT, event), configs_dir=Path(configs)
    )


def race(job: tuple[str, str, str, int]) -> dict[str, Any]:
    event, variant, configs, seed = job
    m = sim_metrics(_setup(event, configs), seed)
    stops = list(m.stops.values())
    return {
        "event": event,
        "variant": variant,
        "stops": statistics.fmean(stops) if stops else float("nan"),
        "passes": m.overtakes,
        "retirements": m.retirements,
        "order": m.order,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--workers", type=int, default=14)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    splits = yaml.safe_load(SPLITS.read_text(encoding="utf-8"))["splits"]
    train, test = list(splits["dry_train"]), list(splits["dry_test"])
    events = train + test
    laps = pd.read_parquet(DATASET)
    traits = per_race_traits(laps, events)
    rel = {
        "all_dry": {c: reliability(traits, c) for c in ("pace", "noise", "wear", "big")},
        "dry_train": {
            c: reliability(traits[traits["event"].isin(train)], c)
            for c in ("pace", "noise", "wear", "big")
        },
    }
    print("reliability", json.dumps(rel))
    pers = personas()
    season = traits.groupby("driver")[["pace", "noise", "wear"]].mean()
    rc = load_racecraft(ROOT / "configs")
    assert rc is not None
    att = {d: v["attack_s"] for d, v in rc["overtaking"]["drivers"].items()}
    dfd = {d: v["defend_s"] for d, v in rc["overtaking"]["drivers"].items()}
    agreement = {
        "consistency_vs_low_noise": agree(
            {d: v["consistency"] for d, v in pers.items()},
            {str(d): -float(v) for d, v in season["noise"].items()},
        ),
        "tire_management_vs_low_wear": agree(
            {d: v["tire_management"] for d, v in pers.items()},
            {str(d): -float(v) for d, v in season["wear"].items()},
        ),
        "overtake_skill_vs_attack": agree({d: v["overtake_skill"] for d, v in pers.items()}, att),
        "defense_skill_vs_defend": agree({d: v["defense_skill"] for d, v in pers.items()}, dfd),
    }
    # race_pace_skill: within-team differences vs measured pace differences
    pairs = []
    for _, members in pd.Series({d: v["team"] for d, v in pers.items()}).groupby(
        lambda d: pers[d]["team"]
    ):
        ds = [d for d in members.index if d in season.index]
        if len(ds) == 2:
            a, b = ds
            pairs.append(
                (
                    pers[a]["race_pace_skill"] - pers[b]["race_pace_skill"],
                    -(season["pace"][a] - season["pace"][b]),
                )
            )
    if pairs:
        x, y = zip(*pairs, strict=True)
        agreement["race_pace_skill_vs_teammate_gap"] = {
            "spearman": round(float(spearmanr(x, y).statistic), 3),
            "pairs": len(pairs),
        }
    print("persona vs data", json.dumps(agreement))
    old = Path(tempfile.mkdtemp(prefix="exp014_configs_"))
    shutil.copytree(ROOT / "configs", old, dirs_exist_ok=True)
    (old / PERSONA_FILE).unlink()
    jobs = [
        (e, v, c, event_seed(e, k))
        for e in events
        for v, c in (("w5", str(ROOT / "configs")), ("v0.13", str(old)))
        for k in range(1, args.seeds + 1)
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(race, jobs, chunksize=2))
    shutil.rmtree(old, ignore_errors=True)
    real = {e: real_metrics(ROOT / "data" / "processed" / e) for e in events}
    sanity = {}
    for v in ("w5", "v0.13"):
        mine = [r for r in rows if r["variant"] == v]
        sanity[v] = {
            "stops": round(statistics.fmean(r["stops"] for r in mine), 3),
            "passes": round(statistics.fmean(r["passes"] for r in mine), 1),
            "retirements": round(statistics.fmean(r["retirements"] for r in mine), 2),
            "spearman": round(
                statistics.fmean(spearman(r["order"], real[r["event"]].order) for r in mine), 3
            ),
        }
    print("races", json.dumps(sanity))
    applied = load_personas(ROOT / "configs")
    result = {
        "reliability": rel,
        "persona_vs_data": agreement,
        "races": sanity,
        "tire_management_applied": bool(applied and applied["wear"]["reliability_drivers"] >= 0.3),
    }
    if args.json:
        args.json.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
