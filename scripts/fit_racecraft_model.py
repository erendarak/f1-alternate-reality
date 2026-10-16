"""Fit the racecraft advantage model on counterfactual targets (M5 slice 4b, D-049).

Input: what the car sees (rl_obs_v3; its current-intent features are set to NORMAL, as in the
data, where the collector always raced NORMAL); output: the predicted advantage of committing to
DEFEND / ATTACK for the next laps over NORMAL, under the car's own objective and risk utility
(``training.counterfactual``). MLP on the GPU, Huber loss, races split 80/20 for validation.

Reported on the validation races with their **exact** targets: the mean realised advantage per
decision of the model's choice (pick the best predicted option if it beats NORMAL by
``--margin``), of always attacking / defending, and of an oracle that knows the outcome; by
battle vs clear track, objective and risk appetite.

Usage:
    uv run python scripts/fit_racecraft_model.py --data artifacts/training/crn_v1 \\
        --out models/policies/racecraft_crn_v1.npz --json docs/experiments/fit_racecraft_crn.json
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from f1ar.training.features import RL_V3_NAMES
from f1ar.training.policy_registry import NumpyPolicy

INTENT_IDX = [RL_V3_NAMES.index(f"car.intent.{i}") for i in ("defend", "normal", "attack")]


def load(dirs: list[Path]) -> dict[str, np.ndarray]:
    """``DIR`` or ``DIR:points`` (keep only the POINTS rows: title fights collected before D-050
    were scored with the linear objective and are dropped)."""
    parts = []
    for i, entry in enumerate(dirs):  # race ids made unique per directory
        text = str(entry)
        points_only = text.endswith(":points")
        data = Path(text.removesuffix(":points"))
        for p in sorted(data.glob("part_*.npz")):
            z = dict(np.load(p, allow_pickle=False))
            z["race"] = z["race"] + i * 10_000_000
            if points_only:
                keep = z["objective"] == "POINTS"
                z = {k: v[keep] for k, v in z.items()}
            parts.append(z)
    out = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    out["obs"][:, INTENT_IDX] = (0.0, 1.0, 0.0)  # as collected (NORMAL); also at use time
    return out


def choose(pred: np.ndarray, margin: float) -> np.ndarray:
    """0 = NORMAL, 1 = DEFEND, 2 = ATTACK."""
    best = pred.argmax(axis=1)
    return np.where(pred.max(axis=1) > margin, best + 1, 0)


def realised(adv: np.ndarray, pick: np.ndarray) -> np.ndarray:
    full = np.concatenate([np.zeros((len(adv), 1)), adv], axis=1)
    out: np.ndarray = full[np.arange(len(adv)), pick]
    return out


def report(
    adv: np.ndarray, pred: np.ndarray, margin: float, mask: np.ndarray, names: list[str]
) -> dict[str, Any]:
    a, p = adv[mask], pred[mask]
    if len(a) == 0:
        return {"decisions": 0}
    pick = choose(p, margin)
    gain = realised(a, pick)
    oracle = np.maximum(0.0, a.max(axis=1))
    out: dict[str, Any] = {
        "decisions": int(len(a)),
        "model": round(float(gain.mean()), 5),
        "model_ci95": round(float(1.96 * gain.std() / np.sqrt(len(gain))), 5),
        "oracle": round(float(oracle.mean()), 5),
    }
    for i, name in enumerate(names):
        out[f"model_{name}_share"] = round(float((pick == i + 1).mean()), 4)
        out[f"always_{name}"] = round(float(a[:, i].mean()), 5)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument(
        "--kind", choices=("racecraft", "pit"), default="racecraft", help="what the data holds"
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    d = load(args.data)
    obs, adv = d["obs"].astype(np.float32), d["adv"].astype(np.float32)
    rng = np.random.default_rng(args.seed)
    races = rng.permutation(np.unique(d["race"]))
    n = len(races)
    calib_races, report_races = set(races[: n // 10]), set(races[n // 10 : n * 3 // 10])
    calib = np.array([r in calib_races for r in d["race"]])
    val = np.array([r in report_races for r in d["race"]])
    train = ~calib & ~val
    names = ["defend", "attack"] if args.kind == "racecraft" else ["deviate"]
    print(
        f"{len(obs)} decisions from {n} races: {train.sum()} train, {calib.sum()} calibration "
        f"(epoch and margin), {val.sum()} report"
    )

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = torch.nn.Sequential(
        torch.nn.Linear(obs.shape[1], args.hidden),
        torch.nn.Tanh(),
        torch.nn.Linear(args.hidden, args.hidden),
        torch.nn.Tanh(),
        torch.nn.Linear(args.hidden, adv.shape[1]),
    ).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    x = torch.as_tensor(obs[train], device=device)
    y = torch.as_tensor(adv[train], device=device)
    xv = torch.as_tensor(obs[calib], device=device)
    yv = torch.as_tensor(adv[calib], device=device)
    loss_fn = torch.nn.HuberLoss(delta=0.05 if args.kind == "racecraft" else 0.2)
    best, best_state = float("inf"), None
    for epoch in range(args.epochs):
        perm = torch.randperm(len(x), device=device)
        for i in range(0, len(x), 512):
            idx = perm[i : i + 512]
            loss = loss_fn(net(x[idx]), y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            v = float(loss_fn(net(xv), yv))
        if v < best:
            best, best_state = v, {k: t.clone() for k, t in net.state_dict().items()}
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            print(f"epoch {epoch}: calibration loss {v:.6f}")
    assert best_state is not None
    net.load_state_dict(best_state)
    with torch.no_grad():
        pred_c = net(xv).cpu().numpy()
        pred = net(torch.as_tensor(obs[val], device=device)).cpu().numpy()
    # the margin that maximises the realised gain on the calibration races
    margins = [0.0, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 1e9]
    gains = {m: float(realised(adv[calib], choose(pred_c, m)).mean()) for m in margins}
    margin = max(margins, key=lambda m: gains[m])
    print("calibration gain by margin", {m: round(g, 5) for m, g in gains.items()})
    adv_v = adv[val]
    groups: dict[str, np.ndarray] = {
        "all": np.ones(len(adv_v), dtype=bool),
        "battle": d["battle"][val],
        "clear": ~d["battle"][val],
        **{f"objective={o}": d["objective"][val] == o for o in np.unique(d["objective"])},
        "risk<0.35": d["risk"][val] < 0.35,
        "risk>0.65": d["risk"][val] > 0.65,
        **(
            {
                "rules_box": d["rules_boxes"][val],
                "rules_stay": ~d["rules_boxes"][val],
                "under_sc_vsc": d["neutral"][val],
            }
            if args.kind == "pit"
            else {}
        ),
    }
    result: dict[str, Any] = {
        "decisions": int(len(obs)),
        "races": int(len(races)),
        "kind": args.kind,
        "report_decisions": int(val.sum()),
        "margin": margin,
        "calibration_gain_by_margin": {str(m): round(g, 6) for m, g in gains.items()},
        "calibration_loss": round(best, 6),
        "groups": {k: report(adv_v, pred, margin, m, names) for k, m in groups.items()},
    }
    for k, v in result["groups"].items():
        print(k, v)
    linears = [m for m in net if isinstance(m, torch.nn.Linear)]
    NumpyPolicy(
        weights=tuple(m.weight.detach().cpu().numpy().astype(np.float64) for m in linears),
        biases=tuple(m.bias.detach().cpu().numpy().astype(np.float64) for m in linears),
    ).save(args.out)
    print(f"wrote {args.out}")
    if args.json:
        args.json.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
