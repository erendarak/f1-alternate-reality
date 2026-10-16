"""Trained strategy policies: the registry, pinning and a torch-free runtime (M5 slice 3, D-040).

A trained policy is two tracked files:

- ``configs/policies/<policy_id>.yaml``: the **manifest**: what the policy is (algorithm,
  architecture), what it was trained on and how (config, seeds, events, steps, git commit),
  what it is pinned to (observation schema + feature-name hash, action count, regulation set,
  engine version) and its **lineage** (parent policy, season, profile revision: the fields
  offseason learning (M9) will fill), plus evaluation results once measured.
- ``models/policies/<policy_id>.npz``: the actor weights (an MLP), checked against the manifest's
  SHA-256 on every load.

The runtime (``NumpyPolicy``) is a plain numpy forward pass, so the app and the engine can run a
pinned policy without PyTorch; training (``f1ar.training.ppo``) exports to it and a test checks
that both give the same logits. A policy refuses to load if its hash, observation schema,
feature list, action count or regulation set doesn't match what the caller runs. The weights are
``learned_parameter`` (SourceType); everything they were trained against (personas, utilities,
team principal rules, PUSH/CONSERVE effects) keeps its own label.
"""

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from f1ar.domain.provenance import SourceType
from f1ar.training.actions import LEGACY_N_ACTIONS, N_ACTIONS, legacy_to_current
from f1ar.training.features import RL_FEATURE_NAMES, RL_FEATURE_SETS

MANIFEST_DIR = Path("configs") / "policies"
ARTIFACT_DIR = Path("models") / "policies"
REGISTRY_SCHEMA = 1
ACTIVATIONS = ("tanh", "relu")


class PolicyLoadError(ValueError):
    """The policy doesn't match its manifest or what the caller runs."""


def feature_names_hash(names: tuple[str, ...] = RL_FEATURE_NAMES) -> str:
    return "sha256:" + hashlib.sha256("\n".join(names).encode()).hexdigest()


def file_sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class NumpyPolicy:
    """The actor MLP: ``hidden = act(W x + b)`` per layer, then ``logits = W h + b``.
    ``weights[i]`` has shape (out, in), as in torch's ``nn.Linear``."""

    weights: tuple[np.ndarray, ...]
    biases: tuple[np.ndarray, ...]
    activation: str = "tanh"

    def __post_init__(self) -> None:
        if self.activation not in ACTIVATIONS:
            raise PolicyLoadError(f"unknown activation {self.activation}")
        if len(self.weights) != len(self.biases) or not self.weights:
            raise PolicyLoadError("weights and biases don't match")

    @property
    def n_inputs(self) -> int:
        return int(self.weights[0].shape[1])

    @property
    def n_actions(self) -> int:
        return int(self.weights[-1].shape[0])

    def logits(self, obs: np.ndarray) -> np.ndarray:
        h = np.asarray(obs, dtype=np.float64)
        for w, b in zip(self.weights[:-1], self.biases[:-1], strict=True):
            h = h @ w.T + b
            h = np.tanh(h) if self.activation == "tanh" else np.maximum(h, 0.0)
        out: np.ndarray = h @ self.weights[-1].T + self.biases[-1]
        if out.shape[-1] == LEGACY_N_ACTIONS:  # slices 1-3b: no intent output; it raced NORMAL
            full = np.full((*out.shape[:-1], N_ACTIONS), -np.inf)
            full[..., [legacy_to_current(a) for a in range(LEGACY_N_ACTIONS)]] = out
            return full
        return out

    def probabilities(self, obs: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Action probabilities with illegal actions at 0 (as MaskablePPO samples them)."""
        z = np.where(mask, self.logits(obs), -np.inf)
        z = z - z.max(axis=-1, keepdims=True)
        p: np.ndarray = np.exp(z)
        out: np.ndarray = p / p.sum(axis=-1, keepdims=True)
        return out

    def act(self, obs: np.ndarray, mask: np.ndarray) -> int:
        """The most likely legal action (deterministic: worlds and evaluation use this)."""
        if not mask.any():
            raise ValueError("no legal action")
        return int(np.argmax(np.where(mask, self.logits(obs), -np.inf)))

    def save(self, path: Path) -> None:
        arrays: dict[str, Any] = {f"w{i}": w for i, w in enumerate(self.weights)}
        arrays |= {f"b{i}": b for i, b in enumerate(self.biases)}
        arrays["activation"] = np.array(self.activation)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            np.savez(f, **arrays)

    @classmethod
    def load(cls, path: Path) -> "NumpyPolicy":
        with np.load(path, allow_pickle=False) as z:
            n = sum(1 for k in z.files if k.startswith("w"))
            return cls(
                weights=tuple(np.array(z[f"w{i}"], dtype=np.float64) for i in range(n)),
                biases=tuple(np.array(z[f"b{i}"], dtype=np.float64) for i in range(n)),
                activation=str(z["activation"]),
            )


@dataclass(frozen=True)
class PolicyManifest:
    policy_id: str
    """``<name>_v<version>``; the id a world pins."""
    algorithm: str
    architecture: str
    artifact: str
    """Repo-relative path of the ``.npz``."""
    artifact_sha256: str
    observation_schema: str
    feature_names_sha256: str
    n_features: int
    n_actions: int
    regulations: str
    engine_version: str
    training: Mapping[str, Any]
    lineage: Mapping[str, Any]
    evaluation: Mapping[str, Any]
    notes: tuple[str, ...] = ()
    source_type: str = SourceType.LEARNED_PARAMETER.value
    registry_schema: int = REGISTRY_SCHEMA

    def to_yaml(self) -> str:
        data = {
            "registry_schema": self.registry_schema,
            "policy_id": self.policy_id,
            "source_type": self.source_type,
            "algorithm": self.algorithm,
            "architecture": self.architecture,
            "artifact": self.artifact,
            "artifact_sha256": self.artifact_sha256,
            "pinned_to": {
                "observation_schema": self.observation_schema,
                "feature_names_sha256": self.feature_names_sha256,
                "n_features": self.n_features,
                "n_actions": self.n_actions,
                "regulations": self.regulations,
                "engine_version": self.engine_version,
            },
            "training": _plain(self.training),
            "lineage": _plain(self.lineage),
            "evaluation": _plain(self.evaluation),
            "notes": list(self.notes),
        }
        return yaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=100)

    @classmethod
    def from_yaml(cls, text: str) -> "PolicyManifest":
        d = yaml.safe_load(text)
        p = d["pinned_to"]
        return cls(
            policy_id=d["policy_id"],
            algorithm=d["algorithm"],
            architecture=d["architecture"],
            artifact=d["artifact"],
            artifact_sha256=d["artifact_sha256"],
            observation_schema=p["observation_schema"],
            feature_names_sha256=p["feature_names_sha256"],
            n_features=int(p["n_features"]),
            n_actions=int(p["n_actions"]),
            regulations=p["regulations"],
            engine_version=p["engine_version"],
            training=d.get("training") or {},
            lineage=d.get("lineage") or {},
            evaluation=d.get("evaluation") or {},
            notes=tuple(d.get("notes") or ()),
            source_type=d.get("source_type", SourceType.LEARNED_PARAMETER.value),
            registry_schema=int(d.get("registry_schema", REGISTRY_SCHEMA)),
        )


def _plain(value: Any) -> Any:
    """JSON-able plain types for YAML (tuples -> lists, numpy scalars -> Python)."""
    return json.loads(json.dumps(value, default=lambda o: o.item() if hasattr(o, "item") else o))


def manifest_path(root: Path, policy_id: str) -> Path:
    return root / MANIFEST_DIR / f"{policy_id}.yaml"


def register_policy(
    root: Path, policy: NumpyPolicy, manifest_fields: Mapping[str, Any]
) -> PolicyManifest:
    """Write the artifact and its manifest; returns the manifest (hash filled in)."""
    policy_id = str(manifest_fields["policy_id"])
    artifact = ARTIFACT_DIR / f"{policy_id}.npz"
    policy.save(root / artifact)
    manifest = PolicyManifest(
        artifact=artifact.as_posix(),
        artifact_sha256=file_sha256(root / artifact),
        n_features=policy.n_inputs,
        n_actions=policy.n_actions,
        **manifest_fields,
    )
    path = manifest_path(root, policy_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(manifest.to_yaml(), encoding="utf-8", newline="\n")
    return manifest


def update_evaluation(root: Path, policy_id: str, evaluation: Mapping[str, Any]) -> None:
    """Record evaluation results in a policy's manifest (the weights don't change)."""
    path = manifest_path(root, policy_id)
    m = PolicyManifest.from_yaml(path.read_text(encoding="utf-8"))
    merged = {**m.evaluation, **evaluation}
    m2 = PolicyManifest(**{**m.__dict__, "evaluation": merged})
    path.write_text(m2.to_yaml(), encoding="utf-8", newline="\n")


@dataclass(frozen=True)
class LoadedPolicy:
    manifest: PolicyManifest
    policy: NumpyPolicy


def load_policy(
    root: Path,
    policy_id: str,
    *,
    regulations: str | None = None,
    engine_version: str | None = None,
) -> LoadedPolicy:
    """Load a registered policy, refusing any mismatch (hash, observation schema, features,
    actions; regulation set and engine version when the caller names them)."""
    path = manifest_path(root, policy_id)
    if not path.exists():
        raise PolicyLoadError(f"no manifest for policy {policy_id} ({path})")
    m = PolicyManifest.from_yaml(path.read_text(encoding="utf-8"))
    problems = []
    if m.policy_id != policy_id:
        problems.append(f"manifest is for {m.policy_id}")
    artifact = root / m.artifact
    if not artifact.exists():
        raise PolicyLoadError(f"{policy_id}: artifact {m.artifact} is missing")
    if file_sha256(artifact) != m.artifact_sha256:
        problems.append("artifact hash doesn't match the manifest")
    if m.observation_schema not in RL_FEATURE_SETS:
        problems.append(
            f"observation {m.observation_schema}, this code builds {sorted(RL_FEATURE_SETS)}"
        )
    elif m.feature_names_sha256 != feature_names_hash(RL_FEATURE_SETS[m.observation_schema]):
        problems.append("feature list differs from this code's")
    if m.n_actions not in (N_ACTIONS, LEGACY_N_ACTIONS):
        problems.append(f"{m.n_actions} actions, this code has {N_ACTIONS}")
    if regulations is not None and m.regulations != regulations:
        problems.append(f"trained under {m.regulations}, the world runs {regulations}")
    if engine_version is not None and m.engine_version != engine_version:
        problems.append(f"trained on {m.engine_version}, the world runs {engine_version}")
    if problems:
        raise PolicyLoadError(f"{policy_id}: " + "; ".join(problems))
    policy = NumpyPolicy.load(artifact)
    if policy.n_inputs != m.n_features or policy.n_actions != m.n_actions:
        raise PolicyLoadError(f"{policy_id}: weight shapes don't match the manifest")
    return LoadedPolicy(m, policy)


def list_policies(root: Path) -> list[PolicyManifest]:
    return [
        PolicyManifest.from_yaml(p.read_text(encoding="utf-8"))
        for p in sorted((root / MANIFEST_DIR).glob("*.yaml"))
    ]
