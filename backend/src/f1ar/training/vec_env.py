"""A vectorised env with several envs per worker process (M5 slice 3, D-040).

SB3's ``SubprocVecEnv`` runs one env per process in lockstep: every step waits for the slowest
worker. Race decisions vary a lot in cost (a reset simulates the start and the first lap, a
safety car or a pit lap costs more than a green lap), so with 14 workers each process sat idle
about half the time. Here each process steps ``group`` envs in a row per call: the per-call
cost is a sum of ``group`` decisions and varies far less, so the wait for the slowest worker
shrinks. Action masks come back with each step's results (MaskablePPO asks for them every step;
``SubprocVecEnv`` needed one more round trip to every worker for that).

Same semantics as ``SubprocVecEnv``: auto-reset on episode end with the last observation in
``info["terminal_observation"]``; env ``i`` gets seed ``seed + i`` at its first reset.
"""

import multiprocessing as mp
from collections.abc import Callable, Iterable, Sequence
from multiprocessing.connection import Connection
from typing import Any

import gymnasium as gym
import numpy as np
from stable_baselines3.common.vec_env.base_vec_env import (
    CloudpickleWrapper,
    VecEnv,
    VecEnvIndices,
    VecEnvObs,
    VecEnvStepReturn,
)

EnvFactory = Callable[[], gym.Env[Any, Any]]


def _worker(remote: Connection, wrapped: CloudpickleWrapper) -> None:
    envs = [f() for f in wrapped.var]
    seeds: list[int | None] = [None] * len(envs)
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "step":
                out = []
                for env, action in zip(envs, data, strict=True):
                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    info["TimeLimit.truncated"] = truncated and not terminated
                    if done:
                        info["terminal_observation"] = obs
                        obs, reset_info = env.reset()
                        info["reset_info"] = reset_info
                    out.append((obs, reward, done, info, env.action_masks()))
                remote.send(out)
            elif cmd == "seed":
                seeds = list(data)
                remote.send(None)
            elif cmd == "reset":
                res = []
                for i, env in enumerate(envs):
                    obs, info = env.reset(seed=seeds[i])
                    seeds[i] = None
                    res.append((obs, info, env.action_masks()))
                remote.send(res)
            elif cmd == "has_attr":
                remote.send([hasattr(envs[i], data[1]) for i in data[0]])
            elif cmd == "get_attr":
                remote.send([getattr(envs[i], data[1]) for i in data[0]])
            elif cmd == "set_attr":
                for i in data[0]:
                    setattr(envs[i], data[1], data[2])
                remote.send(None)
            elif cmd == "env_method":
                idx, name, args, kwargs = data
                remote.send([getattr(envs[i], name)(*args, **kwargs) for i in idx])
            elif cmd == "close":
                for env in envs:
                    env.close()
                remote.close()
                return
            else:
                raise NotImplementedError(cmd)
    except KeyboardInterrupt:
        pass


class GroupedSubprocVecEnv(VecEnv):
    def __init__(self, env_fns: Sequence[EnvFactory], workers: int) -> None:
        n = len(env_fns)
        if not 1 <= workers <= n:
            raise ValueError("need 1 <= workers <= number of envs")
        self.groups = [list(range(w, n, workers)) for w in range(workers)]
        ctx = mp.get_context("spawn")
        self.remotes: list[Any] = []
        self.processes = []
        for group in self.groups:
            parent, child = ctx.Pipe()
            p = ctx.Process(
                target=_worker,
                args=(child, CloudpickleWrapper([env_fns[i] for i in group])),
                daemon=True,
            )
            p.start()
            child.close()
            self.remotes.append(parent)
            self.processes.append(p)
        self.remotes[0].send(("get_attr", ([0], "observation_space")))
        obs_space = self.remotes[0].recv()[0]
        self.remotes[0].send(("get_attr", ([0], "action_space")))
        act_space = self.remotes[0].recv()[0]
        super().__init__(n, obs_space, act_space)
        self._masks = np.ones((n, int(act_space.n)), dtype=bool)
        self._actions: np.ndarray | None = None
        self._seeds: list[int | None] = [None] * n
        self.closed = False

    # --- core API ---------------------------------------------------------------------------

    def seed(self, seed: int | None = None) -> Sequence[None | int]:
        if seed is None:
            return [None] * self.num_envs
        self._seeds = [seed + i for i in range(self.num_envs)]
        for remote, group in zip(self.remotes, self.groups, strict=True):
            remote.send(("seed", [self._seeds[i] for i in group]))
        for remote in self.remotes:
            remote.recv()
        return list(self._seeds)

    def reset(self) -> VecEnvObs:
        for remote in self.remotes:
            remote.send(("reset", None))
        obs = [None] * self.num_envs
        self.reset_infos = [{} for _ in range(self.num_envs)]
        for remote, group in zip(self.remotes, self.groups, strict=True):
            for i, (o, info, mask) in zip(group, remote.recv(), strict=True):
                obs[i] = o
                self.reset_infos[i] = info
                self._masks[i] = mask
        return np.stack(obs)  # type: ignore[arg-type]

    def step_async(self, actions: np.ndarray) -> None:
        for remote, group in zip(self.remotes, self.groups, strict=True):
            remote.send(("step", [actions[i] for i in group]))

    def step_wait(self) -> VecEnvStepReturn:
        obs: list[Any] = [None] * self.num_envs
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        dones = np.zeros(self.num_envs, dtype=bool)
        infos: list[dict[str, Any]] = [{} for _ in range(self.num_envs)]
        for remote, group in zip(self.remotes, self.groups, strict=True):
            for i, (o, r, d, info, mask) in zip(group, remote.recv(), strict=True):
                obs[i], rewards[i], dones[i], infos[i] = o, r, d, info
                self._masks[i] = mask
        return np.stack(obs), rewards, dones, infos

    def close(self) -> None:
        if self.closed:
            return
        for remote in self.remotes:
            remote.send(("close", None))
        for p in self.processes:
            p.join()
        self.closed = True

    # --- attribute access ---------------------------------------------------------------------

    def _route(self, indices: VecEnvIndices) -> list[tuple[int, list[int], list[int]]]:
        """(worker, positions in its group, global indices) for the requested envs."""
        wanted = set(self._get_indices(indices))
        out = []
        for w, group in enumerate(self.groups):
            pos = [k for k, i in enumerate(group) if i in wanted]
            if pos:
                out.append((w, pos, [group[k] for k in pos]))
        return out

    def _gather(self, cmd: str, indices: VecEnvIndices, *payload: Any) -> list[Any]:
        routes = self._route(indices)
        for w, pos, _ in routes:
            self.remotes[w].send((cmd, (pos, *payload)))
        result: dict[int, Any] = {}
        for w, _, glob in routes:
            values = self.remotes[w].recv()
            if values is not None:
                result.update(zip(glob, values, strict=True))
        return [result[i] for i in sorted(result)]

    def has_attr(self, attr_name: str) -> bool:
        """Asked in the workers (SB3's default fetches the attribute, e.g. a bound method that
        drags the whole env through pickle)."""
        return all(self._gather("has_attr", None, attr_name))

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> list[Any]:
        return self._gather("get_attr", indices, attr_name)

    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        self._gather("set_attr", indices, attr_name, value)

    def env_method(
        self,
        method_name: str,
        *method_args: Any,
        indices: VecEnvIndices = None,
        **method_kwargs: Any,
    ) -> list[Any]:
        if method_name == "action_masks" and not method_args and not method_kwargs:
            return [self._masks[i] for i in self._get_indices(indices)]
        return self._gather("env_method", indices, method_name, method_args, method_kwargs)

    def env_is_wrapped(
        self, wrapper_class: type[gym.Wrapper[Any, Any, Any, Any]], indices: VecEnvIndices = None
    ) -> list[bool]:
        return [False for _ in self._get_indices(indices)]

    def _get_indices(self, indices: VecEnvIndices) -> Iterable[int]:
        if indices is None:
            return range(self.num_envs)
        if isinstance(indices, int):
            return [indices]
        return indices
