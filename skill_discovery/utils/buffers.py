"""Trajectory storage: per-env rollout buffers and pickle save/load helpers."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

# Fields recorded at every simulation step, per environment.
STEP_FIELDS = (
    "obs",
    "actions",
    "rewards",
    "dones",
    "base_pos",
    "base_quat",
    "base_lin_vel",
    "base_ang_vel",
    "joint_pos",
    "joint_vel",
)


class TrajectoryBuffer:
    """Accumulates vectorized env steps and splits them into per-episode trajectories.

    All `add()` inputs are arrays with a leading num_envs dimension. When an
    env's `done` flag is set, the accumulated steps for that env are flushed
    into a finished trajectory dict (see `trajectories`).
    """

    def __init__(self, num_envs: int):
        self.num_envs = num_envs
        self.trajectories: List[Dict[str, Any]] = []
        self._episode_counter = 0
        self._current: List[Dict[str, List[np.ndarray]]] = [
            {f: [] for f in STEP_FIELDS} for _ in range(num_envs)
        ]

    def add(self, step_data: Dict[str, np.ndarray]) -> None:
        """Append one vectorized step. Each value has shape (num_envs, ...)."""
        dones = np.asarray(step_data["dones"]).reshape(self.num_envs)
        for env_id in range(self.num_envs):
            store = self._current[env_id]
            for field in STEP_FIELDS:
                store[field].append(np.asarray(step_data[field][env_id]))
            if dones[env_id]:
                self._flush_env(env_id)

    def _flush_env(self, env_id: int) -> None:
        store = self._current[env_id]
        length = len(store["actions"])
        if length > 0:
            traj: Dict[str, Any] = {f: np.stack(store[f]) for f in STEP_FIELDS}
            traj["episode_id"] = self._episode_counter
            traj["env_id"] = env_id
            traj["timestep"] = np.arange(length)
            self.trajectories.append(traj)
            self._episode_counter += 1
        self._current[env_id] = {f: [] for f in STEP_FIELDS}

    def flush_all(self) -> None:
        """Flush all unfinished episodes (call once at the end of collection)."""
        for env_id in range(self.num_envs):
            self._flush_env(env_id)

    @property
    def num_trajectories(self) -> int:
        return len(self.trajectories)

    @property
    def total_steps(self) -> int:
        return sum(len(t["actions"]) for t in self.trajectories)


def save_pickle(obj: Any, path: str | Path) -> None:
    """Pickle `obj` to `path`, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path: str | Path) -> Any:
    """Load a pickled object, with a clear error if the file is missing."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {path}. Run the earlier pipeline stage first "
            f"(collect -> extract -> cluster -> evaluate)."
        )
    with open(path, "rb") as f:
        return pickle.load(f)
