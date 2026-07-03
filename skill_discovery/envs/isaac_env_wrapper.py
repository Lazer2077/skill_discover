"""Wrapper around Isaac Lab vectorized locomotion environments.

Responsibilities:
    * create the gym env for a given Isaac Lab task name,
    * expose a uniform numpy-based step/reset API,
    * extract robot base/joint state robustly across Isaac Lab versions,
      with documented fallbacks when a field is unavailable.

Isaac Lab has renamed its python package over time (``omni.isaac.lab`` ->
``isaaclab``); this module tries both. Scripts must create the simulation app
via ``AppLauncher`` BEFORE importing this module's ``IsaacEnvWrapper.create``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np

from skill_discovery.utils.logging import get_logger

logger = get_logger("isaac_env_wrapper")


def launch_app(headless: bool = True) -> Any:
    """Start the Omniverse simulation app. Must run before any isaaclab import.

    Returns the ``SimulationApp`` handle (call ``.close()`` when done).
    """
    try:  # Isaac Lab >= 2.0
        from isaaclab.app import AppLauncher
    except ImportError:
        try:  # Isaac Lab 1.x
            from omni.isaac.lab.app import AppLauncher
        except ImportError as exc:
            raise ImportError(
                "Isaac Lab not found. Run this script through the Isaac Lab "
                "launcher, e.g. `./isaaclab.sh -p scripts/collect_exploration.py ...`"
            ) from exc
    app_launcher = AppLauncher(headless=headless)
    return app_launcher.app


def _import_isaaclab_tasks() -> None:
    """Register Isaac Lab gym tasks (import side effect), across versions."""
    try:
        import isaaclab_tasks  # noqa: F401  (Isaac Lab >= 2.0)
        return
    except ImportError:
        pass
    try:
        import omni.isaac.lab_tasks  # noqa: F401  (Isaac Lab 1.x)
        return
    except ImportError as exc:
        raise ImportError(
            "Could not import Isaac Lab task registry (isaaclab_tasks / "
            "omni.isaac.lab_tasks). Check your Isaac Lab installation."
        ) from exc


def list_locomotion_tasks() -> list:
    """Return registered Isaac task names (useful when task ids differ by version)."""
    import gymnasium as gym

    _import_isaaclab_tasks()
    return sorted(k for k in gym.registry.keys() if k.startswith("Isaac-"))


@dataclass
class StepResult:
    """One vectorized environment step converted to numpy."""

    obs: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    info: Dict[str, Any] = field(default_factory=dict)


class IsaacEnvWrapper:
    """Uniform numpy interface over an Isaac Lab vectorized environment.

    State extraction strategy (in order of preference):
      1. Read directly from the articulation root/joint buffers in
         ``env.unwrapped.scene`` (works for manager-based and direct envs).
      2. If the robot articulation cannot be located, fall back to zeros and
         log a warning once — the pipeline keeps running on obs/actions only.
    """

    # Common names for the robot articulation in Isaac Lab locomotion scenes.
    _ROBOT_KEYS = ("robot", "ant", "unitree_go2", "anymal", "cartpole")

    def __init__(self, env: Any, device: str = "cuda:0"):
        self.env = env
        self.device = device
        self._robot = self._find_robot_articulation()
        self._warned_missing_state = False

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    @classmethod
    def create(cls, task: str, num_envs: int, device: str = "cuda:0") -> "IsaacEnvWrapper":
        """Create an Isaac Lab env by task name. Requires a running SimulationApp."""
        import gymnasium as gym

        _import_isaaclab_tasks()

        try:  # Isaac Lab >= 2.0
            from isaaclab_tasks.utils import parse_env_cfg
        except ImportError:
            from omni.isaac.lab_tasks.utils import parse_env_cfg  # Isaac Lab 1.x

        if task not in gym.registry:
            available = [t for t in list_locomotion_tasks() if "Velocity" in t or "Ant" in t]
            raise ValueError(
                f"Task '{task}' is not registered. Locomotion-like tasks available: {available}"
            )

        env_cfg = parse_env_cfg(task, device=device, num_envs=num_envs)
        env = gym.make(task, cfg=env_cfg)
        logger.info("Created task '%s' with %d envs on %s", task, num_envs, device)
        return cls(env, device=device)

    # ------------------------------------------------------------------
    # Gym-like API (numpy in/out)
    # ------------------------------------------------------------------
    @property
    def num_envs(self) -> int:
        return int(self.env.unwrapped.num_envs)

    @property
    def action_dim(self) -> int:
        space = self.env.unwrapped.single_action_space
        return int(np.prod(space.shape))

    def reset(self) -> np.ndarray:
        obs, _ = self.env.reset()
        return self._obs_to_numpy(obs)

    def step(self, actions: np.ndarray) -> StepResult:
        import torch

        actions_t = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        obs, rew, terminated, truncated, info = self.env.step(actions_t)
        dones = (terminated | truncated) if hasattr(terminated, "__or__") else np.logical_or(
            terminated, truncated
        )
        return StepResult(
            obs=self._obs_to_numpy(obs),
            rewards=self._to_numpy(rew).reshape(self.num_envs),
            dones=self._to_numpy(dones).astype(bool).reshape(self.num_envs),
            info=info if isinstance(info, dict) else {},
        )

    def close(self) -> None:
        self.env.close()

    # ------------------------------------------------------------------
    # Robot state extraction
    # ------------------------------------------------------------------
    def get_robot_state(self) -> Dict[str, np.ndarray]:
        """Return base and joint state, shape (num_envs, ...), in world frame.

        Falls back to zeros (with a one-time warning) if the articulation is
        not found — see class docstring.
        """
        if self._robot is None:
            if not self._warned_missing_state:
                logger.warning(
                    "Robot articulation not found in scene; base/joint state "
                    "will be zeros. Descriptors relying on them will degrade."
                )
                self._warned_missing_state = True
            n = self.num_envs
            return {
                "base_pos": np.zeros((n, 3)),
                "base_quat": np.tile(np.array([1.0, 0, 0, 0]), (n, 1)),
                "base_lin_vel": np.zeros((n, 3)),
                "base_ang_vel": np.zeros((n, 3)),
                "joint_pos": np.zeros((n, 1)),
                "joint_vel": np.zeros((n, 1)),
            }

        data = self._robot.data
        # root_pos_w includes env origins; subtract them so positions are
        # comparable across parallel envs.
        origins = self._to_numpy(self.env.unwrapped.scene.env_origins)
        return {
            "base_pos": self._to_numpy(data.root_pos_w) - origins,
            "base_quat": self._to_numpy(data.root_quat_w),  # (w, x, y, z)
            "base_lin_vel": self._to_numpy(data.root_lin_vel_w),
            "base_ang_vel": self._to_numpy(data.root_ang_vel_w),
            "joint_pos": self._to_numpy(data.joint_pos),
            "joint_vel": self._to_numpy(data.joint_vel),
        }

    def _find_robot_articulation(self) -> Optional[Any]:
        scene = getattr(self.env.unwrapped, "scene", None)
        if scene is None:
            return None
        articulations = getattr(scene, "articulations", {})
        # Prefer well-known names, otherwise take the first articulation.
        for key in self._ROBOT_KEYS:
            if key in articulations:
                return articulations[key]
        if articulations:
            key = next(iter(articulations))
            logger.info("Using articulation '%s' as the robot.", key)
            return articulations[key]
        return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _to_numpy(x: Any) -> np.ndarray:
        if hasattr(x, "detach"):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    def _obs_to_numpy(self, obs: Any) -> np.ndarray:
        """Flatten dict observations ('policy' group first) into (num_envs, obs_dim)."""
        if isinstance(obs, dict):
            if "policy" in obs:
                return self._to_numpy(obs["policy"]).reshape(self.num_envs, -1)
            parts = [self._to_numpy(v).reshape(self.num_envs, -1) for v in obs.values()]
            return np.concatenate(parts, axis=-1)
        return self._to_numpy(obs).reshape(self.num_envs, -1)
