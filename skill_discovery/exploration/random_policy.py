"""Random Gaussian exploration policy with optional temporal smoothing.

V1 deliberately does NOT train a neural policy: broad random exploration is
enough to expose diverse locomotion segments to the discovery pipeline
(cf. Quality-Diversity methods, Cully et al. 2015). Temporal smoothing
(first-order low-pass over actions) produces more coherent gaits than
white-noise actions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RandomExplorationPolicy:
    """Gaussian action noise, optionally low-pass filtered across steps.

    action_t = smoothing * action_{t-1} + (1 - smoothing) * eps,
    eps ~ N(0, action_std^2), clipped to [-clip, clip].
    """

    num_envs: int
    action_dim: int
    action_std: float = 0.5
    action_smoothing: float = 0.2
    action_clip: float = 1.0
    seed: int = 42

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        self._prev_actions = np.zeros((self.num_envs, self.action_dim), dtype=np.float32)

    def act(self, obs: np.ndarray) -> np.ndarray:
        """Sample actions of shape (num_envs, action_dim); `obs` is unused in V1."""
        noise = self._rng.normal(0.0, self.action_std, size=self._prev_actions.shape)
        actions = self.action_smoothing * self._prev_actions + (1.0 - self.action_smoothing) * noise
        actions = np.clip(actions, -self.action_clip, self.action_clip).astype(np.float32)
        self._prev_actions = actions
        return actions

    def reset_env(self, env_ids: np.ndarray) -> None:
        """Reset the smoothing state for envs that just terminated."""
        self._prev_actions[env_ids] = 0.0
