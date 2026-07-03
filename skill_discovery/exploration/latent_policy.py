"""Latent-conditioned random exploration policy.

Each env holds a latent vector z that biases its action distribution, so
different envs (and different windows in time) explore *different* behavior
modes — a cheap, training-free analogue of skill-conditioned exploration in
DIAYN (Eysenbach et al., 2018) and DADS (Sharma et al., 2019).

The latent maps to a per-joint action offset and amplitude through a fixed
random projection. Latents are resampled every `latent_horizon` steps.

FUTURE (V2+): replace the fixed random projection with a learned
skill-conditioned policy pi(a | s, z) trained with a diversity objective.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LatentExplorationPolicy:
    """Random policy modulated by per-env latent skill vectors."""

    num_envs: int
    action_dim: int
    latent_dim: int = 8
    latent_horizon: int = 64
    action_std: float = 0.4
    action_smoothing: float = 0.2
    action_clip: float = 1.0
    seed: int = 42

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        # Fixed random projections: latent -> per-joint offset and amplitude gain.
        self._proj_offset = self._rng.normal(0, 0.4, size=(self.latent_dim, self.action_dim))
        self._proj_gain = self._rng.normal(0, 0.3, size=(self.latent_dim, self.action_dim))
        self._z = np.zeros((self.num_envs, self.latent_dim), dtype=np.float32)
        self._steps_since_resample = np.zeros(self.num_envs, dtype=np.int64)
        self._prev_actions = np.zeros((self.num_envs, self.action_dim), dtype=np.float32)
        for i in range(self.num_envs):
            self._z[i] = self.sample_skill_latent()

    def sample_skill_latent(self) -> np.ndarray:
        """Sample a unit-norm latent vector."""
        z = self._rng.normal(size=self.latent_dim)
        return (z / (np.linalg.norm(z) + 1e-8)).astype(np.float32)

    def act(self, obs: np.ndarray, z: np.ndarray | None = None) -> np.ndarray:
        """Sample latent-biased actions of shape (num_envs, action_dim)."""
        if z is not None:
            self._z = np.asarray(z, dtype=np.float32).reshape(self.num_envs, self.latent_dim)

        # Periodically resample latents to switch behavior modes mid-episode.
        self._steps_since_resample += 1
        expired = np.where(self._steps_since_resample >= self.latent_horizon)[0]
        for i in expired:
            self._z[i] = self.sample_skill_latent()
            self._steps_since_resample[i] = 0

        offset = self._z @ self._proj_offset
        gain = 1.0 + np.tanh(self._z @ self._proj_gain)  # in (0, 2)
        noise = self._rng.normal(0.0, self.action_std, size=self._prev_actions.shape)
        raw = offset + gain * noise
        actions = self.action_smoothing * self._prev_actions + (1.0 - self.action_smoothing) * raw
        actions = np.clip(actions, -self.action_clip, self.action_clip).astype(np.float32)
        self._prev_actions = actions
        return actions

    def reset_env(self, env_ids: np.ndarray) -> None:
        """Resample latent and clear smoothing state for terminated envs."""
        for i in np.atleast_1d(env_ids):
            self._z[i] = self.sample_skill_latent()
            self._steps_since_resample[i] = 0
            self._prev_actions[i] = 0.0
