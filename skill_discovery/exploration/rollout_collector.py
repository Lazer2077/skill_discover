"""Collect exploration rollouts from an Isaac Lab environment."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
from tqdm import tqdm

from skill_discovery.envs.isaac_env_wrapper import IsaacEnvWrapper
from skill_discovery.utils.buffers import TrajectoryBuffer
from skill_discovery.utils.logging import get_logger

logger = get_logger("rollout_collector")


class RolloutCollector:
    """Runs an exploration policy in a vectorized env and stores trajectories.

    `total_steps` counts *env-steps summed over all parallel envs*, so wall
    time is roughly total_steps / num_envs simulation steps.
    """

    def __init__(self, env: IsaacEnvWrapper, policy: Any):
        self.env = env
        self.policy = policy

    def collect(self, total_steps: int) -> List[Dict[str, np.ndarray]]:
        """Collect at least `total_steps` env-steps; returns per-episode trajectories."""
        num_envs = self.env.num_envs
        sim_steps = int(np.ceil(total_steps / num_envs))
        buffer = TrajectoryBuffer(num_envs)

        obs = self.env.reset()
        logger.info("Collecting %d env-steps (%d sim steps x %d envs)...", total_steps, sim_steps, num_envs)

        for _ in tqdm(range(sim_steps), desc="collect"):
            actions = self.policy.act(obs)
            state = self.env.get_robot_state()  # state BEFORE stepping, aligned with action
            result = self.env.step(actions)
            buffer.add(
                {
                    "obs": obs,
                    "actions": actions,
                    "rewards": result.rewards,
                    "dones": result.dones,
                    **state,
                }
            )
            obs = result.obs
            done_ids = np.where(result.dones)[0]
            if len(done_ids) and hasattr(self.policy, "reset_env"):
                self.policy.reset_env(done_ids)

        buffer.flush_all()
        logger.info(
            "Collected %d trajectories, %d total steps.",
            buffer.num_trajectories,
            buffer.total_steps,
        )
        return buffer.trajectories
