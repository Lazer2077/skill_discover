"""Greedy skill composition for 2D target reaching.

The high-level controller treats each discovered skill as a temporally
extended action (an option, Sutton et al. 1999): at every decision step it
predicts each skill's average world-frame outcome from the current pose,
picks the one that most reduces distance-to-target, and replays that skill's
representative action sequence open-loop for H low-level steps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from skill_discovery.envs.isaac_env_wrapper import IsaacEnvWrapper
from skill_discovery.library.skill_library import Skill, SkillLibrary
from skill_discovery.utils.logging import get_logger
from skill_discovery.utils.math_utils import quat_to_yaw

logger = get_logger("skill_composer")


@dataclass
class CompositionResult:
    """Outcome of one target-reaching trial."""

    success: bool
    final_distance: float
    skill_sequence: List[int]
    num_skills_used: int
    energy_proxy: float
    base_positions: np.ndarray  # (T, 3) world-frame trace of env 0
    terminated_early: bool = False
    final_height: float = 0.0
    min_height: float = 0.0


class GreedySkillComposer:
    """One-step-lookahead skill selection toward a 2D target."""

    def __init__(
        self,
        skill_library: SkillLibrary,
        target_threshold: float = 0.5,
        max_high_level_steps: int = 20,
        lambda_energy: float = 0.05,
    ):
        self.library = skill_library
        self.target_threshold = target_threshold
        self.max_high_level_steps = max_high_level_steps
        self.lambda_energy = lambda_energy

    def select_skill(
        self,
        current_state: Dict[str, float],
        target_position: np.ndarray,
        skill_library: Optional[SkillLibrary] = None,
    ) -> Skill:
        """Pick the skill minimizing predicted distance-to-target + energy penalty.

        `current_state` needs keys: x, y, yaw (world frame).
        """
        library = skill_library or self.library
        x, y, yaw = current_state["x"], current_state["y"], current_state["yaw"]
        best_skill, best_cost = None, np.inf
        for skill in library.skills.values():
            px, py, _ = skill.predict_outcome(x, y, yaw)
            cost = float(np.linalg.norm(target_position - np.array([px, py])))
            cost += self.lambda_energy * skill.mean_energy
            if cost < best_cost:
                best_skill, best_cost = skill, cost
        assert best_skill is not None, "Skill library is empty."
        return best_skill

    def rollout(self, env: IsaacEnvWrapper, target_position: np.ndarray) -> CompositionResult:
        """Run one target-reaching trial in env 0 of a (possibly vectorized) env.

        Only env 0 is evaluated; other parallel envs receive the same actions
        (tiled) and are ignored. Termination of env 0 ends the trial.
        """
        env.reset()
        target_position = np.asarray(target_position, dtype=np.float64)

        skill_sequence: List[int] = []
        positions: List[np.ndarray] = []
        energy_total, energy_steps = 0.0, 0

        for _ in range(self.max_high_level_steps):
            state = env.get_robot_state()
            pos = state["base_pos"][0]
            yaw = float(quat_to_yaw(state["base_quat"][0]))
            positions.append(pos.copy())

            distance = float(np.linalg.norm(target_position - pos[:2]))
            if distance < self.target_threshold:
                break

            skill = self.select_skill({"x": pos[0], "y": pos[1], "yaw": yaw}, target_position)
            skill_sequence.append(skill.skill_id)

            aborted = False
            for action in skill.action_sequence:
                actions = np.tile(action, (env.num_envs, 1))
                result = env.step(actions)
                joint_vel = env.get_robot_state()["joint_vel"][0]
                min_dim = min(len(action), len(joint_vel))
                energy_total += float(np.mean(np.abs(action[:min_dim] * joint_vel[:min_dim])))
                energy_steps += 1
                if result.dones[0]:
                    aborted = True  # env 0 fell or timed out; pose was reset
                    break
            if aborted:
                logger.info("Episode terminated mid-skill; continuing from reset pose.")

        state = env.get_robot_state()
        final_pos = state["base_pos"][0]
        positions.append(final_pos.copy())
        final_distance = float(np.linalg.norm(target_position - final_pos[:2]))

        return CompositionResult(
            success=final_distance < self.target_threshold,
            final_distance=final_distance,
            skill_sequence=skill_sequence,
            num_skills_used=len(skill_sequence),
            energy_proxy=energy_total / max(energy_steps, 1),
            base_positions=np.stack(positions),
        )
