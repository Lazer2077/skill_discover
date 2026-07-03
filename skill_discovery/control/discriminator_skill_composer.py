"""Skill composition gated by a learned state-skill discriminator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from skill_discovery.control.skill_composer import CompositionResult
from skill_discovery.envs.isaac_env_wrapper import IsaacEnvWrapper
from skill_discovery.learning.state_features import env_state_feature
from skill_discovery.learning.state_skill_discriminator import StateSkillDiscriminator
from skill_discovery.library.skill_library import Skill
from skill_discovery.online.online_action_set import OnlineActionSet
from skill_discovery.utils.math_utils import quat_to_yaw


@dataclass
class DiscriminatorDecision:
    skill_id: int
    applicability: float
    predicted_distance: float
    cost: float


class DiscriminatorGuidedSkillComposer:
    """Choose and apply skills that are both goal-useful and dynamically applicable."""

    def __init__(
        self,
        action_set: OnlineActionSet,
        discriminator: StateSkillDiscriminator,
        target_threshold: float = 0.5,
        max_high_level_steps: int = 20,
        lambda_energy: float = 0.05,
        lambda_discriminator: float = 1.0,
        lambda_reliability: float = 0.25,
        lambda_no_progress: float = 1.0,
        min_predicted_progress: float = 0.02,
        applicability_threshold: float | None = None,
        strict_applicability: bool = False,
    ):
        self.action_set = action_set
        self.library = action_set.to_skill_library()
        self.discriminator = discriminator
        self.target_threshold = target_threshold
        self.max_high_level_steps = max_high_level_steps
        self.lambda_energy = lambda_energy
        self.lambda_discriminator = lambda_discriminator
        self.lambda_reliability = lambda_reliability
        self.lambda_no_progress = lambda_no_progress
        self.min_predicted_progress = min_predicted_progress
        self.applicability_threshold = (
            discriminator.threshold if applicability_threshold is None else applicability_threshold
        )
        self.strict_applicability = strict_applicability
        self.last_decisions: List[DiscriminatorDecision] = []

    def select_skill(
        self,
        current_state: Dict[str, float],
        target_position: np.ndarray,
        state_feature: np.ndarray,
    ) -> Skill:
        """Select by distance-to-goal plus energy plus learned applicability penalty."""
        x, y, yaw = current_state["x"], current_state["y"], current_state["yaw"]
        current_distance = float(np.linalg.norm(target_position - np.asarray([x, y])))
        best_skill: Optional[Skill] = None
        best_decision: Optional[DiscriminatorDecision] = None
        fallback_skill: Optional[Skill] = None
        fallback_decision: Optional[DiscriminatorDecision] = None

        for skill in self.library.skills.values():
            online_skill = self.action_set.skills[skill.skill_id]
            applicability = self.discriminator.score(state_feature, online_skill.initial_state_features)
            reliability = min(1.0, np.log1p(max(online_skill.num_segments, 0)) / np.log1p(8.0))
            px, py, _ = skill.predict_outcome(x, y, yaw)
            predicted_distance = float(np.linalg.norm(target_position - np.asarray([px, py])))
            no_progress = max(0.0, predicted_distance - current_distance)
            cost = (
                predicted_distance
                + self.lambda_energy * skill.mean_energy
                + self.lambda_discriminator * (1.0 - applicability)
                + self.lambda_reliability * (1.0 - reliability)
                + self.lambda_no_progress * no_progress
            )
            decision = DiscriminatorDecision(
                skill_id=skill.skill_id,
                applicability=applicability,
                predicted_distance=predicted_distance,
                cost=float(cost),
            )
            if fallback_decision is None or decision.cost < fallback_decision.cost:
                fallback_skill = skill
                fallback_decision = decision
            if self.strict_applicability and applicability < self.applicability_threshold:
                continue
            if best_decision is None or decision.cost < best_decision.cost:
                best_skill = skill
                best_decision = decision

        if best_skill is None:
            assert fallback_skill is not None and fallback_decision is not None
            best_skill = fallback_skill
            best_decision = fallback_decision
        assert best_decision is not None
        self.last_decisions.append(best_decision)
        return best_skill

    def rollout(self, env: IsaacEnvWrapper, target_position: np.ndarray) -> CompositionResult:
        """Run one discriminator-guided target-reaching rollout in env 0."""
        obs = env.reset()
        target_position = np.asarray(target_position, dtype=np.float64)
        self.last_decisions = []

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

            feature = env_state_feature(
                obs,
                state,
                env_id=0,
                include_obs=self.action_set.config.include_obs_in_state_feature,
                max_obs_dim=self.action_set.config.max_obs_dim,
            )
            skill = self.select_skill({"x": pos[0], "y": pos[1], "yaw": yaw}, target_position, feature)
            if self.last_decisions[-1].predicted_distance > distance - self.min_predicted_progress:
                break
            skill_sequence.append(skill.skill_id)

            aborted = False
            for action in skill.action_sequence:
                actions = np.tile(action, (env.num_envs, 1))
                result = env.step(actions)
                obs = result.obs
                joint_vel = env.get_robot_state()["joint_vel"][0]
                min_dim = min(len(action), len(joint_vel))
                energy_total += float(np.mean(np.abs(action[:min_dim] * joint_vel[:min_dim])))
                energy_steps += 1
                if result.dones[0]:
                    aborted = True
                    break
            if aborted:
                obs = env.reset()

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
