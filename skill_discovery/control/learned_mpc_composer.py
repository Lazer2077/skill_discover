"""Receding-horizon MPC using learned skill outcome and success models."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import List

import numpy as np

from skill_discovery.control.skill_composer import CompositionResult
from skill_discovery.envs.isaac_env_wrapper import IsaacEnvWrapper
from skill_discovery.learning.skill_outcome_model import SkillOutcomeModel
from skill_discovery.learning.state_features import env_state_feature
from skill_discovery.learning.state_skill_discriminator import StateSkillDiscriminator
from skill_discovery.online.online_action_set import OnlineActionSet
from skill_discovery.utils.math_utils import body_to_world_2d, quat_to_yaw, wrap_angle


@dataclass
class LearnedMPCDecision:
    skill_sequence: List[int]
    cost: float
    first_success_prob: float
    first_applicability: float
    predicted_final_distance: float


class LearnedMPCSkillComposer:
    """Execute skills selected by a learned skill-level dynamics model."""

    def __init__(
        self,
        action_set: OnlineActionSet,
        discriminator: StateSkillDiscriminator,
        outcome_model: SkillOutcomeModel,
        horizon: int = 3,
        target_threshold: float = 0.5,
        max_high_level_steps: int = 20,
        lambda_energy: float = 0.05,
        lambda_discriminator: float = 0.5,
        lambda_success: float = 1.0,
        lambda_reliability: float = 0.25,
    ):
        self.action_set = action_set
        self.library = action_set.to_skill_library()
        self.discriminator = discriminator
        self.outcome_model = outcome_model
        self.horizon = horizon
        self.target_threshold = target_threshold
        self.max_high_level_steps = max_high_level_steps
        self.lambda_energy = lambda_energy
        self.lambda_discriminator = lambda_discriminator
        self.lambda_success = lambda_success
        self.lambda_reliability = lambda_reliability
        self.last_decisions: List[LearnedMPCDecision] = []

    def plan(self, pose: tuple[float, float, float], target_xy: np.ndarray, state_feature: np.ndarray) -> LearnedMPCDecision:
        skill_ids = self.action_set.skill_ids
        if not skill_ids:
            raise RuntimeError("Online action set is empty.")

        # Cache one-step predictions from the current state feature. Receding
        # horizon replans from real state after every executed skill.
        pred = self.outcome_model.predict(state_feature, np.asarray(skill_ids, dtype=np.int64))
        by_skill = {sid: i for i, sid in enumerate(skill_ids)}
        applicability = {
            sid: self.discriminator.score(state_feature, self.action_set.skills[sid].initial_state_features)
            for sid in skill_ids
        }
        reliability = {
            sid: min(1.0, np.log1p(max(self.action_set.skills[sid].num_segments, 0)) / np.log1p(8.0))
            for sid in skill_ids
        }

        best_seq: List[int] = []
        best_cost = float("inf")
        best_final_distance = float("inf")
        for seq in product(skill_ids, repeat=max(1, self.horizon)):
            x, y, yaw = pose
            cost = 0.0
            final_distance = float("inf")
            for step, sid in enumerate(seq):
                i = by_skill[sid]
                dxy = body_to_world_2d(np.asarray([pred["delta_x"][i], pred["delta_y"][i]]), yaw)
                x += float(dxy[0])
                y += float(dxy[1])
                yaw = float(wrap_angle(yaw + pred["delta_yaw"][i]))
                final_distance = float(np.linalg.norm(target_xy - np.asarray([x, y])))
                cost += (
                    final_distance
                    + self.lambda_energy * float(pred["energy"][i])
                    + self.lambda_success * (1.0 - float(pred["success_prob"][i]))
                    + self.lambda_discriminator * (1.0 - applicability[sid])
                    + self.lambda_reliability * (1.0 - reliability[sid])
                )
                # Discount later steps mildly; first-step quality matters most for
                # receding-horizon control with open-loop action chunks.
                if step > 0:
                    cost *= 0.95
            if cost < best_cost:
                best_cost = float(cost)
                best_seq = list(seq)
                best_final_distance = final_distance

        first = best_seq[0]
        first_idx = by_skill[first]
        return LearnedMPCDecision(
            skill_sequence=best_seq,
            cost=best_cost,
            first_success_prob=float(pred["success_prob"][first_idx]),
            first_applicability=float(applicability[first]),
            predicted_final_distance=best_final_distance,
        )

    def rollout(self, env: IsaacEnvWrapper, target_position: np.ndarray) -> CompositionResult:
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
            decision = self.plan((float(pos[0]), float(pos[1]), yaw), target_position, feature)
            self.last_decisions.append(decision)
            skill = self.library.get_skill(decision.skill_sequence[0])
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
