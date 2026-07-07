"""Compose learned closed-loop skill policies for target reaching."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from skill_discovery.control.archive_chunk_composer import ArchiveChunkComposer
from skill_discovery.control.skill_composer import CompositionResult
from skill_discovery.envs.isaac_env_wrapper import IsaacEnvWrapper
from skill_discovery.learning.state_features import env_state_feature
from skill_discovery.learning.skill_policy import SkillPolicy
from skill_discovery.online.online_action_set import OnlineActionSet
from skill_discovery.utils.math_utils import quat_to_yaw, world_to_body_2d

FeatureSlice = Tuple[int, int]


class SkillPolicyComposer:
    """High-level archive selection with low-level closed-loop skill policy."""

    def __init__(
        self,
        action_set: OnlineActionSet,
        skill_policy: SkillPolicy,
        target_threshold: float = 0.5,
        max_high_level_steps: int = 20,
        execution_horizon: int = 8,
        lambda_state: float = 0.25,
        lambda_energy: float = 0.05,
        lambda_stability: float = 0.5,
        lambda_no_progress: float = 1.5,
        lambda_utility: float = 0.1,
        lambda_progress: float = 0.25,
        applicability_temperature: float = 1.5,
        k_nearest: int = 0,
        condition_label_mode: str | None = None,
        feature_zero_slices: Sequence[FeatureSlice] | None = None,
        relative_target: bool = False,
        policy_blend: float = 1.0,
        backward_policy_blend: float | None = None,
        backward_target_x_threshold: float = -0.05,
        scale_motion_to_execution_horizon: bool = True,
        min_final_height_fraction: float = 0.0,
        policy_command_slice: FeatureSlice | None = None,
        policy_command_gain: float = 1.0,
        policy_command_max: float = 1.0,
    ):
        condition_label_mode = condition_label_mode or getattr(skill_policy, "label_mode", "skill")
        if condition_label_mode not in {"skill", "archive"}:
            raise ValueError("condition_label_mode must be 'skill' or 'archive'.")
        self.action_set = action_set
        self.skill_policy = skill_policy
        self.condition_label_mode = condition_label_mode
        self.relative_target = bool(relative_target)
        self.policy_blend = float(np.clip(policy_blend, 0.0, 1.0))
        self.backward_policy_blend = (
            None if backward_policy_blend is None else float(np.clip(backward_policy_blend, 0.0, 1.0))
        )
        self.backward_target_x_threshold = float(backward_target_x_threshold)
        self.min_final_height_fraction = max(0.0, float(min_final_height_fraction))
        self.policy_command_slice = policy_command_slice
        self.policy_command_gain = float(policy_command_gain)
        self.policy_command_max = max(float(policy_command_max), 1e-6)
        self.target_threshold = target_threshold
        self.max_high_level_steps = max_high_level_steps
        self.execution_horizon = max(1, int(execution_horizon))
        self.selector = ArchiveChunkComposer(
            action_set=action_set,
            target_threshold=target_threshold,
            max_high_level_steps=max_high_level_steps,
            execution_horizon=execution_horizon,
            lambda_state=lambda_state,
            lambda_energy=lambda_energy,
            lambda_stability=lambda_stability,
            lambda_no_progress=lambda_no_progress,
            lambda_utility=lambda_utility,
            lambda_progress=lambda_progress,
            applicability_temperature=applicability_temperature,
            k_nearest=k_nearest,
            feature_zero_slices=feature_zero_slices,
            scale_motion_to_execution_horizon=scale_motion_to_execution_horizon,
            min_final_height_fraction=min_final_height_fraction,
        )
        self.last_decisions = []

    @property
    def archive_size(self) -> int:
        return self.selector.archive_size

    def _policy_observation(
        self,
        obs_row: np.ndarray,
        pos_xy: np.ndarray,
        yaw: float,
        target_position: np.ndarray,
    ) -> np.ndarray:
        obs_out = np.asarray(obs_row, dtype=np.float32).copy()
        if self.policy_command_slice is None:
            return obs_out
        start, end = self.policy_command_slice
        if start >= len(obs_out):
            return obs_out
        local_target = world_to_body_2d(target_position - pos_xy, yaw)
        command = np.zeros(max(end - start, 0), dtype=np.float32)
        if len(command) >= 2:
            command[:2] = np.clip(
                local_target[:2] * self.policy_command_gain,
                -self.policy_command_max,
                self.policy_command_max,
            )
        if len(command) >= 3:
            heading_error = float(np.arctan2(local_target[1], max(local_target[0], 1e-6)))
            command[2] = float(np.clip(heading_error, -self.policy_command_max, self.policy_command_max))
        obs_out[start : min(end, len(obs_out))] = command[: max(0, min(end, len(obs_out)) - start)]
        return obs_out

    def rollout(self, env: IsaacEnvWrapper, target_position: np.ndarray) -> CompositionResult:
        obs = env.reset()
        start_height = float(env.get_robot_state()["base_pos"][0][2])
        target_position = np.asarray(target_position, dtype=np.float64)
        if self.relative_target:
            start_pos = env.get_robot_state()["base_pos"][0]
            target_position = start_pos[:2].astype(np.float64) + target_position
        self.last_decisions = []

        selected_skills: List[int] = []
        positions: List[np.ndarray] = []
        energy_total, energy_steps = 0.0, 0
        terminated_early = False

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
            decision = self.selector.select_chunk((float(pos[0]), float(pos[1]), yaw), target_position, feature)
            skill_id = int(max(decision.skill_id, 0))
            condition_id = int(decision.archive_index if self.condition_label_mode == "archive" else skill_id)
            decision.executed_steps = self.execution_horizon
            self.last_decisions.append(decision)
            selected_skills.append(condition_id)
            segment = self.action_set._archive_segments[decision.archive_index]
            action_seq = np.asarray(segment.get("action_seq", []), dtype=np.float32)
            segment_len = len(action_seq)
            phase_denominator = max(segment_len - 1, 1)
            active_blend = self.policy_blend
            if self.backward_policy_blend is not None:
                local_target = world_to_body_2d(target_position - pos[:2], yaw)
                if float(local_target[0]) < self.backward_target_x_threshold:
                    active_blend = self.backward_policy_blend

            aborted = False
            for low_step in range(self.execution_horizon):
                low_state = env.get_robot_state()
                low_pos = low_state["base_pos"][0]
                low_yaw = float(quat_to_yaw(low_state["base_quat"][0]))
                policy_obs = self._policy_observation(obs[0], low_pos[:2], low_yaw, target_position)
                policy_action = self.skill_policy.predict(
                    policy_obs,
                    condition_id,
                    phase=float(low_step) / float(phase_denominator),
                )
                if active_blend < 1.0 and segment_len > 0:
                    replay_action = action_seq[min(low_step, segment_len - 1)]
                    action = active_blend * policy_action + (1.0 - active_blend) * replay_action
                    action = action.astype(np.float32)
                else:
                    action = policy_action
                result = env.step(np.tile(action, (env.num_envs, 1)))
                obs = result.obs
                joint_vel = env.get_robot_state()["joint_vel"][0]
                min_dim = min(len(action), len(joint_vel))
                energy_total += float(np.mean(np.abs(action[:min_dim] * joint_vel[:min_dim])))
                energy_steps += 1
                if result.dones[0]:
                    terminated_early = True
                    aborted = True
                    break
            if aborted:
                break

        state = env.get_robot_state()
        final_pos = state["base_pos"][0]
        positions.append(final_pos.copy())
        final_distance = float(np.linalg.norm(target_position - final_pos[:2]))
        final_height = float(final_pos[2])
        min_height = float(np.min(np.stack(positions)[:, 2]))
        height_ok = final_height >= self.min_final_height_fraction * max(start_height, 1e-6)
        return CompositionResult(
            success=(not terminated_early) and height_ok and final_distance < self.target_threshold,
            final_distance=final_distance,
            skill_sequence=selected_skills,
            num_skills_used=len(selected_skills),
            energy_proxy=energy_total / max(energy_steps, 1),
            base_positions=np.stack(positions),
            terminated_early=terminated_early,
            final_height=final_height,
            min_height=min_height,
        )
