"""Closed-loop control by selecting from the online action archive.

The clustered skill library is useful for summaries, but locomotion control is
often too feedback-sensitive for replaying one 32-step representative skill
open-loop.  This controller keeps the same discovered action-set idea but
selects directly from the retained archive at every high-level step:

* goal usefulness comes from each archived segment's body-frame displacement,
* dynamic applicability comes from nearest initial-state feature distance,
* only a short action prefix is executed before replanning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

from skill_discovery.control.skill_composer import CompositionResult
from skill_discovery.descriptors.locomotion_descriptors import DESCRIPTOR_NAMES
from skill_discovery.envs.isaac_env_wrapper import IsaacEnvWrapper
from skill_discovery.learning.state_features import env_state_feature
from skill_discovery.online.online_action_set import OnlineActionSet
from skill_discovery.utils.math_utils import body_to_world_2d, quat_to_yaw

FeatureSlice = Tuple[int, int]


@dataclass
class ArchiveChunkDecision:
    archive_index: int
    skill_id: int
    cost: float
    current_distance: float
    predicted_distance: float
    predicted_progress: float
    state_distance: float
    applicability: float
    utility: float
    energy: float
    stability: float
    executed_steps: int = 0


class ArchiveChunkComposer:
    """State-conditioned nearest/archive controller for target reaching."""

    def __init__(
        self,
        action_set: OnlineActionSet,
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
        feature_zero_slices: Sequence[FeatureSlice] | None = None,
        relative_target: bool = False,
        scale_motion_to_execution_horizon: bool = True,
        min_final_height_fraction: float = 0.0,
    ):
        if not action_set._archive_segments:
            raise ValueError("OnlineActionSet has no archive segments to control from.")
        if not action_set._archive_initial_features:
            raise ValueError("OnlineActionSet has no archive initial-state features.")
        self.action_set = action_set
        self.target_threshold = target_threshold
        self.max_high_level_steps = max_high_level_steps
        self.execution_horizon = max(1, int(execution_horizon))
        self.lambda_state = lambda_state
        self.lambda_energy = lambda_energy
        self.lambda_stability = lambda_stability
        self.lambda_no_progress = lambda_no_progress
        self.lambda_utility = lambda_utility
        self.lambda_progress = lambda_progress
        self.applicability_temperature = max(float(applicability_temperature), 1e-6)
        self.k_nearest = max(0, int(k_nearest))
        self.feature_zero_slices = list(feature_zero_slices or [])
        self.relative_target = bool(relative_target)
        self.scale_motion_to_execution_horizon = bool(scale_motion_to_execution_horizon)
        self.min_final_height_fraction = max(0.0, float(min_final_height_fraction))
        self.last_decisions: List[ArchiveChunkDecision] = []

        self._segments = list(action_set._archive_segments)
        self._descriptors = np.stack(action_set._archive_descriptors).astype(np.float64)
        if self.scale_motion_to_execution_horizon:
            self._motion_scales = np.asarray(
                [
                    min(1.0, self.execution_horizon / max(len(segment.get("action_seq", [])), 1))
                    for segment in self._segments
                ],
                dtype=np.float64,
            )
        else:
            self._motion_scales = np.ones(len(self._segments), dtype=np.float64)
        self._features = self._apply_feature_zero_slices(
            np.stack(action_set._archive_initial_features).astype(np.float64)
        )
        self._utilities = np.asarray(action_set._archive_utilities, dtype=np.float64)
        if action_set._archive_skill_ids and len(action_set._archive_skill_ids) == len(self._segments):
            self._skill_ids = np.asarray(action_set._archive_skill_ids, dtype=np.int64)
        else:
            self._skill_ids = np.full(len(self._segments), -1, dtype=np.int64)

        self._feature_mean = self._features.mean(axis=0)
        # A small floor prevents high-dimensional observation noise from
        # dominating the distance metric when a feature channel barely varies.
        self._feature_scale = np.maximum(self._features.std(axis=0), 0.25)
        util_std = float(np.std(self._utilities))
        self._utility_z = (self._utilities - float(np.mean(self._utilities))) / max(util_std, 1e-6)
        self._idx = {name: i for i, name in enumerate(DESCRIPTOR_NAMES)}

    @property
    def archive_size(self) -> int:
        return len(self._segments)

    def _apply_feature_zero_slices(self, feature: np.ndarray) -> np.ndarray:
        feature = np.asarray(feature, dtype=np.float64).copy()
        for start, end in self.feature_zero_slices:
            if start < feature.shape[-1]:
                feature[..., start : min(end, feature.shape[-1])] = 0.0
        return feature

    def _align_feature(self, feature: np.ndarray) -> np.ndarray:
        feature = np.asarray(feature, dtype=np.float64).reshape(-1)
        target_dim = self._features.shape[1]
        if len(feature) == target_dim:
            return self._apply_feature_zero_slices(feature)
        if len(feature) > target_dim:
            return self._apply_feature_zero_slices(feature[:target_dim])
        out = np.zeros(target_dim, dtype=np.float64)
        out[: len(feature)] = feature
        return self._apply_feature_zero_slices(out)

    def _state_distances(self, feature: np.ndarray) -> np.ndarray:
        feature = self._align_feature(feature)
        diff = (self._features - feature[None]) / self._feature_scale[None]
        return np.linalg.norm(diff, axis=1) / np.sqrt(self._features.shape[1])

    def select_chunk(
        self,
        pose: tuple[float, float, float],
        target_position: np.ndarray,
        state_feature: np.ndarray,
    ) -> ArchiveChunkDecision:
        x, y, yaw = pose
        target_position = np.asarray(target_position, dtype=np.float64)
        current_xy = np.asarray([x, y], dtype=np.float64)
        current_distance = float(np.linalg.norm(target_position - current_xy))

        state_dist = self._state_distances(state_feature)
        applicability = np.exp(-0.5 * (state_dist / self.applicability_temperature) ** 2)

        dx = self._descriptors[:, self._idx["delta_x"]] * self._motion_scales
        dy = self._descriptors[:, self._idx["delta_y"]] * self._motion_scales
        dxy_world = body_to_world_2d(np.stack([dx, dy], axis=1), yaw)
        predicted_xy = current_xy[None] + dxy_world
        predicted_distance = np.linalg.norm(target_position[None] - predicted_xy, axis=1)
        predicted_progress = current_distance - predicted_distance
        no_progress = np.maximum(0.0, -predicted_progress)

        energy = self._descriptors[:, self._idx["energy_proxy"]]
        stability = np.clip(self._descriptors[:, self._idx["stability_score"]], 0.0, 1.0)
        cost = (
            predicted_distance
            + self.lambda_state * state_dist
            + self.lambda_energy * energy
            + self.lambda_stability * (1.0 - stability)
            + self.lambda_no_progress * no_progress
            - self.lambda_utility * self._utility_z
            - self.lambda_progress * np.maximum(0.0, predicted_progress)
        )

        if 0 < self.k_nearest < len(cost):
            keep = np.argpartition(state_dist, self.k_nearest - 1)[: self.k_nearest]
            local = int(keep[np.argmin(cost[keep])])
        else:
            local = int(np.argmin(cost))

        return ArchiveChunkDecision(
            archive_index=local,
            skill_id=int(self._skill_ids[local]),
            cost=float(cost[local]),
            current_distance=current_distance,
            predicted_distance=float(predicted_distance[local]),
            predicted_progress=float(predicted_progress[local]),
            state_distance=float(state_dist[local]),
            applicability=float(applicability[local]),
            utility=float(self._utilities[local]),
            energy=float(energy[local]),
            stability=float(stability[local]),
        )

    def rollout(self, env: IsaacEnvWrapper, target_position: np.ndarray) -> CompositionResult:
        obs = env.reset()
        start_height = float(env.get_robot_state()["base_pos"][0][2])
        target_position = np.asarray(target_position, dtype=np.float64)
        if self.relative_target:
            start_pos = env.get_robot_state()["base_pos"][0]
            target_position = start_pos[:2].astype(np.float64) + target_position
        self.last_decisions = []

        selected_indices: List[int] = []
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
            decision = self.select_chunk((float(pos[0]), float(pos[1]), yaw), target_position, feature)
            segment = self._segments[decision.archive_index]
            action_seq = np.asarray(segment["action_seq"], dtype=np.float32)
            action_prefix = action_seq[: self.execution_horizon]
            decision.executed_steps = int(len(action_prefix))
            self.last_decisions.append(decision)
            selected_indices.append(decision.archive_index)

            aborted = False
            for action in action_prefix:
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
            skill_sequence=selected_indices,
            num_skills_used=len(selected_indices),
            energy_proxy=energy_total / max(energy_steps, 1),
            base_positions=np.stack(positions),
            terminated_early=terminated_early,
            final_height=final_height,
            min_height=min_height,
        )
