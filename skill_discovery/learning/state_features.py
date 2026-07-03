"""State feature extraction for skill applicability learning.

The discriminator should answer a local dynamics question: "does the current
state look like states where this action chunk was collected?"  We therefore
avoid global x/y position and use observation, body height/orientation,
base velocity, and joint state.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from skill_discovery.utils.math_utils import quat_to_roll_pitch, quat_to_yaw


def _flat(x: Any, max_dim: int | None = None) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32).reshape(-1)
    if max_dim is not None:
        arr = arr[:max_dim]
    return arr


def _quat_local_features(quat: Any) -> np.ndarray:
    quat_arr = np.asarray(quat, dtype=np.float64).reshape(4)
    roll, pitch = quat_to_roll_pitch(quat_arr)
    yaw = float(quat_to_yaw(quat_arr))
    return np.asarray([float(roll), float(pitch), np.sin(yaw), np.cos(yaw)], dtype=np.float32)


def segment_initial_state_feature(
    segment: Dict[str, Any],
    include_obs: bool = True,
    max_obs_dim: int | None = None,
) -> np.ndarray:
    """Build a fixed local-dynamics feature from the first step of a segment."""
    parts = []
    if include_obs and "obs_seq" in segment:
        parts.append(_flat(segment["obs_seq"][0], max_obs_dim))
    if "base_pos_seq" in segment:
        # Keep only height; global x/y should not define dynamic similarity.
        parts.append(np.asarray([segment["base_pos_seq"][0][2]], dtype=np.float32))
    if "base_quat_seq" in segment:
        parts.append(_quat_local_features(segment["base_quat_seq"][0]))
    if "base_lin_vel_seq" in segment:
        parts.append(_flat(segment["base_lin_vel_seq"][0]))
    if "base_ang_vel_seq" in segment:
        parts.append(_flat(segment["base_ang_vel_seq"][0]))
    if "joint_pos_seq" in segment:
        parts.append(_flat(segment["joint_pos_seq"][0]))
    if "joint_vel_seq" in segment:
        parts.append(_flat(segment["joint_vel_seq"][0]))
    if not parts:
        raise ValueError("Segment has no state fields from which to build an initial-state feature.")
    return np.concatenate(parts).astype(np.float32)


def env_state_feature(
    obs: np.ndarray,
    robot_state: Dict[str, np.ndarray],
    env_id: int = 0,
    include_obs: bool = True,
    max_obs_dim: int | None = None,
) -> np.ndarray:
    """Build the same local-dynamics feature from a live vectorized env state."""
    parts = []
    if include_obs:
        parts.append(_flat(np.asarray(obs)[env_id], max_obs_dim))
    parts.append(np.asarray([robot_state["base_pos"][env_id][2]], dtype=np.float32))
    parts.append(_quat_local_features(robot_state["base_quat"][env_id]))
    parts.append(_flat(robot_state["base_lin_vel"][env_id]))
    parts.append(_flat(robot_state["base_ang_vel"][env_id]))
    parts.append(_flat(robot_state["joint_pos"][env_id]))
    parts.append(_flat(robot_state["joint_vel"][env_id]))
    return np.concatenate(parts).astype(np.float32)
