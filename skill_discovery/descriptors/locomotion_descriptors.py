"""Behavior descriptors for locomotion segments.

Each segment is summarized by a low-dimensional descriptor vector in the
spirit of Quality-Diversity behavior characterizations (Cully et al., 2015).
Displacements and velocities are expressed in the *body frame at the segment
start* so that descriptors are invariant to where/which way the robot was
facing — "walk forward" clusters together regardless of world heading.

Energy note: joint torques are not recorded in V1, so energy is approximated
as mean(|action * joint_velocity|). For torque-controlled envs this is
proportional to mechanical power; for position-controlled envs it is a proxy.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from skill_discovery.utils.math_utils import (
    quat_to_roll_pitch,
    quat_to_yaw,
    world_to_body_2d,
    wrap_angle,
)

DESCRIPTOR_NAMES: Tuple[str, ...] = (
    "delta_x",
    "delta_y",
    "delta_yaw",
    "average_forward_velocity",
    "average_lateral_velocity",
    "average_yaw_rate",
    "mean_body_height",
    "body_height_std",
    "mean_action_norm",
    "mean_joint_velocity_norm",
    "energy_proxy",
    "stability_score",
    "smoothness_score",
)


class LocomotionDescriptorExtractor:
    """Compute named behavior descriptors from a trajectory segment."""

    names: Tuple[str, ...] = DESCRIPTOR_NAMES

    def compute(self, segment: Dict[str, Any]) -> Dict[str, float]:
        """Return the descriptor as a dict of named floats."""
        pos = np.asarray(segment["base_pos_seq"], dtype=np.float64)      # (T, 3)
        quat = np.asarray(segment["base_quat_seq"], dtype=np.float64)    # (T, 4) wxyz
        lin_vel = np.asarray(segment["base_lin_vel_seq"], dtype=np.float64)
        ang_vel = np.asarray(segment["base_ang_vel_seq"], dtype=np.float64)
        actions = np.asarray(segment["action_seq"], dtype=np.float64)
        joint_vel = np.asarray(segment["joint_vel_seq"], dtype=np.float64)

        yaw = quat_to_yaw(quat)
        yaw0 = float(yaw[0])
        roll, pitch = quat_to_roll_pitch(quat)

        # Displacement in the start-of-segment body frame.
        delta_world = pos[-1, :2] - pos[0, :2]
        delta_body = world_to_body_2d(delta_world, yaw0)
        delta_yaw = float(wrap_angle(yaw[-1] - yaw0))

        # Velocities rotated into each step's instantaneous body frame.
        vel_body = np.stack(
            [world_to_body_2d(lin_vel[t, :2], yaw[t]) for t in range(len(yaw))]
        )

        # Stability: upright, steady-height, and did not terminate the episode.
        tilt_var = float(np.var(roll) + np.var(pitch))
        height_std = float(np.std(pos[:, 2]))
        terminated = bool(segment.get("terminated_early", False))
        stability = float(
            np.exp(-5.0 * tilt_var) * np.exp(-10.0 * height_std) * (0.0 if terminated else 1.0)
        )

        # Smoothness: small step-to-step action changes -> score near 1.
        if len(actions) > 1:
            action_diff = float(np.mean(np.linalg.norm(np.diff(actions, axis=0), axis=-1)))
        else:
            action_diff = 0.0
        smoothness = float(np.exp(-action_diff))

        # Energy proxy; see module docstring for the torque caveat.
        min_dim = min(actions.shape[-1], joint_vel.shape[-1])
        energy = float(np.mean(np.abs(actions[:, :min_dim] * joint_vel[:, :min_dim])))

        return {
            "delta_x": float(delta_body[0]),
            "delta_y": float(delta_body[1]),
            "delta_yaw": delta_yaw,
            "average_forward_velocity": float(np.mean(vel_body[:, 0])),
            "average_lateral_velocity": float(np.mean(vel_body[:, 1])),
            "average_yaw_rate": float(np.mean(ang_vel[:, 2])),
            "mean_body_height": float(np.mean(pos[:, 2])),
            "body_height_std": height_std,
            "mean_action_norm": float(np.mean(np.linalg.norm(actions, axis=-1))),
            "mean_joint_velocity_norm": float(np.mean(np.linalg.norm(joint_vel, axis=-1))),
            "energy_proxy": energy,
            "stability_score": stability,
            "smoothness_score": smoothness,
        }

    def to_vector(self, descriptor_dict: Dict[str, float]) -> np.ndarray:
        """Convert a descriptor dict to a fixed-order numpy vector."""
        return np.array([descriptor_dict[name] for name in self.names], dtype=np.float64)

    def compute_matrix(self, segments: List[Dict[str, Any]]) -> Tuple[np.ndarray, List[Dict[str, float]]]:
        """Descriptors for many segments: (matrix of shape (N, D), list of dicts)."""
        dicts = [self.compute(seg) for seg in segments]
        matrix = np.stack([self.to_vector(d) for d in dicts]) if dicts else np.zeros((0, len(self.names)))
        return matrix, dicts


def interpret_descriptor(
    desc: Dict[str, float],
    displacement_scale: float = 1.0,
    energy_scale: float = 1.0,
) -> str:
    """Heuristic human-readable label for a (mean) descriptor.

    `displacement_scale` / `energy_scale` should be typical per-segment
    magnitudes for the dataset (e.g. dataset means), so thresholds adapt to
    robot size, control mode, and segment horizon.
    """
    dx, dy, dyaw = desc["delta_x"], desc["delta_y"], desc["delta_yaw"]
    scale = max(displacement_scale, 1e-6)
    tags: List[str] = []

    if abs(dyaw) > 0.3:
        tags.append("turn-left" if dyaw > 0 else "turn-right")
    if abs(dx) > 0.4 * scale:
        tags.append("forward" if dx > 0 else "backward")
    if abs(dy) > 0.4 * scale:
        tags.append("lateral-left" if dy > 0 else "lateral-right")
    if not tags:
        tags.append("stable-low-motion" if desc["stability_score"] > 0.5 else "in-place-unstable")

    if desc["stability_score"] < 0.3:
        tags.append("unstable")
    tags.append("low-energy" if desc["energy_proxy"] < max(energy_scale, 1e-6) else "high-energy")
    return "-".join(tags)
