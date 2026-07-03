"""Small math helpers shared across the pipeline (NumPy only)."""

from __future__ import annotations

import numpy as np


def quat_to_yaw(quat: np.ndarray) -> np.ndarray:
    """Extract yaw (rotation about world z) from quaternion(s) in (w, x, y, z) order.

    Args:
        quat: array of shape (..., 4).

    Returns:
        yaw angles in radians, shape (...,).
    """
    quat = np.asarray(quat, dtype=np.float64)
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def quat_to_roll_pitch(quat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Extract roll and pitch from quaternion(s) in (w, x, y, z) order."""
    quat = np.asarray(quat, dtype=np.float64)
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sinp = np.clip(2.0 * (w * y - z * x), -1.0, 1.0)
    pitch = np.arcsin(sinp)
    return roll, pitch


def wrap_angle(angle: np.ndarray | float) -> np.ndarray | float:
    """Wrap angle(s) to [-pi, pi)."""
    return (np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi


def rotate_2d(vec: np.ndarray, angle: float) -> np.ndarray:
    """Rotate 2D vector(s) of shape (..., 2) by `angle` radians (counter-clockwise)."""
    c, s = np.cos(angle), np.sin(angle)
    rot = np.array([[c, -s], [s, c]], dtype=np.float64)
    return np.asarray(vec) @ rot.T


def world_to_body_2d(vec_world: np.ndarray, yaw: float) -> np.ndarray:
    """Express a world-frame 2D vector in the body (heading) frame."""
    return rotate_2d(vec_world, -yaw)


def body_to_world_2d(vec_body: np.ndarray, yaw: float) -> np.ndarray:
    """Express a body-frame 2D vector in the world frame."""
    return rotate_2d(vec_body, yaw)


def set_global_seed(seed: int) -> None:
    """Seed NumPy, Python's random, and torch (if available) for reproducibility."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
