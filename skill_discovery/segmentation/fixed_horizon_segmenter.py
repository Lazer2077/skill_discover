"""Fixed-horizon segmentation of collected trajectories.

Segments never cross episode boundaries: each trajectory in the input is one
episode, and windows are cut inside it only. A trailing window shorter than
`segment_horizon` is kept only if it reaches `min_segment_length`.

FUTURE (V2+): replace fixed windows with adaptive segmentation, e.g. an
option termination function or changepoint detection on gait phase.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np

from skill_discovery.utils.logging import get_logger

logger = get_logger("segmenter")

_SEQ_FIELDS = {
    "obs": "obs_seq",
    "actions": "action_seq",
    "rewards": "reward_seq",
    "base_pos": "base_pos_seq",
    "base_quat": "base_quat_seq",
    "base_lin_vel": "base_lin_vel_seq",
    "base_ang_vel": "base_ang_vel_seq",
    "joint_pos": "joint_pos_seq",
    "joint_vel": "joint_vel_seq",
}


@dataclass
class FixedHorizonSegmenter:
    """Slice trajectories into overlapping fixed-length windows."""

    segment_horizon: int = 32
    segment_stride: int = 16
    min_segment_length: int = 16

    def segment_trajectory(self, traj: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Split one episode trajectory into segments."""
        length = len(traj["actions"])
        segments: List[Dict[str, Any]] = []
        for start in range(0, length, self.segment_stride):
            end = min(start + self.segment_horizon, length)
            if end - start < self.min_segment_length:
                break
            seg: Dict[str, Any] = {
                out_key: np.asarray(traj[in_key][start:end])
                for in_key, out_key in _SEQ_FIELDS.items()
                if in_key in traj
            }
            seg["start_state"] = {
                "base_pos": traj["base_pos"][start],
                "base_quat": traj["base_quat"][start],
            }
            seg["end_state"] = {
                "base_pos": traj["base_pos"][end - 1],
                "base_quat": traj["base_quat"][end - 1],
            }
            # True when the segment's final step ends the episode (fall/timeout).
            seg["terminated_early"] = bool(np.asarray(traj["dones"][start:end]).any())
            seg["env_id"] = traj["env_id"]
            seg["episode_id"] = traj["episode_id"]
            seg["start_timestep"] = start
            segments.append(seg)
            if end == length:
                break
        return segments

    def segment_all(self, trajectories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Segment every trajectory; returns a flat list of segments."""
        segments: List[Dict[str, Any]] = []
        for traj in trajectories:
            segments.extend(self.segment_trajectory(traj))
        logger.info(
            "Segmented %d trajectories into %d segments (H=%d, stride=%d).",
            len(trajectories), len(segments), self.segment_horizon, self.segment_stride,
        )
        return segments
