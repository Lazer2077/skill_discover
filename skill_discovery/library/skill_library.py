"""Skill library: the persistent artifact of the discovery pipeline.

In V1 a skill is an *open-loop motion primitive*: its representative
segment's action sequence (the segment closest to the cluster center in
normalized descriptor space), plus the cluster's average outcome statistics
(body-frame delta_x/delta_y/delta_yaw, energy, stability).

FUTURE (V2+): replace representative action sequences with a learned
closed-loop low-level skill policy pi(a | s, z=skill_id), and attach a
skill-level dynamics model p(s' | s, skill) for planning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from skill_discovery.descriptors.locomotion_descriptors import DESCRIPTOR_NAMES
from skill_discovery.utils.buffers import load_pickle, save_pickle
from skill_discovery.utils.logging import get_logger
from skill_discovery.utils.math_utils import body_to_world_2d, wrap_angle

logger = get_logger("skill_library")


@dataclass
class Skill:
    """One discovered skill (meta-behavior)."""

    skill_id: int
    center_descriptor: np.ndarray                 # mean descriptor, original units
    representative_segments: List[Dict[str, Any]]  # up to k segments nearest the center
    mean_delta_x: float                            # body-frame, per segment execution
    mean_delta_y: float
    mean_delta_yaw: float
    mean_energy: float
    mean_stability: float
    num_segments: int
    interpretation: str = ""

    @property
    def action_sequence(self) -> np.ndarray:
        """Open-loop action sequence of the most representative segment, (H, action_dim)."""
        return self.representative_segments[0]["action_seq"]

    def predict_outcome(self, x: float, y: float, yaw: float) -> tuple:
        """Predict world-frame (x, y, yaw) after executing this skill from a pose."""
        dxy_world = body_to_world_2d(np.array([self.mean_delta_x, self.mean_delta_y]), yaw)
        return (
            x + float(dxy_world[0]),
            y + float(dxy_world[1]),
            float(wrap_angle(yaw + self.mean_delta_yaw)),
        )


class SkillLibrary:
    """Container for discovered skills with outcome-based lookup and save/load."""

    def __init__(self) -> None:
        self.skills: Dict[int, Skill] = {}
        self.descriptor_names: List[str] = list(DESCRIPTOR_NAMES)
        self.metadata: Dict[str, Any] = {}

    def build(
        self,
        segments: List[Dict[str, Any]],
        descriptors: np.ndarray,
        skill_ids: np.ndarray,
        descriptor_dicts: Optional[List[Dict[str, float]]] = None,
        num_representatives: int = 3,
    ) -> "SkillLibrary":
        """Assemble skills from clustered segments.

        Representative segments are those closest to the cluster center in
        z-normalized descriptor space; noise points (skill_id < 0, HDBSCAN)
        are ignored.
        """
        from skill_discovery.descriptors.locomotion_descriptors import interpret_descriptor

        name_idx = {n: i for i, n in enumerate(DESCRIPTOR_NAMES)}
        std = descriptors.std(axis=0) + 1e-8

        # Dataset-level scales for the heuristic interpretation strings.
        disp_scale = float(
            np.mean(np.abs(descriptors[:, [name_idx["delta_x"], name_idx["delta_y"]]]))
        )
        energy_scale = float(np.mean(descriptors[:, name_idx["energy_proxy"]]))

        for sid in np.unique(skill_ids[skill_ids >= 0]):
            mask = skill_ids == sid
            cluster_desc = descriptors[mask]
            center = cluster_desc.mean(axis=0)

            dists = np.linalg.norm((cluster_desc - center) / std, axis=-1)
            order = np.argsort(dists)[:num_representatives]
            member_indices = np.where(mask)[0]
            reps = [segments[member_indices[i]] for i in order]

            center_dict = {n: float(center[name_idx[n]]) for n in DESCRIPTOR_NAMES}
            self.skills[int(sid)] = Skill(
                skill_id=int(sid),
                center_descriptor=center,
                representative_segments=reps,
                mean_delta_x=center_dict["delta_x"],
                mean_delta_y=center_dict["delta_y"],
                mean_delta_yaw=center_dict["delta_yaw"],
                mean_energy=center_dict["energy_proxy"],
                mean_stability=center_dict["stability_score"],
                num_segments=int(mask.sum()),
                interpretation=interpret_descriptor(center_dict, disp_scale, energy_scale),
            )
        logger.info("Built skill library with %d skills.", len(self.skills))
        return self

    def get_skill(self, skill_id: int) -> Skill:
        if skill_id not in self.skills:
            raise KeyError(f"Skill {skill_id} not in library (available: {sorted(self.skills)}).")
        return self.skills[skill_id]

    @property
    def skill_ids(self) -> List[int]:
        return sorted(self.skills)

    def find_skill_by_desired_outcome(
        self, delta_x: float, delta_y: float, delta_yaw: float = 0.0, yaw_weight: float = 0.5
    ) -> Skill:
        """Return the skill whose mean body-frame outcome best matches the request."""
        target = np.array([delta_x, delta_y])
        best_skill, best_cost = None, np.inf
        for skill in self.skills.values():
            cost = float(
                np.linalg.norm(np.array([skill.mean_delta_x, skill.mean_delta_y]) - target)
                + yaw_weight * abs(wrap_angle(skill.mean_delta_yaw - delta_yaw))
            )
            if cost < best_cost:
                best_skill, best_cost = skill, cost
        assert best_skill is not None, "Skill library is empty."
        return best_skill

    def summary(self) -> List[Dict[str, Any]]:
        """JSON-serializable per-skill summary."""
        return [
            {
                "skill_id": s.skill_id,
                "num_segments": s.num_segments,
                "mean_delta_x": round(s.mean_delta_x, 4),
                "mean_delta_y": round(s.mean_delta_y, 4),
                "mean_delta_yaw": round(s.mean_delta_yaw, 4),
                "mean_energy": round(s.mean_energy, 4),
                "mean_stability": round(s.mean_stability, 4),
                "interpretation": s.interpretation,
            }
            for s in (self.skills[i] for i in self.skill_ids)
        ]

    def save(self, path: str) -> None:
        save_pickle(
            {"skills": self.skills, "descriptor_names": self.descriptor_names, "metadata": self.metadata},
            path,
        )
        logger.info("Saved skill library to %s", path)

    @classmethod
    def load(cls, path: str) -> "SkillLibrary":
        data = load_pickle(path)
        lib = cls()
        lib.skills = data["skills"]
        lib.descriptor_names = data["descriptor_names"]
        lib.metadata = data.get("metadata", {})
        return lib
