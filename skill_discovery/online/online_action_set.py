"""Online action-set maintenance for V2.

The action set is updated while exploration runs: new segments are scored for
reward, displacement, stability, smoothness, energy, and descriptor novelty.
Nearby candidates refine an existing skill; novel/high-utility candidates add
or replace entries.  This gives V2 a continuously improving set of reusable
open-loop skill/action chunks without changing the V1 SkillLibrary format.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from skill_discovery.descriptors.locomotion_descriptors import DESCRIPTOR_NAMES, interpret_descriptor
from skill_discovery.learning.state_features import segment_initial_state_feature
from skill_discovery.library.skill_library import Skill, SkillLibrary
from skill_discovery.utils.buffers import load_pickle, save_pickle
from skill_discovery.utils.logging import get_logger

logger = get_logger("online_action_set")


@dataclass
class OnlineActionSetConfig:
    max_skills: int = 16
    clustering_method: str = "weighted_kmeans"  # weighted_kmeans | nearest
    novelty_threshold: float = 3.0
    replace_margin: float = 0.05
    num_representatives: int = 3
    max_initial_states_per_skill: int = 128
    max_descriptor_history: int = 50000
    max_archive_size: int = 5000
    min_stability_for_archive: float = 0.2
    min_success_stability: float = 0.35
    reward_weight: float = 0.2
    stability_weight: float = 1.0
    displacement_weight: float = 1.0
    smoothness_weight: float = 0.25
    energy_weight: float = 0.1
    include_obs_in_state_feature: bool = True
    max_obs_dim: Optional[int] = None


@dataclass
class OnlineUpdateStats:
    iteration: int
    candidates: int
    added: int = 0
    merged: int = 0
    replaced: int = 0
    skipped: int = 0
    num_skills: int = 0
    mean_candidate_utility: float = 0.0
    mean_descriptor_novelty: float = 0.0


@dataclass
class OnlineSkill:
    skill_id: int
    center_descriptor: np.ndarray
    representative_segments: List[Dict[str, Any]]
    initial_state_features: List[np.ndarray]
    num_segments: int = 1
    utility: float = 0.0
    mean_reward: float = 0.0
    last_updated_iteration: int = 0
    _utility_sum: float = 0.0
    _reward_sum: float = 0.0

    @property
    def action_sequence(self) -> np.ndarray:
        return self.representative_segments[0]["action_seq"]


class OnlineActionSet:
    """Continuously updated library of action chunks discovered during exploration."""

    def __init__(self, config: OnlineActionSetConfig | None = None):
        self.config = config or OnlineActionSetConfig()
        self.skills: Dict[int, OnlineSkill] = {}
        self.descriptor_names = list(DESCRIPTOR_NAMES)
        self.metadata: Dict[str, Any] = {}
        self._next_skill_id = 0
        self._descriptor_history: List[np.ndarray] = []
        self._archive_segments: List[Dict[str, Any]] = []
        self._archive_descriptors: List[np.ndarray] = []
        self._archive_initial_features: List[np.ndarray] = []
        self._archive_utilities: List[float] = []
        self._archive_rewards: List[float] = []
        self._archive_skill_ids: List[int] = []

    @property
    def skill_ids(self) -> List[int]:
        return sorted(self.skills)

    @property
    def state_feature_dim(self) -> int | None:
        for skill in self.skills.values():
            if skill.initial_state_features:
                return len(skill.initial_state_features[0])
        return None

    def update(
        self,
        segments: List[Dict[str, Any]],
        descriptors: np.ndarray,
        descriptor_dicts: Optional[List[Dict[str, float]]] = None,
        iteration: int = 0,
    ) -> OnlineUpdateStats:
        """Incrementally merge/add/prune candidates from one exploration batch."""
        descriptors = np.asarray(descriptors, dtype=np.float64)
        if len(segments) != len(descriptors):
            raise ValueError(f"segments ({len(segments)}) and descriptors ({len(descriptors)}) mismatch.")

        if self.config.clustering_method == "weighted_kmeans":
            return self._update_archive_and_recluster(segments, descriptors, iteration)
        if self.config.clustering_method != "nearest":
            raise ValueError(
                f"Unknown online clustering_method={self.config.clustering_method!r}. "
                "Use 'weighted_kmeans' or 'nearest'."
            )

        stats = OnlineUpdateStats(iteration=iteration, candidates=len(segments))
        utilities: List[float] = []
        novelties: List[float] = []

        for segment, desc in zip(segments, descriptors):
            if not np.all(np.isfinite(desc)):
                stats.skipped += 1
                continue
            self._record_descriptor(desc)
            utility = self._candidate_utility(segment, desc)
            novelty, nearest_id = self._nearest_descriptor_distance(desc)
            utilities.append(utility)
            novelties.append(novelty)

            segment_copy = dict(segment)
            segment_copy["_online_score"] = utility
            try:
                init_feature = segment_initial_state_feature(
                    segment_copy,
                    include_obs=self.config.include_obs_in_state_feature,
                    max_obs_dim=self.config.max_obs_dim,
                )
            except ValueError:
                stats.skipped += 1
                continue

            should_add = nearest_id is None or novelty >= self.config.novelty_threshold
            if should_add:
                if len(self.skills) < self.config.max_skills:
                    self._add_skill(segment_copy, desc, init_feature, utility, iteration)
                    stats.added += 1
                else:
                    worst_id = min(self.skills, key=lambda sid: self.skills[sid].utility)
                    if utility > self.skills[worst_id].utility + self.config.replace_margin:
                        del self.skills[worst_id]
                        self._add_skill(segment_copy, desc, init_feature, utility, iteration)
                        stats.replaced += 1
                    elif nearest_id is not None:
                        self._merge_skill(nearest_id, segment_copy, desc, init_feature, utility, iteration)
                        stats.merged += 1
                    else:
                        stats.skipped += 1
            else:
                assert nearest_id is not None
                self._merge_skill(nearest_id, segment_copy, desc, init_feature, utility, iteration)
                stats.merged += 1

        stats.num_skills = len(self.skills)
        finite_novelties = [n for n in novelties if np.isfinite(n)]
        stats.mean_candidate_utility = float(np.mean(utilities)) if utilities else 0.0
        stats.mean_descriptor_novelty = float(np.mean(finite_novelties)) if finite_novelties else 0.0
        logger.info(
            "Online action-set update %d: candidates=%d added=%d merged=%d replaced=%d skipped=%d skills=%d",
            iteration,
            stats.candidates,
            stats.added,
            stats.merged,
            stats.replaced,
            stats.skipped,
            stats.num_skills,
        )
        return stats

    def _update_archive_and_recluster(
        self,
        segments: List[Dict[str, Any]],
        descriptors: np.ndarray,
        iteration: int,
    ) -> OnlineUpdateStats:
        """Archive candidate chunks and rebuild skills with weighted k-means.

        This is more stable than the original one-pass nearest-center update:
        every online iteration re-clusters the retained candidate archive using
        descriptor weights that emphasize locomotion outcome over incidental
        energy/action-scale dimensions.
        """
        old_num_skills = len(self.skills)
        stats = OnlineUpdateStats(iteration=iteration, candidates=len(segments))
        utilities: List[float] = []
        novelties: List[float] = []
        idx = {n: i for i, n in enumerate(DESCRIPTOR_NAMES)}

        for segment, desc in zip(segments, descriptors):
            if not np.all(np.isfinite(desc)):
                stats.skipped += 1
                continue
            self._record_descriptor(desc)
            stability = float(desc[idx["stability_score"]])
            utility = self._candidate_utility(segment, desc)
            novelty, _ = self._nearest_descriptor_distance(desc)
            utilities.append(utility)
            if np.isfinite(novelty):
                novelties.append(novelty)

            # Keep high-utility unstable chunks only if they are very novel; otherwise
            # avoid teaching the action set lots of fall/termination primitives.
            if stability < self.config.min_stability_for_archive and novelty < self.config.novelty_threshold:
                stats.skipped += 1
                continue

            segment_copy = dict(segment)
            segment_copy["_online_score"] = utility
            try:
                init_feature = segment_initial_state_feature(
                    segment_copy,
                    include_obs=self.config.include_obs_in_state_feature,
                    max_obs_dim=self.config.max_obs_dim,
                )
            except ValueError:
                stats.skipped += 1
                continue

            self._archive_segments.append(segment_copy)
            self._archive_descriptors.append(np.asarray(desc, dtype=np.float64).copy())
            self._archive_initial_features.append(init_feature.astype(np.float32))
            self._archive_utilities.append(float(utility))
            self._archive_rewards.append(float(np.mean(segment.get("reward_seq", np.asarray([0.0])))))
            stats.merged += 1

        self._trim_archive()
        self._recluster_archive(iteration)
        stats.added = max(0, len(self.skills) - old_num_skills)
        stats.replaced = max(0, old_num_skills + stats.merged - len(self.skills))
        stats.num_skills = len(self.skills)
        stats.mean_candidate_utility = float(np.mean(utilities)) if utilities else 0.0
        stats.mean_descriptor_novelty = float(np.mean(novelties)) if novelties else 0.0
        logger.info(
            "Online weighted-kmeans update %d: candidates=%d accepted=%d skipped=%d skills=%d archive=%d",
            iteration,
            stats.candidates,
            stats.merged,
            stats.skipped,
            stats.num_skills,
            len(self._archive_segments),
        )
        return stats

    def _trim_archive(self) -> None:
        max_size = self.config.max_archive_size
        if len(self._archive_segments) <= max_size:
            return
        # Retain the most useful chunks; this makes the archive a quality-diversity
        # memory rather than a raw replay buffer.
        old_label_count = len(self._archive_skill_ids)
        old_archive_count = len(self._archive_segments)
        keep = np.argsort(np.asarray(self._archive_utilities))[-max_size:]
        keep = np.sort(keep)
        self._archive_segments = [self._archive_segments[i] for i in keep]
        self._archive_descriptors = [self._archive_descriptors[i] for i in keep]
        self._archive_initial_features = [self._archive_initial_features[i] for i in keep]
        self._archive_utilities = [self._archive_utilities[i] for i in keep]
        self._archive_rewards = [self._archive_rewards[i] for i in keep]
        if old_label_count == old_archive_count:
            self._archive_skill_ids = [self._archive_skill_ids[i] for i in keep]
        else:
            self._archive_skill_ids = []

    def _recluster_archive(self, iteration: int) -> None:
        if not self._archive_descriptors:
            self.skills = {}
            return
        descriptors = np.stack(self._archive_descriptors)
        n_clusters = min(self.config.max_skills, len(descriptors))
        if n_clusters <= 1:
            labels = np.zeros(len(descriptors), dtype=np.int64)
        else:
            X = self._weighted_descriptor_matrix(descriptors)
            labels = self._weighted_kmeans_labels(X, n_clusters=n_clusters)

        new_skills: Dict[int, OnlineSkill] = {}
        utilities = np.asarray(self._archive_utilities, dtype=np.float64)
        rewards = np.asarray(self._archive_rewards, dtype=np.float64)

        # Order clusters by utility so ids are deterministic for a given archive.
        cluster_ids = np.unique(labels)
        cluster_ids = sorted(cluster_ids, key=lambda cid: float(np.mean(utilities[labels == cid])), reverse=True)
        old_to_new = {int(cluster_id): int(new_id) for new_id, cluster_id in enumerate(cluster_ids)}
        self._archive_skill_ids = [old_to_new[int(label)] for label in labels]
        for new_id, cluster_id in enumerate(cluster_ids):
            member_idx = np.where(labels == cluster_id)[0]
            cluster_desc = descriptors[member_idx]
            center = cluster_desc.mean(axis=0)
            scale = self._descriptor_scale()
            dists = np.linalg.norm(
                (cluster_desc - center[None]) / scale[None] * self._descriptor_weights()[None],
                axis=-1,
            )
            member_utils = utilities[member_idx]
            # Representatives should be both central and useful.
            rep_score = member_utils - 0.1 * dists
            rep_order = member_idx[np.argsort(rep_score)[::-1]]
            state_order = member_idx[np.argsort(member_utils)[::-1]]
            reps = [self._archive_segments[i] for i in rep_order[: self.config.num_representatives]]
            init_features = [
                self._archive_initial_features[i]
                for i in state_order[: self.config.max_initial_states_per_skill]
            ]
            utility_sum = float(np.sum(member_utils))
            reward_sum = float(np.sum(rewards[member_idx]))
            num_segments = int(len(member_idx))
            new_skills[new_id] = OnlineSkill(
                skill_id=new_id,
                center_descriptor=center,
                representative_segments=reps,
                initial_state_features=init_features,
                num_segments=num_segments,
                utility=utility_sum / max(num_segments, 1),
                mean_reward=reward_sum / max(num_segments, 1),
                last_updated_iteration=iteration,
                _utility_sum=utility_sum,
                _reward_sum=reward_sum,
            )
        self.skills = new_skills
        self._next_skill_id = len(new_skills)

    @staticmethod
    def _weighted_kmeans_labels(X: np.ndarray, n_clusters: int, max_iters: int = 50) -> np.ndarray:
        """Small NumPy k-means with farthest-point initialization.

        Avoids importing sklearn after Isaac Sim starts; importing sklearn inside
        a running Kit/SimulationApp process can terminate the process on some
        Isaac Sim 5.1 setups.
        """
        X = np.asarray(X, dtype=np.float64)
        n = len(X)
        if n_clusters <= 1 or n <= 1:
            return np.zeros(n, dtype=np.int64)

        centers = [0]
        min_d = np.linalg.norm(X - X[0], axis=1)
        for _ in range(1, min(n_clusters, n)):
            idx = int(np.argmax(min_d))
            centers.append(idx)
            min_d = np.minimum(min_d, np.linalg.norm(X - X[idx], axis=1))
        center_arr = X[centers].copy()

        labels = np.zeros(n, dtype=np.int64)
        for _ in range(max_iters):
            dists = np.linalg.norm(X[:, None, :] - center_arr[None, :, :], axis=-1)
            new_labels = np.argmin(dists, axis=1)
            if np.array_equal(labels, new_labels):
                break
            labels = new_labels
            for k in range(len(center_arr)):
                mask = labels == k
                if np.any(mask):
                    center_arr[k] = X[mask].mean(axis=0)
                else:
                    # Re-seed empty clusters with the currently worst represented point.
                    nearest = np.min(dists, axis=1)
                    center_arr[k] = X[int(np.argmax(nearest))]
        return labels

    def outcome_dataset(self) -> Dict[str, np.ndarray]:
        """Return archive data for training skill-level dynamics/success models."""
        if not self._archive_descriptors or not self._archive_skill_ids:
            raise ValueError("No labeled archive data available. Run update() first.")
        idx = {n: i for i, n in enumerate(DESCRIPTOR_NAMES)}
        descriptors = np.stack(self._archive_descriptors)
        state_features = np.stack(self._archive_initial_features).astype(np.float32)
        skill_ids = np.asarray(self._archive_skill_ids, dtype=np.int64)
        outcomes = descriptors[
            :,
            [
                idx["delta_x"],
                idx["delta_y"],
                idx["delta_yaw"],
                idx["energy_proxy"],
            ],
        ].astype(np.float32)
        success = []
        for segment, desc in zip(self._archive_segments, descriptors):
            stable = float(desc[idx["stability_score"]]) >= self.config.min_success_stability
            success.append(float(stable and not bool(segment.get("terminated_early", False))))
        return {
            "state_features": state_features,
            "skill_ids": skill_ids,
            "outcomes": outcomes,
            "success": np.asarray(success, dtype=np.float32),
            "num_skills": np.asarray([len(self.skills)], dtype=np.int64),
        }

    def _record_descriptor(self, desc: np.ndarray) -> None:
        self._descriptor_history.append(np.asarray(desc, dtype=np.float64))
        if len(self._descriptor_history) > self.config.max_descriptor_history:
            self._descriptor_history = self._descriptor_history[-self.config.max_descriptor_history :]

    def _descriptor_scale(self) -> np.ndarray:
        if len(self._descriptor_history) < 2:
            return np.ones(len(DESCRIPTOR_NAMES), dtype=np.float64)
        return np.std(np.stack(self._descriptor_history), axis=0) + 1e-6

    def _descriptor_weights(self) -> np.ndarray:
        """Weights for descriptor-space clustering.

        Outcome dimensions carry the behavior identity; energy/action norms are
        useful but should not split otherwise identical locomotion outcomes into
        many singleton skills.
        """
        weights = np.ones(len(DESCRIPTOR_NAMES), dtype=np.float64)
        by_name = {
            "delta_x": 2.5,
            "delta_y": 2.5,
            "delta_yaw": 1.8,
            "average_forward_velocity": 1.5,
            "average_lateral_velocity": 1.5,
            "average_yaw_rate": 1.2,
            "mean_body_height": 0.4,
            "body_height_std": 0.8,
            "mean_action_norm": 0.35,
            "mean_joint_velocity_norm": 0.35,
            "energy_proxy": 0.5,
            "stability_score": 1.0,
            "smoothness_score": 0.8,
        }
        for i, name in enumerate(DESCRIPTOR_NAMES):
            weights[i] = by_name.get(name, 1.0)
        return weights

    def _weighted_descriptor_matrix(self, descriptors: np.ndarray) -> np.ndarray:
        scale = self._descriptor_scale()
        return ((descriptors - descriptors.mean(axis=0, keepdims=True)) / scale[None]) * self._descriptor_weights()[None]

    def _nearest_descriptor_distance(self, desc: np.ndarray) -> tuple[float, int | None]:
        if not self.skills:
            return float("inf"), None
        scale = self._descriptor_scale()
        ids = self.skill_ids
        centers = np.stack([self.skills[sid].center_descriptor for sid in ids])
        dists = np.linalg.norm((centers - desc[None]) / scale[None], axis=-1)
        idx = int(np.argmin(dists))
        return float(dists[idx]), ids[idx]

    def _candidate_utility(self, segment: Dict[str, Any], desc: np.ndarray) -> float:
        idx = {n: i for i, n in enumerate(DESCRIPTOR_NAMES)}
        reward = float(np.mean(segment.get("reward_seq", np.asarray([0.0]))))
        displacement = float(np.linalg.norm(desc[[idx["delta_x"], idx["delta_y"]]]))
        stability = float(desc[idx["stability_score"]])
        smoothness = float(desc[idx["smoothness_score"]])
        energy = float(desc[idx["energy_proxy"]])
        return (
            self.config.reward_weight * reward
            + self.config.stability_weight * stability
            + self.config.displacement_weight * displacement
            + self.config.smoothness_weight * smoothness
            - self.config.energy_weight * energy
        )

    def _add_skill(
        self,
        segment: Dict[str, Any],
        desc: np.ndarray,
        init_feature: np.ndarray,
        utility: float,
        iteration: int,
    ) -> int:
        skill_id = self._next_skill_id
        self._next_skill_id += 1
        reward = float(np.mean(segment.get("reward_seq", np.asarray([0.0]))))
        self.skills[skill_id] = OnlineSkill(
            skill_id=skill_id,
            center_descriptor=np.asarray(desc, dtype=np.float64).copy(),
            representative_segments=[segment],
            initial_state_features=[init_feature.astype(np.float32)],
            num_segments=1,
            utility=float(utility),
            mean_reward=reward,
            last_updated_iteration=iteration,
            _utility_sum=float(utility),
            _reward_sum=reward,
        )
        return skill_id

    def _merge_skill(
        self,
        skill_id: int,
        segment: Dict[str, Any],
        desc: np.ndarray,
        init_feature: np.ndarray,
        utility: float,
        iteration: int,
    ) -> None:
        skill = self.skills[skill_id]
        n = skill.num_segments
        skill.center_descriptor = (skill.center_descriptor * n + desc) / (n + 1)
        skill.num_segments += 1
        skill._utility_sum += float(utility)
        skill.utility = skill._utility_sum / skill.num_segments
        reward = float(np.mean(segment.get("reward_seq", np.asarray([0.0]))))
        skill._reward_sum += reward
        skill.mean_reward = skill._reward_sum / skill.num_segments
        skill.last_updated_iteration = iteration

        skill.representative_segments.append(segment)
        skill.representative_segments.sort(key=lambda s: float(s.get("_online_score", 0.0)), reverse=True)
        del skill.representative_segments[self.config.num_representatives :]

        skill.initial_state_features.append(init_feature.astype(np.float32))
        if len(skill.initial_state_features) > self.config.max_initial_states_per_skill:
            skill.initial_state_features = skill.initial_state_features[-self.config.max_initial_states_per_skill :]

    def to_skill_library(self) -> SkillLibrary:
        """Convert the online action set into the V1 SkillLibrary interface."""
        idx = {n: i for i, n in enumerate(DESCRIPTOR_NAMES)}
        if self._descriptor_history:
            descs = np.stack(self._descriptor_history)
            disp_scale = float(np.mean(np.abs(descs[:, [idx["delta_x"], idx["delta_y"]]])))
            energy_scale = float(np.mean(descs[:, idx["energy_proxy"]]))
        else:
            disp_scale = 1.0
            energy_scale = 1.0

        lib = SkillLibrary()
        for sid in self.skill_ids:
            online_skill = self.skills[sid]
            center = online_skill.center_descriptor
            center_dict = {n: float(center[idx[n]]) for n in DESCRIPTOR_NAMES}
            lib.skills[sid] = Skill(
                skill_id=sid,
                center_descriptor=center.copy(),
                representative_segments=list(online_skill.representative_segments),
                mean_delta_x=center_dict["delta_x"],
                mean_delta_y=center_dict["delta_y"],
                mean_delta_yaw=center_dict["delta_yaw"],
                mean_energy=center_dict["energy_proxy"],
                mean_stability=center_dict["stability_score"],
                num_segments=online_skill.num_segments,
                interpretation=interpret_descriptor(center_dict, disp_scale, energy_scale),
            )
        lib.metadata = dict(self.metadata)
        lib.metadata.update({"online_v2": True, "num_online_skills": len(self.skills)})
        return lib

    def summary(self) -> List[Dict[str, Any]]:
        lib = self.to_skill_library()
        rows = []
        for row in lib.summary():
            skill = self.skills[row["skill_id"]]
            row.update(
                {
                    "utility": round(float(skill.utility), 4),
                    "mean_reward": round(float(skill.mean_reward), 4),
                    "initial_state_examples": len(skill.initial_state_features),
                    "last_updated_iteration": skill.last_updated_iteration,
                }
            )
            rows.append(row)
        return rows

    def save(self, path: str) -> None:
        save_pickle(
            {
                "config": self.config,
                "skills": self.skills,
                "descriptor_names": self.descriptor_names,
                "metadata": self.metadata,
                "next_skill_id": self._next_skill_id,
                "descriptor_history": self._descriptor_history,
                "archive_segments": self._archive_segments,
                "archive_descriptors": self._archive_descriptors,
                "archive_initial_features": self._archive_initial_features,
                "archive_utilities": self._archive_utilities,
                "archive_rewards": self._archive_rewards,
                "archive_skill_ids": self._archive_skill_ids,
            },
            path,
        )
        logger.info("Saved online action set to %s", path)

    @classmethod
    def load(cls, path: str) -> "OnlineActionSet":
        data = load_pickle(path)
        obj = cls(config=data["config"])
        obj.skills = data["skills"]
        obj.descriptor_names = data["descriptor_names"]
        obj.metadata = data.get("metadata", {})
        obj._next_skill_id = data.get("next_skill_id", max(obj.skills, default=-1) + 1)
        obj._descriptor_history = data.get("descriptor_history", [])
        obj._archive_segments = data.get("archive_segments", [])
        obj._archive_descriptors = data.get("archive_descriptors", [])
        obj._archive_initial_features = data.get("archive_initial_features", [])
        obj._archive_utilities = data.get("archive_utilities", [])
        obj._archive_rewards = data.get("archive_rewards", [])
        obj._archive_skill_ids = data.get("archive_skill_ids", [])
        return obj
