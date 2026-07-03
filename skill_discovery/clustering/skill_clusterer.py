"""Clustering of behavior descriptors into discrete skills.

Supports KMeans (default), GaussianMixture, and HDBSCAN (optional; falls back
to KMeans with a warning if the `hdbscan` package is not installed).
Descriptors are z-score normalized internally before clustering.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from skill_discovery.utils.buffers import load_pickle, save_pickle
from skill_discovery.utils.logging import get_logger

logger = get_logger("skill_clusterer")


class SkillClusterer:
    """Fit/predict skill ids over descriptor matrices, with save/load."""

    SUPPORTED_METHODS = ("kmeans", "gmm", "hdbscan")

    def __init__(self, method: str = "kmeans", num_skills: int = 8, random_seed: int = 42):
        if method not in self.SUPPORTED_METHODS:
            raise ValueError(f"Unknown clustering method '{method}'. Use one of {self.SUPPORTED_METHODS}.")
        self.method = method
        self.num_skills = num_skills
        self.random_seed = random_seed
        self.scaler: Optional[StandardScaler] = None
        self.model = None
        self.labels_: Optional[np.ndarray] = None
        self.cluster_centers_: Optional[np.ndarray] = None  # in ORIGINAL descriptor units

    def fit(self, descriptor_matrix: np.ndarray) -> np.ndarray:
        """Fit the clusterer; returns skill ids for the training descriptors."""
        if len(descriptor_matrix) < self.num_skills:
            raise ValueError(
                f"Need at least num_skills={self.num_skills} segments, got {len(descriptor_matrix)}. "
                "Collect more exploration data or lower --num_skills."
            )
        self.scaler = StandardScaler().fit(descriptor_matrix)
        X = self.scaler.transform(descriptor_matrix)

        method = self.method
        if method == "hdbscan":
            try:
                import hdbscan  # type: ignore
            except ImportError:
                logger.warning("hdbscan not installed; falling back to kmeans.")
                method = "kmeans"

        if method == "kmeans":
            self.model = KMeans(n_clusters=self.num_skills, random_state=self.random_seed, n_init=10)
            labels = self.model.fit_predict(X)
        elif method == "gmm":
            self.model = GaussianMixture(
                n_components=self.num_skills, random_state=self.random_seed, n_init=3
            )
            labels = self.model.fit_predict(X)
        else:  # hdbscan; may emit label -1 for noise points
            self.model = hdbscan.HDBSCAN(min_cluster_size=max(10, len(X) // (self.num_skills * 4)))
            labels = self.model.fit_predict(X)

        self.labels_ = labels
        self.cluster_centers_ = self._compute_centers(descriptor_matrix, labels)
        counts = np.bincount(labels[labels >= 0])
        logger.info(
            "Clustered %d segments into %d skills (method=%s). Sizes: %s",
            len(X), len(self.cluster_centers_), method, counts.tolist(),
        )
        return labels

    def predict(self, descriptor_matrix: np.ndarray) -> np.ndarray:
        """Assign skill ids to new descriptors."""
        if self.model is None or self.scaler is None:
            raise RuntimeError("SkillClusterer is not fitted. Call fit() or load() first.")
        X = self.scaler.transform(descriptor_matrix)
        if hasattr(self.model, "predict"):
            return self.model.predict(X)
        # HDBSCAN has no predict(); use nearest fitted cluster center instead.
        centers_scaled = self.scaler.transform(self.cluster_centers_)
        dists = np.linalg.norm(X[:, None, :] - centers_scaled[None, :, :], axis=-1)
        return np.argmin(dists, axis=1)

    @staticmethod
    def _compute_centers(descriptor_matrix: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Per-cluster mean descriptor in original (unnormalized) units."""
        ids = np.unique(labels[labels >= 0])
        return np.stack([descriptor_matrix[labels == sid].mean(axis=0) for sid in ids])

    def save(self, path: str) -> None:
        save_pickle(
            {
                "method": self.method,
                "num_skills": self.num_skills,
                "random_seed": self.random_seed,
                "scaler": self.scaler,
                "model": self.model,
                "labels": self.labels_,
                "cluster_centers": self.cluster_centers_,
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> "SkillClusterer":
        data = load_pickle(path)
        obj = cls(method=data["method"], num_skills=data["num_skills"], random_seed=data["random_seed"])
        obj.scaler = data["scaler"]
        obj.model = data["model"]
        obj.labels_ = data["labels"]
        obj.cluster_centers_ = data["cluster_centers"]
        return obj
