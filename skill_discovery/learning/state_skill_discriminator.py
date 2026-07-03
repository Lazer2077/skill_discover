"""Learning-based discriminator for state-skill applicability.

The model is a pair classifier.  Given a current local state feature and one
recorded initial-state feature for a skill/action chunk, it predicts whether
executing that skill from the current state is dynamically plausible.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from skill_discovery.utils.logging import get_logger

logger = get_logger("state_skill_discriminator")


class _PairMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


@dataclass
class DiscriminatorTrainStats:
    num_pairs: int
    positive_pairs: int
    negative_pairs: int
    final_loss: float
    final_accuracy: float


class StateSkillDiscriminator:
    """Binary classifier over (state, skill-initial-state) pairs."""

    def __init__(
        self,
        state_dim: int | None = None,
        hidden_dim: int = 128,
        threshold: float = 0.55,
        hybrid_alpha: float = 0.5,
        rbf_temperature: float = 1.5,
        device: str = "cpu",
        seed: int = 42,
    ):
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.threshold = threshold
        self.hybrid_alpha = hybrid_alpha
        self.rbf_temperature = rbf_temperature
        self.device = torch.device(device)
        self.seed = seed
        self.model: _PairMLP | None = None
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None
        self.state_mean: np.ndarray | None = None
        self.state_std: np.ndarray | None = None
        self.is_fitted = False

    @property
    def pair_dim(self) -> int:
        if self.state_dim is None:
            raise RuntimeError("State dimension is unknown. Fit or load the discriminator first.")
        return 4 * self.state_dim

    def _ensure_model(self, state_dim: int) -> None:
        if self.model is None:
            self.state_dim = int(state_dim)
            self.model = _PairMLP(self.pair_dim, self.hidden_dim).to(self.device)

    @staticmethod
    def make_pair_feature(state_feature: np.ndarray, reference_feature: np.ndarray) -> np.ndarray:
        state = np.asarray(state_feature, dtype=np.float32).reshape(-1)
        ref = np.asarray(reference_feature, dtype=np.float32).reshape(-1)
        if state.shape != ref.shape:
            raise ValueError(f"State feature shape {state.shape} != reference shape {ref.shape}.")
        diff = state - ref
        return np.concatenate([state, ref, diff, np.abs(diff)]).astype(np.float32)

    def _normalise(self, pair_features: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("Discriminator normalizer is not fitted.")
        return ((pair_features - self.mean) / self.std).astype(np.float32)

    def build_training_pairs(
        self,
        action_set: Any,
        negative_ratio: int = 2,
        max_pairs: int = 20000,
        seed: int | None = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create positive/negative pair data from an OnlineActionSet-like object."""
        rng = np.random.default_rng(self.seed if seed is None else seed)
        entries = [s for s in action_set.skills.values() if s.initial_state_features]
        if not entries:
            raise ValueError("No skills with initial-state features are available.")

        pair_features: List[np.ndarray] = []
        labels: List[float] = []
        state_dim = len(entries[0].initial_state_features[0])

        for entry in entries:
            own_refs = entry.initial_state_features
            for state in own_refs:
                if len(state) != state_dim:
                    continue
                ref = own_refs[int(rng.integers(len(own_refs)))]
                pair_features.append(self.make_pair_feature(state, ref))
                labels.append(1.0)

                for _ in range(max(1, negative_ratio)):
                    other_entries = [e for e in entries if e.skill_id != entry.skill_id and e.initial_state_features]
                    if other_entries:
                        other = other_entries[int(rng.integers(len(other_entries)))]
                        neg_ref = other.initial_state_features[int(rng.integers(len(other.initial_state_features)))]
                    else:
                        # Single-skill bootstrap: synthesize a dynamically distant negative.
                        scale = np.std(np.stack(own_refs), axis=0) + 0.5
                        neg_ref = state + rng.normal(0.0, scale).astype(np.float32)
                    pair_features.append(self.make_pair_feature(state, neg_ref))
                    labels.append(0.0)

        if len(pair_features) > max_pairs:
            idx = rng.choice(len(pair_features), size=max_pairs, replace=False)
            pair_features = [pair_features[i] for i in idx]
            labels = [labels[i] for i in idx]

        return np.stack(pair_features).astype(np.float32), np.asarray(labels, dtype=np.float32)

    def fit(
        self,
        action_set: Any,
        epochs: int = 5,
        batch_size: int = 256,
        lr: float = 1e-3,
        negative_ratio: int = 2,
        max_pairs: int = 20000,
    ) -> DiscriminatorTrainStats:
        """Train the discriminator from the current online action set."""
        torch.manual_seed(self.seed)
        X, y = self.build_training_pairs(action_set, negative_ratio=negative_ratio, max_pairs=max_pairs)
        self._ensure_model(X.shape[1] // 4)
        assert self.model is not None

        state_refs = [
            np.asarray(ref, dtype=np.float32)
            for skill in action_set.skills.values()
            for ref in skill.initial_state_features
        ]
        if state_refs:
            states = np.stack(state_refs)
            self.state_mean = states.mean(axis=0, keepdims=True)
            self.state_std = states.std(axis=0, keepdims=True) + 0.25
        else:
            self.state_mean = None
            self.state_std = None

        self.mean = X.mean(axis=0, keepdims=True)
        self.std = X.std(axis=0, keepdims=True) + 1e-6
        Xn = self._normalise(X)

        ds = TensorDataset(torch.from_numpy(Xn), torch.from_numpy(y))
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)
        opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        loss_fn = nn.BCEWithLogitsLoss()

        final_loss = 0.0
        for _ in range(max(1, epochs)):
            for xb, yb in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                logits = self.model(xb)
                loss = loss_fn(logits, yb)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                final_loss = float(loss.detach().cpu())

        with torch.no_grad():
            logits = self.model(torch.from_numpy(Xn).to(self.device))
            probs = torch.sigmoid(logits).cpu().numpy()
        acc = float(np.mean((probs >= 0.5) == (y >= 0.5)))
        self.is_fitted = True
        stats = DiscriminatorTrainStats(
            num_pairs=int(len(y)),
            positive_pairs=int(np.sum(y >= 0.5)),
            negative_pairs=int(np.sum(y < 0.5)),
            final_loss=final_loss,
            final_accuracy=acc,
        )
        logger.info(
            "Trained state-skill discriminator: pairs=%d pos=%d neg=%d loss=%.4f acc=%.3f",
            stats.num_pairs,
            stats.positive_pairs,
            stats.negative_pairs,
            stats.final_loss,
            stats.final_accuracy,
        )
        return stats

    def neural_score(self, state_feature: np.ndarray, reference_features: Iterable[np.ndarray]) -> float:
        """Return max learned applicability probability over reference states."""
        refs = list(reference_features)
        if not refs:
            return 0.0
        if not self.is_fitted or self.model is None:
            return 1.0
        pair_features = np.stack([self.make_pair_feature(state_feature, ref) for ref in refs])
        Xn = self._normalise(pair_features)
        with torch.no_grad():
            logits = self.model(torch.from_numpy(Xn).to(self.device))
            probs = torch.sigmoid(logits).cpu().numpy()
        return float(np.max(probs))

    def rbf_score(self, state_feature: np.ndarray, reference_features: Iterable[np.ndarray]) -> float:
        """Distance-based initial-state similarity used as a robust small-data prior."""
        refs = list(reference_features)
        if not refs:
            return 0.0
        state = np.asarray(state_feature, dtype=np.float32).reshape(1, -1)
        ref_arr = np.stack([np.asarray(r, dtype=np.float32).reshape(-1) for r in refs])
        if self.state_std is not None:
            scale = self.state_std.reshape(1, -1)
        else:
            scale = ref_arr.std(axis=0, keepdims=True) + 0.25
        scale = np.maximum(scale, 0.25)
        dists = np.linalg.norm((ref_arr - state) / scale, axis=-1) / np.sqrt(ref_arr.shape[-1])
        sims = np.exp(-0.5 * (dists / max(self.rbf_temperature, 1e-6)) ** 2)
        return float(np.max(sims))

    def score(self, state_feature: np.ndarray, reference_features: Iterable[np.ndarray]) -> float:
        """Return hybrid applicability over a skill's reference states.

        The neural discriminator learns nonlinear state/skill compatibility, while
        the RBF score keeps behavior sane when online data is still small.  The
        hybrid score is the action-set判别标准 used by the V2 controller.
        """
        refs = list(reference_features)
        if not refs:
            return 0.0
        neural = self.neural_score(state_feature, refs)
        rbf = self.rbf_score(state_feature, refs)
        alpha = float(np.clip(self.hybrid_alpha, 0.0, 1.0))
        return float(alpha * neural + (1.0 - alpha) * rbf)

    def save(self, path: str | Path) -> None:
        if self.model is None:
            raise RuntimeError("Cannot save an uninitialized discriminator.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dim": self.state_dim,
                "hidden_dim": self.hidden_dim,
                "threshold": self.threshold,
                "hybrid_alpha": self.hybrid_alpha,
                "rbf_temperature": self.rbf_temperature,
                "seed": self.seed,
                "state_dict": self.model.state_dict(),
                "mean": self.mean,
                "std": self.std,
                "state_mean": self.state_mean,
                "state_std": self.state_std,
                "is_fitted": self.is_fitted,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "StateSkillDiscriminator":
        try:
            data = torch.load(path, map_location=device, weights_only=False)
        except TypeError:
            data = torch.load(path, map_location=device)
        obj = cls(
            state_dim=data["state_dim"],
            hidden_dim=data["hidden_dim"],
            threshold=data["threshold"],
            hybrid_alpha=data.get("hybrid_alpha", 0.5),
            rbf_temperature=data.get("rbf_temperature", 1.5),
            device=device,
            seed=data.get("seed", 42),
        )
        obj._ensure_model(data["state_dim"])
        assert obj.model is not None
        obj.model.load_state_dict(data["state_dict"])
        obj.mean = data["mean"]
        obj.std = data["std"]
        obj.state_mean = data.get("state_mean")
        obj.state_std = data.get("state_std")
        obj.is_fitted = bool(data.get("is_fitted", True))
        obj.model.eval()
        return obj
