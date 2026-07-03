"""Learned skill-level dynamics and success model.

The model predicts the outcome of executing one discovered skill from a local
robot state:

    (state_feature, skill_id) -> delta_x, delta_y, delta_yaw, energy, success_prob

It is intentionally small and data-efficient because the online action set is
updated every exploration batch.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from skill_discovery.utils.logging import get_logger

logger = get_logger("skill_outcome_model")


class _OutcomeNet(nn.Module):
    def __init__(self, state_dim: int, num_skills: int, hidden_dim: int = 128, embedding_dim: int = 16):
        super().__init__()
        self.embedding = nn.Embedding(num_skills, embedding_dim)
        self.trunk = nn.Sequential(
            nn.Linear(state_dim + embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.outcome_head = nn.Linear(hidden_dim, 4)
        self.success_head = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor, skill_id: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.embedding(skill_id)
        h = self.trunk(torch.cat([state, z], dim=-1))
        return self.outcome_head(h), self.success_head(h).squeeze(-1)


@dataclass
class OutcomeTrainStats:
    num_samples: int
    final_loss: float
    final_regression_loss: float
    final_success_loss: float
    success_accuracy: float


class SkillOutcomeModel:
    """State-conditioned skill outcome predictor."""

    def __init__(
        self,
        state_dim: int | None = None,
        num_skills: int | None = None,
        hidden_dim: int = 128,
        embedding_dim: int = 16,
        device: str = "cpu",
        seed: int = 42,
    ):
        self.state_dim = state_dim
        self.num_skills = num_skills
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.device = torch.device(device)
        self.seed = seed
        self.model: _OutcomeNet | None = None
        self.state_mean: np.ndarray | None = None
        self.state_std: np.ndarray | None = None
        self.outcome_mean: np.ndarray | None = None
        self.outcome_std: np.ndarray | None = None
        self.is_fitted = False

    def _ensure_model(self, state_dim: int, num_skills: int) -> None:
        if self.model is None or self.state_dim != state_dim or self.num_skills != num_skills:
            self.state_dim = int(state_dim)
            self.num_skills = int(num_skills)
            self.model = _OutcomeNet(
                state_dim=self.state_dim,
                num_skills=self.num_skills,
                hidden_dim=self.hidden_dim,
                embedding_dim=self.embedding_dim,
            ).to(self.device)

    def fit(
        self,
        action_set: Any,
        epochs: int = 5,
        batch_size: int = 256,
        lr: float = 1e-3,
        success_weight: float = 0.5,
    ) -> OutcomeTrainStats:
        data = action_set.outcome_dataset()
        states = data["state_features"].astype(np.float32)
        skill_ids = data["skill_ids"].astype(np.int64)
        outcomes = data["outcomes"].astype(np.float32)
        success = data["success"].astype(np.float32)
        num_skills = int(data["num_skills"][0])
        self._ensure_model(states.shape[1], num_skills)
        assert self.model is not None

        torch.manual_seed(self.seed)
        self.state_mean = states.mean(axis=0, keepdims=True)
        self.state_std = states.std(axis=0, keepdims=True) + 1e-6
        self.outcome_mean = outcomes.mean(axis=0, keepdims=True)
        self.outcome_std = outcomes.std(axis=0, keepdims=True) + 1e-6
        states_n = ((states - self.state_mean) / self.state_std).astype(np.float32)
        outcomes_n = ((outcomes - self.outcome_mean) / self.outcome_std).astype(np.float32)

        ds = TensorDataset(
            torch.from_numpy(states_n),
            torch.from_numpy(skill_ids),
            torch.from_numpy(outcomes_n),
            torch.from_numpy(success),
        )
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)
        opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        reg_loss_fn = nn.SmoothL1Loss()
        success_loss_fn = nn.BCEWithLogitsLoss()

        final_loss = final_reg = final_success = 0.0
        for _ in range(max(1, epochs)):
            for xb, sid, yb, sb in loader:
                xb = xb.to(self.device)
                sid = sid.to(self.device)
                yb = yb.to(self.device)
                sb = sb.to(self.device)
                pred, logits = self.model(xb, sid)
                reg_loss = reg_loss_fn(pred, yb)
                success_loss = success_loss_fn(logits, sb)
                loss = reg_loss + success_weight * success_loss
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                final_loss = float(loss.detach().cpu())
                final_reg = float(reg_loss.detach().cpu())
                final_success = float(success_loss.detach().cpu())

        with torch.no_grad():
            xb = torch.from_numpy(states_n).to(self.device)
            sid = torch.from_numpy(skill_ids).to(self.device)
            _, logits = self.model(xb, sid)
            probs = torch.sigmoid(logits).cpu().numpy()
        acc = float(np.mean((probs >= 0.5) == (success >= 0.5)))
        self.is_fitted = True
        stats = OutcomeTrainStats(
            num_samples=int(len(states)),
            final_loss=final_loss,
            final_regression_loss=final_reg,
            final_success_loss=final_success,
            success_accuracy=acc,
        )
        logger.info(
            "Trained skill outcome model: samples=%d loss=%.4f reg=%.4f success=%.4f acc=%.3f",
            stats.num_samples,
            stats.final_loss,
            stats.final_regression_loss,
            stats.final_success_loss,
            stats.success_accuracy,
        )
        return stats

    def predict(self, state_feature: np.ndarray, skill_ids: np.ndarray) -> Dict[str, np.ndarray]:
        if not self.is_fitted or self.model is None:
            raise RuntimeError("SkillOutcomeModel is not fitted.")
        assert self.state_mean is not None and self.state_std is not None
        assert self.outcome_mean is not None and self.outcome_std is not None
        state = np.asarray(state_feature, dtype=np.float32).reshape(1, -1)
        skill_ids = np.asarray(skill_ids, dtype=np.int64).reshape(-1)
        states = np.repeat(state, len(skill_ids), axis=0)
        states_n = ((states - self.state_mean) / self.state_std).astype(np.float32)
        with torch.no_grad():
            pred_n, logits = self.model(
                torch.from_numpy(states_n).to(self.device),
                torch.from_numpy(skill_ids).to(self.device),
            )
            pred_n_np = pred_n.cpu().numpy()
            success = torch.sigmoid(logits).cpu().numpy()
        pred = pred_n_np * self.outcome_std + self.outcome_mean
        return {
            "delta_x": pred[:, 0],
            "delta_y": pred[:, 1],
            "delta_yaw": pred[:, 2],
            "energy": np.maximum(pred[:, 3], 0.0),
            "success_prob": success,
        }

    def save(self, path: str | Path) -> None:
        if self.model is None:
            raise RuntimeError("Cannot save an uninitialized outcome model.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dim": self.state_dim,
                "num_skills": self.num_skills,
                "hidden_dim": self.hidden_dim,
                "embedding_dim": self.embedding_dim,
                "seed": self.seed,
                "state_dict": self.model.state_dict(),
                "state_mean": self.state_mean,
                "state_std": self.state_std,
                "outcome_mean": self.outcome_mean,
                "outcome_std": self.outcome_std,
                "is_fitted": self.is_fitted,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "SkillOutcomeModel":
        try:
            data = torch.load(path, map_location=device, weights_only=False)
        except TypeError:
            data = torch.load(path, map_location=device)
        obj = cls(
            state_dim=data["state_dim"],
            num_skills=data["num_skills"],
            hidden_dim=data["hidden_dim"],
            embedding_dim=data["embedding_dim"],
            device=device,
            seed=data.get("seed", 42),
        )
        obj._ensure_model(data["state_dim"], data["num_skills"])
        assert obj.model is not None
        obj.model.load_state_dict(data["state_dict"])
        obj.state_mean = data["state_mean"]
        obj.state_std = data["state_std"]
        obj.outcome_mean = data["outcome_mean"]
        obj.outcome_std = data["outcome_std"]
        obj.is_fitted = bool(data.get("is_fitted", True))
        obj.model.eval()
        return obj
