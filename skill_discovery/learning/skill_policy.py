"""Skill-conditioned behavior cloning policy.

This is the first closed-loop upgrade over replayed action chunks.  The model
learns a low-level policy from the online archive:

    (current_observation, skill_id_or_archive_id[, phase]) -> action

At test time a high-level selector can still choose skills/chunks by predicted
goal progress, while this policy produces each low-level action from the live
robot state instead of replaying a fixed recorded sequence.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from skill_discovery.utils.logging import get_logger

logger = get_logger("skill_policy")

ObsSlice = Tuple[int, int]


class _SkillPolicyNet(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        num_skills: int,
        hidden_dim: int = 256,
        embedding_dim: int = 16,
        use_phase: bool = False,
    ):
        super().__init__()
        self.use_phase = use_phase
        self.embedding = nn.Embedding(num_skills, embedding_dim)
        input_dim = obs_dim + embedding_dim + (1 if use_phase else 0)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(
        self,
        obs: torch.Tensor,
        skill_id: torch.Tensor,
        phase: torch.Tensor | None = None,
    ) -> torch.Tensor:
        z = self.embedding(skill_id)
        parts = [obs, z]
        if self.use_phase:
            if phase is None:
                phase = torch.zeros((obs.shape[0], 1), dtype=obs.dtype, device=obs.device)
            elif phase.ndim == 1:
                phase = phase[:, None]
            parts.append(phase.to(dtype=obs.dtype, device=obs.device))
        return self.net(torch.cat(parts, dim=-1))


@dataclass
class SkillPolicyTrainStats:
    num_samples: int
    num_skills: int
    obs_dim: int
    action_dim: int
    final_loss: float
    action_mse: float


def parse_obs_slices(raw: str | None) -> List[ObsSlice]:
    """Parse comma-separated slices such as '9:12,36:48'."""
    if raw is None or not raw.strip():
        return []
    slices: List[ObsSlice] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        start, end = item.split(":")
        slices.append((int(start), int(end)))
    return slices


class SkillPolicy:
    """Small skill-conditioned behavior-cloning policy."""

    def __init__(
        self,
        obs_dim: int | None = None,
        action_dim: int | None = None,
        num_skills: int | None = None,
        hidden_dim: int = 256,
        embedding_dim: int = 16,
        zero_obs_slices: Sequence[ObsSlice] | None = None,
        label_mode: str = "skill",
        use_phase: bool = False,
        device: str = "cpu",
        seed: int = 42,
    ):
        if label_mode not in {"skill", "archive"}:
            raise ValueError("label_mode must be 'skill' or 'archive'.")
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_skills = num_skills
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.zero_obs_slices = list(zero_obs_slices or [])
        self.label_mode = label_mode
        self.use_phase = bool(use_phase)
        self.device = torch.device(device)
        self.seed = seed
        self.model: _SkillPolicyNet | None = None
        self.obs_mean: np.ndarray | None = None
        self.obs_std: np.ndarray | None = None
        self.action_mean: np.ndarray | None = None
        self.action_std: np.ndarray | None = None
        self.action_min: np.ndarray | None = None
        self.action_max: np.ndarray | None = None
        self.is_fitted = False

    def _ensure_model(self, obs_dim: int, action_dim: int, num_skills: int) -> None:
        if (
            self.model is None
            or self.obs_dim != obs_dim
            or self.action_dim != action_dim
            or self.num_skills != num_skills
        ):
            self.obs_dim = int(obs_dim)
            self.action_dim = int(action_dim)
            self.num_skills = int(num_skills)
            self.model = _SkillPolicyNet(
                obs_dim=self.obs_dim,
                action_dim=self.action_dim,
                num_skills=self.num_skills,
                hidden_dim=self.hidden_dim,
                embedding_dim=self.embedding_dim,
                use_phase=self.use_phase,
            ).to(self.device)

    def _transform_obs(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32).copy()
        for start, end in self.zero_obs_slices:
            if start < obs.shape[-1]:
                obs[..., start : min(end, obs.shape[-1])] = 0.0
        return obs

    def _dataset_from_action_set(
        self,
        action_set: Any,
        max_samples: int | None = None,
        seed: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if not action_set._archive_segments:
            raise ValueError("OnlineActionSet has no archive segments.")
        if not action_set._archive_skill_ids or len(action_set._archive_skill_ids) != len(action_set._archive_segments):
            raise ValueError("OnlineActionSet has no archive skill labels. Run weighted_kmeans update first.")

        obs_rows: List[np.ndarray] = []
        action_rows: List[np.ndarray] = []
        skill_rows: List[int] = []
        phase_rows: List[np.ndarray] = []
        for archive_index, (segment, skill_id) in enumerate(
            zip(action_set._archive_segments, action_set._archive_skill_ids)
        ):
            if skill_id < 0:
                continue
            obs_seq = np.asarray(segment.get("obs_seq"), dtype=np.float32)
            action_seq = np.asarray(segment.get("action_seq"), dtype=np.float32)
            if obs_seq.ndim != 2 or action_seq.ndim != 2:
                continue
            n = min(len(obs_seq), len(action_seq))
            if n <= 0:
                continue
            obs_rows.append(obs_seq[:n])
            action_rows.append(action_seq[:n])
            label = archive_index if self.label_mode == "archive" else int(skill_id)
            skill_rows.extend([int(label)] * n)
            phase_rows.append(np.linspace(0.0, 1.0, n, dtype=np.float32))

        if not obs_rows:
            raise ValueError("No valid observation/action samples found in archive.")
        obs = self._transform_obs(np.concatenate(obs_rows, axis=0))
        actions = np.concatenate(action_rows, axis=0).astype(np.float32)
        skill_ids = np.asarray(skill_rows, dtype=np.int64)
        phases = np.concatenate(phase_rows, axis=0).astype(np.float32)

        if max_samples is not None and len(obs) > max_samples:
            rng = np.random.default_rng(self.seed if seed is None else seed)
            keep = rng.choice(len(obs), size=max_samples, replace=False)
            obs = obs[keep]
            actions = actions[keep]
            skill_ids = skill_ids[keep]
            phases = phases[keep]
        return obs, skill_ids, phases, actions

    def fit(
        self,
        action_set: Any,
        epochs: int = 10,
        batch_size: int = 512,
        lr: float = 3e-4,
        max_samples: int | None = 200000,
        balance_skills: bool = True,
    ) -> SkillPolicyTrainStats:
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        obs, skill_ids, phases, actions = self._dataset_from_action_set(action_set, max_samples=max_samples)
        if self.label_mode == "archive":
            num_skills = len(action_set._archive_segments)
        else:
            num_skills = max(int(np.max(skill_ids)) + 1, len(action_set.skills))
        self._ensure_model(obs.shape[1], actions.shape[1], num_skills)
        assert self.model is not None

        self.obs_mean = obs.mean(axis=0, keepdims=True)
        self.obs_std = obs.std(axis=0, keepdims=True) + 1e-6
        self.action_mean = actions.mean(axis=0, keepdims=True)
        self.action_std = actions.std(axis=0, keepdims=True) + 1e-6
        self.action_min = actions.min(axis=0, keepdims=True)
        self.action_max = actions.max(axis=0, keepdims=True)

        obs_n = ((obs - self.obs_mean) / self.obs_std).astype(np.float32)
        actions_n = ((actions - self.action_mean) / self.action_std).astype(np.float32)
        ds = TensorDataset(
            torch.from_numpy(obs_n),
            torch.from_numpy(skill_ids),
            torch.from_numpy(phases[:, None]),
            torch.from_numpy(actions_n),
        )
        sampler = None
        shuffle = True
        if balance_skills:
            counts = np.bincount(skill_ids, minlength=num_skills).astype(np.float64)
            weights = 1.0 / np.maximum(counts[skill_ids], 1.0)
            sampler = WeightedRandomSampler(
                weights=torch.from_numpy(weights).double(),
                num_samples=len(weights),
                replacement=True,
            )
            shuffle = False
        loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, sampler=sampler, drop_last=False)
        opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-5)
        loss_fn = nn.SmoothL1Loss()

        final_loss = 0.0
        for _ in range(max(1, epochs)):
            self.model.train()
            for xb, sid, pb, yb in loader:
                xb = xb.to(self.device)
                sid = sid.to(self.device)
                pb = pb.to(self.device)
                yb = yb.to(self.device)
                pred = self.model(xb, sid, pb if self.use_phase else None)
                loss = loss_fn(pred, yb)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                opt.step()
                final_loss = float(loss.detach().cpu())

        self.model.eval()
        with torch.no_grad():
            pred_n = self.model(
                torch.from_numpy(obs_n).to(self.device),
                torch.from_numpy(skill_ids).to(self.device),
                torch.from_numpy(phases[:, None]).to(self.device) if self.use_phase else None,
            ).cpu().numpy()
        pred = pred_n * self.action_std + self.action_mean
        mse = float(np.mean((pred - actions) ** 2))
        self.is_fitted = True
        stats = SkillPolicyTrainStats(
            num_samples=int(len(obs)),
            num_skills=int(num_skills),
            obs_dim=int(obs.shape[1]),
            action_dim=int(actions.shape[1]),
            final_loss=final_loss,
            action_mse=mse,
        )
        logger.info(
            "Trained skill policy: samples=%d skills=%d loss=%.4f action_mse=%.6f",
            stats.num_samples,
            stats.num_skills,
            stats.final_loss,
            stats.action_mse,
        )
        return stats

    def predict(self, obs: np.ndarray, skill_id: int, phase: float = 0.0) -> np.ndarray:
        if not self.is_fitted or self.model is None:
            raise RuntimeError("SkillPolicy is not fitted.")
        assert self.obs_mean is not None and self.obs_std is not None
        assert self.action_mean is not None and self.action_std is not None
        obs_arr = self._transform_obs(np.asarray(obs, dtype=np.float32).reshape(1, -1))
        if self.obs_dim is not None and obs_arr.shape[1] != self.obs_dim:
            if obs_arr.shape[1] > self.obs_dim:
                obs_arr = obs_arr[:, : self.obs_dim]
            else:
                padded = np.zeros((1, self.obs_dim), dtype=np.float32)
                padded[:, : obs_arr.shape[1]] = obs_arr
                obs_arr = padded
        obs_n = ((obs_arr - self.obs_mean) / self.obs_std).astype(np.float32)
        sid = np.asarray([int(np.clip(skill_id, 0, max((self.num_skills or 1) - 1, 0)))], dtype=np.int64)
        with torch.no_grad():
            pred_n = self.model(
                torch.from_numpy(obs_n).to(self.device),
                torch.from_numpy(sid).to(self.device),
                torch.asarray([[float(np.clip(phase, 0.0, 1.0))]], dtype=torch.float32, device=self.device)
                if self.use_phase
                else None,
            ).cpu().numpy()
        action = pred_n * self.action_std + self.action_mean
        if self.action_min is not None and self.action_max is not None:
            margin = 0.1 * (self.action_max - self.action_min + 1e-6)
            action = np.clip(action, self.action_min - margin, self.action_max + margin)
        return action.reshape(-1).astype(np.float32)

    def save(self, path: str | Path) -> None:
        if self.model is None:
            raise RuntimeError("Cannot save an uninitialized skill policy.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "obs_dim": self.obs_dim,
                "action_dim": self.action_dim,
                "num_skills": self.num_skills,
                "hidden_dim": self.hidden_dim,
                "embedding_dim": self.embedding_dim,
                "zero_obs_slices": self.zero_obs_slices,
                "label_mode": self.label_mode,
                "use_phase": self.use_phase,
                "seed": self.seed,
                "state_dict": self.model.state_dict(),
                "obs_mean": self.obs_mean,
                "obs_std": self.obs_std,
                "action_mean": self.action_mean,
                "action_std": self.action_std,
                "action_min": self.action_min,
                "action_max": self.action_max,
                "is_fitted": self.is_fitted,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "SkillPolicy":
        try:
            data = torch.load(path, map_location=device, weights_only=False)
        except TypeError:
            data = torch.load(path, map_location=device)
        obj = cls(
            obs_dim=data["obs_dim"],
            action_dim=data["action_dim"],
            num_skills=data["num_skills"],
            hidden_dim=data["hidden_dim"],
            embedding_dim=data["embedding_dim"],
            zero_obs_slices=data.get("zero_obs_slices", []),
            label_mode=data.get("label_mode", "skill"),
            use_phase=data.get("use_phase", False),
            device=device,
            seed=data.get("seed", 42),
        )
        obj._ensure_model(data["obs_dim"], data["action_dim"], data["num_skills"])
        assert obj.model is not None
        obj.model.load_state_dict(data["state_dict"])
        obj.obs_mean = data["obs_mean"]
        obj.obs_std = data["obs_std"]
        obj.action_mean = data["action_mean"]
        obj.action_std = data["action_std"]
        obj.action_min = data.get("action_min")
        obj.action_max = data.get("action_max")
        obj.is_fitted = bool(data.get("is_fitted", True))
        obj.model.eval()
        return obj
