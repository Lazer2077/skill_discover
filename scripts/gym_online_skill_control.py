"""Online skill/action-set control in OpenAI Gymnasium environments.

Default target: ``Pendulum-v1`` because it is continuous-control and ships with
Gymnasium by default.  The script implements the same V2.1 ideas without Isaac:

* online archive of action chunks,
* weighted k-means over chunk descriptors,
* learning-based state-skill discriminator,
* learned state-conditioned skill outcome model,
* receding-horizon skill MPC,
* bandit-style exploration that increasingly reuses high-value skills.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from skill_discovery.learning.state_skill_discriminator import StateSkillDiscriminator


def make_gym_env(env_id: str, args: argparse.Namespace) -> gym.Env:
    try:
        return gym.make(env_id, max_episode_steps=args.max_episode_steps)
    except TypeError:
        return gym.make(env_id)


@dataclass
class GymSkill:
    skill_id: int
    action_sequences: List[np.ndarray]
    initial_state_features: List[np.ndarray]
    center_descriptor: np.ndarray
    mean_return: float
    mean_utility: float
    mean_progress: float
    mean_energy: float
    num_segments: int

    @property
    def action_sequence(self) -> np.ndarray:
        return self.action_sequences[0]


@dataclass
class EliteEpisode:
    start_obs: np.ndarray
    final_obs: np.ndarray
    actions: np.ndarray
    total_return: float
    length: int


class GymActionSet:
    def __init__(self, max_skills: int, max_archive_size: int = 4000):
        self.max_skills = max_skills
        self.max_archive_size = max_archive_size
        self.skills: Dict[int, GymSkill] = {}
        self.archive: List[Dict[str, np.ndarray | float]] = []
        self.elite_episodes: List[EliteEpisode] = []

    @property
    def skill_ids(self) -> List[int]:
        return sorted(self.skills)

    def update(self, chunks: List[Dict[str, Any]], elite_episodes: List[EliteEpisode] | None = None) -> Dict[str, Any]:
        self.archive.extend(chunks)
        if elite_episodes:
            self.elite_episodes.extend(elite_episodes)
            self.elite_episodes.sort(key=lambda ep: ep.total_return, reverse=True)
            self.elite_episodes = self.elite_episodes[:16]
        if len(self.archive) > self.max_archive_size:
            self.archive.sort(key=lambda c: float(c["utility"]), reverse=True)
            self.archive = self.archive[: self.max_archive_size]
        self._recluster()
        return {
            "chunks": len(chunks),
            "archive": len(self.archive),
            "num_skills": len(self.skills),
            "elite_episodes": len(self.elite_episodes),
        }

    def _recluster(self) -> None:
        if not self.archive:
            return
        desc = np.stack([c["descriptor"] for c in self.archive]).astype(np.float64)
        k = min(self.max_skills, len(desc))
        X = self._weighted_descriptor_matrix(desc)
        labels = self._kmeans(X, k)
        utilities = np.asarray([float(c["utility"]) for c in self.archive])
        new_skills: Dict[int, GymSkill] = {}
        cluster_ids = sorted(np.unique(labels), key=lambda cid: float(np.mean(utilities[labels == cid])), reverse=True)
        for new_id, cid in enumerate(cluster_ids):
            idx = np.where(labels == cid)[0]
            members = [self.archive[i] for i in idx]
            members.sort(key=lambda c: float(c["utility"]), reverse=True)
            reps = [np.asarray(c["actions"], dtype=np.float32) for c in members[:3]]
            init_features = [np.asarray(c["start_obs"], dtype=np.float32) for c in members[:128]]
            center = desc[idx].mean(axis=0)
            new_skills[new_id] = GymSkill(
                skill_id=new_id,
                action_sequences=reps,
                initial_state_features=init_features,
                center_descriptor=center,
                mean_return=float(np.mean([c["return"] for c in members])),
                mean_utility=float(np.mean([c["utility"] for c in members])),
                mean_progress=float(np.mean([c["progress"] for c in members])),
                mean_energy=float(np.mean([c["energy"] for c in members])),
                num_segments=len(members),
            )
            for c in members:
                c["skill_id"] = new_id
        self.skills = new_skills

    def _weighted_descriptor_matrix(self, desc: np.ndarray) -> np.ndarray:
        scale = desc.std(axis=0, keepdims=True) + 1e-6
        X = (desc - desc.mean(axis=0, keepdims=True)) / scale
        # descriptor layout: delta_obs, start_obs, return, energy, smoothness,
        # delta_norm, progress, final_task_score
        obs_dim = (desc.shape[1] - 6) // 2
        weights = np.ones(desc.shape[1])
        weights[:obs_dim] = 2.0
        weights[obs_dim : 2 * obs_dim] = 0.4
        weights[2 * obs_dim] = 1.5      # return
        weights[2 * obs_dim + 1] = 0.5  # energy
        weights[2 * obs_dim + 2] = 0.8  # smoothness
        weights[2 * obs_dim + 3] = 1.2  # delta norm
        weights[2 * obs_dim + 4] = 2.0  # task progress
        weights[2 * obs_dim + 5] = 1.0  # final task score
        return X * weights[None]

    @staticmethod
    def _kmeans(X: np.ndarray, k: int, max_iters: int = 50) -> np.ndarray:
        if k <= 1:
            return np.zeros(len(X), dtype=np.int64)
        centers = [0]
        min_d = np.linalg.norm(X - X[0], axis=1)
        for _ in range(1, k):
            idx = int(np.argmax(min_d))
            centers.append(idx)
            min_d = np.minimum(min_d, np.linalg.norm(X - X[idx], axis=1))
        C = X[centers].copy()
        labels = np.zeros(len(X), dtype=np.int64)
        for _ in range(max_iters):
            d = np.linalg.norm(X[:, None] - C[None], axis=-1)
            new = np.argmin(d, axis=1)
            if np.array_equal(labels, new):
                break
            labels = new
            for i in range(k):
                mask = labels == i
                if np.any(mask):
                    C[i] = X[mask].mean(axis=0)
        return labels

    def outcome_dataset(self) -> Dict[str, np.ndarray]:
        rows = [c for c in self.archive if "skill_id" in c]
        return {
            "states": np.stack([c["start_obs"] for c in rows]).astype(np.float32),
            "skill_ids": np.asarray([c["skill_id"] for c in rows], dtype=np.int64),
            "delta_obs": np.stack([c["delta_obs"] for c in rows]).astype(np.float32),
            "returns": np.asarray([c["return"] for c in rows], dtype=np.float32),
            "success": np.asarray([c["success"] for c in rows], dtype=np.float32),
            "num_skills": np.asarray([len(self.skills)], dtype=np.int64),
        }


class GymOutcomeNet(nn.Module):
    def __init__(self, obs_dim: int, num_skills: int, hidden: int = 128, emb: int = 16):
        super().__init__()
        self.embedding = nn.Embedding(num_skills, emb)
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim + emb, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.delta_head = nn.Linear(hidden, obs_dim)
        self.return_head = nn.Linear(hidden, 1)
        self.success_head = nn.Linear(hidden, 1)

    def forward(self, obs: torch.Tensor, sid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.trunk(torch.cat([obs, self.embedding(sid)], dim=-1))
        return self.delta_head(h), self.return_head(h).squeeze(-1), self.success_head(h).squeeze(-1)


class GymOutcomeModel:
    def __init__(self, device: str = "cpu", seed: int = 42):
        self.device = torch.device(device)
        self.seed = seed
        self.model: GymOutcomeNet | None = None
        self.obs_mean = self.obs_std = None
        self.delta_mean = self.delta_std = None
        self.return_mean = 0.0
        self.return_std = 1.0

    def fit(self, action_set: GymActionSet, epochs: int = 8, batch_size: int = 256, lr: float = 1e-3) -> Dict[str, float]:
        data = action_set.outcome_dataset()
        obs = data["states"]
        sid = data["skill_ids"]
        delta = data["delta_obs"]
        returns = data["returns"]
        success = data["success"]
        torch.manual_seed(self.seed)
        self.obs_mean = obs.mean(axis=0, keepdims=True)
        self.obs_std = obs.std(axis=0, keepdims=True) + 1e-6
        self.delta_mean = delta.mean(axis=0, keepdims=True)
        self.delta_std = delta.std(axis=0, keepdims=True) + 1e-6
        self.return_mean = float(returns.mean())
        self.return_std = float(returns.std() + 1e-6)
        obs_n = ((obs - self.obs_mean) / self.obs_std).astype(np.float32)
        delta_n = ((delta - self.delta_mean) / self.delta_std).astype(np.float32)
        ret_n = ((returns - self.return_mean) / self.return_std).astype(np.float32)
        self.model = GymOutcomeNet(obs.shape[1], int(data["num_skills"][0])).to(self.device)
        ds = TensorDataset(
            torch.from_numpy(obs_n),
            torch.from_numpy(sid),
            torch.from_numpy(delta_n),
            torch.from_numpy(ret_n),
            torch.from_numpy(success),
        )
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
        opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        loss_reg = nn.SmoothL1Loss()
        loss_bin = nn.BCEWithLogitsLoss()
        final = 0.0
        for _ in range(max(1, epochs)):
            for ob, s, d, r, ok in loader:
                ob, s, d, r, ok = ob.to(self.device), s.to(self.device), d.to(self.device), r.to(self.device), ok.to(self.device)
                pd, pr, ps = self.model(ob, s)
                loss = loss_reg(pd, d) + 0.5 * loss_reg(pr, r) + 0.3 * loss_bin(ps, ok)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                final = float(loss.detach().cpu())
        return {"samples": int(len(obs)), "loss": final}

    def predict(self, obs: np.ndarray, skill_ids: np.ndarray) -> Dict[str, np.ndarray]:
        assert self.model is not None
        obs = np.asarray(obs, dtype=np.float32).reshape(1, -1)
        skill_ids = np.asarray(skill_ids, dtype=np.int64)
        obs_rep = np.repeat(obs, len(skill_ids), axis=0)
        obs_n = ((obs_rep - self.obs_mean) / self.obs_std).astype(np.float32)
        with torch.no_grad():
            d, r, ok = self.model(
                torch.from_numpy(obs_n).to(self.device),
                torch.from_numpy(skill_ids).to(self.device),
            )
        delta = d.cpu().numpy() * self.delta_std + self.delta_mean
        returns = r.cpu().numpy() * self.return_std + self.return_mean
        success = torch.sigmoid(ok).cpu().numpy()
        return {"delta_obs": delta, "return": returns, "success": success}


def task_score(env_id: str, obs: np.ndarray) -> float:
    """Dense diagnostic score used only for skill discovery/planning guidance."""
    o = np.asarray(obs, dtype=np.float32)
    if env_id.startswith("Pendulum") and o.shape[0] >= 3:
        theta = float(np.arctan2(o[1], o[0]))
        theta_dot = float(o[2])
        return -(theta * theta + 0.1 * theta_dot * theta_dot)
    if env_id.startswith("MountainCarContinuous") and o.shape[0] >= 2:
        position = float(o[0])
        velocity = float(o[1])
        return position + 5.0 * abs(velocity)
    return 0.0


def task_success(env_id: str, obs: np.ndarray, ret: float) -> float:
    o = np.asarray(obs, dtype=np.float32)
    if env_id.startswith("Pendulum") and o.shape[0] >= 3:
        return float(task_score(env_id, o) > -0.25)
    if env_id.startswith("MountainCarContinuous") and o.shape[0] >= 2:
        return float(o[0] >= 0.45 or ret > 50.0)
    return float(ret > 0.0)


def is_elite_episode(env_id: str, final_obs: np.ndarray, ep_return: float, min_return: float) -> bool:
    if env_id.startswith("MountainCarContinuous"):
        return bool(task_success(env_id, final_obs, ep_return) or ep_return >= min_return)
    return bool(ep_return >= min_return)


def make_chunk(obs_seq: List[np.ndarray], act_seq: List[np.ndarray], rew_seq: List[float], args: argparse.Namespace) -> Dict[str, Any]:
    obs = np.asarray(obs_seq, dtype=np.float32)
    actions = np.asarray(act_seq, dtype=np.float32)
    rewards = np.asarray(rew_seq, dtype=np.float32)
    delta = obs[-1] - obs[0]
    energy = float(np.mean(np.square(actions)))
    smooth = float(np.exp(-np.mean(np.abs(np.diff(actions, axis=0)))) if len(actions) > 1 else 1.0)
    ret = float(np.sum(rewards))
    start_score = task_score(args.env, obs[0])
    final_score = task_score(args.env, obs[-1])
    progress = final_score - start_score
    descriptor = np.concatenate(
        [
            delta,
            obs[0],
            np.asarray([ret, energy, smooth, np.linalg.norm(delta), progress, final_score], dtype=np.float32),
        ]
    )
    utility = (
        ret
        + args.utility_progress_weight * progress
        + args.utility_state_weight * final_score
        + 0.5 * float(np.linalg.norm(delta))
        - 0.05 * energy
        + 0.1 * smooth
    )
    return {
        "start_obs": obs[0],
        "delta_obs": delta,
        "actions": actions,
        "return": ret,
        "energy": energy,
        "progress": progress,
        "final_score": final_score,
        "success": task_success(args.env, obs[-1], ret),
        "utility": utility,
        "descriptor": descriptor.astype(np.float32),
    }


def sample_exploration_actions(env: gym.Env, args: argparse.Namespace, rng: np.random.Generator) -> np.ndarray:
    low, high = env.action_space.low, env.action_space.high
    shape = (args.chunk_horizon, env.action_space.shape[0])
    mode = rng.random()
    if mode < 0.55:
        base = rng.uniform(low, high, size=(1, env.action_space.shape[0]))
        noise = rng.normal(0.0, 0.08 * (high - low), size=shape)
        return np.clip(base + noise, low, high)
    if mode < 0.8:
        start = rng.uniform(low, high, size=(1, env.action_space.shape[0]))
        end = rng.uniform(low, high, size=(1, env.action_space.shape[0]))
        alpha = np.linspace(0.0, 1.0, args.chunk_horizon, dtype=np.float32)[:, None]
        return np.clip((1.0 - alpha) * start + alpha * end, low, high)
    return rng.uniform(low, high, size=shape)


def collect_chunks(
    env: gym.Env,
    action_set: GymActionSet,
    args: argparse.Namespace,
    rng: np.random.Generator,
    iteration: int,
) -> tuple[List[Dict[str, Any]], List[EliteEpisode]]:
    chunks: List[Dict[str, Any]] = []
    elites: List[EliteEpisode] = []
    low, high = env.action_space.low, env.action_space.high
    for ep in range(args.episodes_per_iter):
        obs, _ = env.reset(seed=args.seed + 1000 * iteration + ep)
        start_obs = obs.copy()
        obs_seq, act_seq, rew_seq = [obs.copy()], [], []
        ep_actions: List[np.ndarray] = []
        ep_rewards: List[float] = []
        t = 0
        done = False
        while t < args.max_episode_steps and not done:
            use_skill = action_set.skills and rng.random() > args.random_action_prob
            if use_skill:
                ids = action_set.skill_ids
                vals = np.asarray([action_set.skills[s].mean_utility for s in ids], dtype=np.float64)
                probs = np.exp((vals - vals.max()) / max(args.bandit_temperature, 1e-6))
                probs = probs / probs.sum()
                skill = action_set.skills[int(rng.choice(ids, p=probs))]
                actions = skill.action_sequence + rng.normal(0, args.skill_action_noise, skill.action_sequence.shape)
            else:
                actions = sample_exploration_actions(env, args, rng)
            for action in actions:
                action = np.clip(action, low, high).astype(np.float32)
                next_obs, reward, terminated, truncated, _ = env.step(action)
                act_seq.append(action)
                rew_seq.append(float(reward))
                ep_actions.append(action.copy())
                ep_rewards.append(float(reward))
                obs = next_obs
                obs_seq.append(obs.copy())
                t += 1
                if len(act_seq) == args.chunk_horizon:
                    chunks.append(make_chunk(obs_seq, act_seq, rew_seq, args))
                    obs_seq, act_seq, rew_seq = [obs.copy()], [], []
                done = terminated or truncated
                if done or t >= args.max_episode_steps:
                    break
            if terminated and len(act_seq) >= max(2, args.chunk_horizon // 4):
                chunks.append(make_chunk(obs_seq, act_seq, rew_seq, args))
        ep_return = float(np.sum(ep_rewards))
        if ep_actions and is_elite_episode(args.env, obs, ep_return, args.elite_min_return):
            elites.append(
                EliteEpisode(
                    start_obs=start_obs.astype(np.float32),
                    final_obs=np.asarray(obs, dtype=np.float32),
                    actions=np.asarray(ep_actions, dtype=np.float32),
                    total_return=ep_return,
                    length=len(ep_actions),
                )
            )
    return chunks, elites


def train_discriminator(action_set: GymActionSet, args: argparse.Namespace) -> StateSkillDiscriminator:
    disc = StateSkillDiscriminator(
        threshold=0.55,
        hybrid_alpha=0.5,
        rbf_temperature=1.5,
        device=args.device,
        seed=args.seed,
    )
    disc.fit(action_set, epochs=args.discriminator_epochs, batch_size=128, negative_ratio=2)
    return disc


def skill_mpc_episode(env: gym.Env, action_set: GymActionSet, disc: StateSkillDiscriminator, model: GymOutcomeModel, args: argparse.Namespace, seed: int) -> float:
    obs, _ = env.reset(seed=seed)
    total = 0.0
    ids = action_set.skill_ids
    ids_arr = np.asarray(ids, dtype=np.int64)
    low, high = env.action_space.low, env.action_space.high
    obs_low = getattr(env.observation_space, "low", None)
    obs_high = getattr(env.observation_space, "high", None)
    t = 0
    while t < args.max_episode_steps:
        reliability = {sid: min(1.0, np.log1p(action_set.skills[sid].num_segments) / np.log1p(8.0)) for sid in ids}
        beams: List[tuple[tuple[int, ...], np.ndarray, float]] = [((), obs.copy(), 0.0)]
        for _depth in range(args.mpc_horizon):
            expanded: List[tuple[tuple[int, ...], np.ndarray, float]] = []
            for prefix, sim_obs, prefix_cost in beams:
                pred = model.predict(sim_obs, ids_arr)
                current_score = task_score(args.env, sim_obs)
                for i, sid in enumerate(ids):
                    next_obs = sim_obs + pred["delta_obs"][i]
                    if obs_low is not None and obs_high is not None and np.all(np.isfinite(obs_low)) and np.all(np.isfinite(obs_high)):
                        next_obs = np.clip(next_obs, obs_low, obs_high)
                    next_score = task_score(args.env, next_obs)
                    applicability = disc.rbf_score(sim_obs, action_set.skills[sid].initial_state_features)
                    step_cost = -float(pred["return"][i])
                    step_cost += args.lambda_discriminator * (1.0 - applicability)
                    step_cost += args.lambda_success * (1.0 - float(pred["success"][i]))
                    step_cost += args.lambda_reliability * (1.0 - reliability[sid])
                    step_cost += -args.lambda_task_score * next_score
                    step_cost += -args.lambda_task_progress * (next_score - current_score)
                    expanded.append((prefix + (sid,), next_obs, prefix_cost + step_cost))
            expanded.sort(key=lambda item: item[2])
            beams = expanded[: args.mpc_beam_width]
        best_seq = beams[0][0]
        skill = action_set.skills[int(best_seq[0])]
        for action in skill.action_sequence:
            obs, reward, terminated, truncated, _ = env.step(np.clip(action, low, high).astype(np.float32))
            total += float(reward)
            t += 1
            if terminated or truncated:
                return total
            if t >= args.max_episode_steps:
                break
    return total


def baseline_episode(env: gym.Env, mode: str, args: argparse.Namespace, seed: int) -> float:
    rng = np.random.default_rng(seed)
    obs, _ = env.reset(seed=seed)
    total = 0.0
    for _ in range(args.max_episode_steps):
        if mode == "zero":
            action = np.zeros(env.action_space.shape, dtype=np.float32)
        else:
            action = rng.uniform(env.action_space.low, env.action_space.high).astype(np.float32)
        obs, reward, terminated, truncated, _ = env.step(action)
        total += float(reward)
        if terminated or truncated:
            break
    return total


def elite_replay_episode(env: gym.Env, action_set: GymActionSet, args: argparse.Namespace, seed: int) -> float:
    if not action_set.elite_episodes:
        return float("nan")
    obs, _ = env.reset(seed=seed)
    starts = np.stack([ep.start_obs for ep in action_set.elite_episodes])
    scale = starts.std(axis=0, keepdims=True) + 0.25
    dists = np.linalg.norm((starts - obs.reshape(1, -1)) / scale, axis=-1)
    elite = action_set.elite_episodes[int(np.argmin(dists))]
    total = 0.0
    low, high = env.action_space.low, env.action_space.high
    for action in elite.actions[: args.max_episode_steps]:
        obs, reward, terminated, truncated, _ = env.step(np.clip(action, low, high).astype(np.float32))
        total += float(reward)
        if terminated or truncated:
            break
    return total


def nearest_chunk_episode(env: gym.Env, action_set: GymActionSet, args: argparse.Namespace, seed: int) -> float:
    if not action_set.archive:
        return float("nan")
    starts = np.stack([np.asarray(c["start_obs"], dtype=np.float32) for c in action_set.archive])
    utilities = np.asarray([float(c["utility"]) for c in action_set.archive], dtype=np.float32)
    util_std = float(utilities.std() + 1e-6)
    utilities = (utilities - float(utilities.mean())) / util_std
    scale = starts.std(axis=0, keepdims=True) + 0.25
    obs, _ = env.reset(seed=seed)
    low, high = env.action_space.low, env.action_space.high
    total = 0.0
    t = 0
    while t < args.max_episode_steps:
        dists = np.linalg.norm((starts - obs.reshape(1, -1)) / scale, axis=-1) / np.sqrt(starts.shape[-1])
        scores = -dists + args.nn_utility_weight * utilities
        chunk = action_set.archive[int(np.argmax(scores))]
        for action in np.asarray(chunk["actions"], dtype=np.float32):
            obs, reward, terminated, truncated, _ = env.step(np.clip(action, low, high).astype(np.float32))
            total += float(reward)
            t += 1
            if terminated or truncated or t >= args.max_episode_steps:
                return total
    return total


def default_rl_target_reward(env_id: str) -> float | None:
    if env_id.startswith("Pendulum"):
        return -250.0
    if env_id.startswith("MountainCarContinuous"):
        return 90.0
    return None


def rl_policy_episode(env: gym.Env, rl_model: Any, args: argparse.Namespace, seed: int) -> float:
    obs, _ = env.reset(seed=seed)
    total = 0.0
    for _ in range(args.max_episode_steps):
        action, _ = rl_model.predict(obs, deterministic=True)
        action = np.clip(action, env.action_space.low, env.action_space.high).astype(np.float32)
        obs, reward, terminated, truncated, _ = env.step(action)
        total += float(reward)
        if terminated or truncated:
            break
    return total


def evaluate_rl_policy(rl_model: Any, args: argparse.Namespace, seed_offset: int = 40000) -> Dict[str, Any]:
    env = make_gym_env(args.env, args)
    returns = [
        rl_policy_episode(env, rl_model, args, args.seed + seed_offset + i)
        for i in range(args.rl_eval_episodes)
    ]
    env.close()
    return {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "returns": [float(x) for x in returns],
    }


def train_or_load_rl_policy(args: argparse.Namespace) -> tuple[Any, Dict[str, Any]]:
    try:
        from stable_baselines3 import PPO, SAC, TD3
    except Exception as exc:  # pragma: no cover - depends on optional package.
        raise RuntimeError(
            "source_policy=rl requires stable-baselines3. Install with `python -m pip install stable-baselines3`."
        ) from exc

    algos = {"SAC": SAC, "TD3": TD3, "PPO": PPO}
    algo_name = args.rl_algo.upper()
    if algo_name not in algos:
        raise ValueError(f"Unsupported --rl_algo {args.rl_algo!r}. Choose one of {sorted(algos)}.")
    Algo = algos[algo_name]

    output_path = Path(args.output)
    model_path = Path(args.rl_model_path) if args.rl_model_path else output_path.with_suffix(f".{algo_name.lower()}.zip")
    if args.rl_load_if_exists and model_path.exists():
        model = Algo.load(str(model_path), device=args.device)
        stats = {
            "algo": algo_name,
            "model_path": str(model_path),
            "loaded": True,
            "timesteps": 0,
            "converged": None,
            "eval_history": [],
        }
        return model, stats

    train_env = make_gym_env(args.env, args)
    common_kwargs = {"seed": args.seed, "verbose": 0, "device": args.device}
    if algo_name == "SAC":
        model = Algo(
            "MlpPolicy",
            train_env,
            learning_rate=args.rl_learning_rate,
            learning_starts=min(args.rl_learning_starts, max(1, args.rl_train_steps // 10)),
            batch_size=args.rl_batch_size,
            train_freq=1,
            gradient_steps=1,
            **common_kwargs,
        )
    elif algo_name == "TD3":
        model = Algo(
            "MlpPolicy",
            train_env,
            learning_rate=args.rl_learning_rate,
            learning_starts=min(args.rl_learning_starts, max(1, args.rl_train_steps // 10)),
            batch_size=args.rl_batch_size,
            train_freq=1,
            gradient_steps=1,
            **common_kwargs,
        )
    else:
        model = Algo(
            "MlpPolicy",
            train_env,
            learning_rate=args.rl_learning_rate,
            n_steps=min(2048, max(64, args.rl_eval_freq)),
            batch_size=min(args.rl_batch_size, 256),
            **common_kwargs,
        )

    target = args.rl_target_reward
    if target is None:
        target = default_rl_target_reward(args.env)
    eval_history: List[Dict[str, Any]] = []
    timesteps = 0
    target_streak = 0
    converged = False
    while timesteps < args.rl_train_steps:
        step_chunk = min(args.rl_eval_freq, args.rl_train_steps - timesteps)
        model.learn(total_timesteps=step_chunk, reset_num_timesteps=(timesteps == 0), progress_bar=False)
        timesteps += step_chunk
        eval_stats = evaluate_rl_policy(model, args, seed_offset=41000 + timesteps)
        eval_stats["timesteps"] = timesteps
        eval_history.append(eval_stats)
        print(
            f"[rl] {algo_name} steps={timesteps} mean_return={eval_stats['mean_return']:.2f} "
            f"target={target if target is not None else 'none'}",
            flush=True,
        )
        if target is not None and timesteps >= args.rl_min_steps_before_stop and eval_stats["mean_return"] >= target:
            target_streak += 1
        else:
            target_streak = 0
        if target is not None and target_streak >= args.rl_patience_evals:
            converged = True
            break

    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path))
    train_env.close()
    return model, {
        "algo": algo_name,
        "model_path": str(model_path),
        "loaded": False,
        "timesteps": timesteps,
        "target_reward": target,
        "target_streak": target_streak,
        "converged": converged,
        "eval_history": eval_history,
    }


def collect_chunks_from_rl_policy(
    env: gym.Env,
    rl_model: Any,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> tuple[List[Dict[str, Any]], List[EliteEpisode]]:
    chunks: List[Dict[str, Any]] = []
    elites: List[EliteEpisode] = []
    low, high = env.action_space.low, env.action_space.high
    for ep in range(args.rl_collect_episodes):
        obs, _ = env.reset(seed=args.seed + 50000 + ep)
        start_obs = obs.copy()
        obs_seq, act_seq, rew_seq = [obs.copy()], [], []
        ep_actions: List[np.ndarray] = []
        ep_rewards: List[float] = []
        for _ in range(args.max_episode_steps):
            action, _ = rl_model.predict(obs, deterministic=not args.rl_stochastic_collect)
            if args.rl_action_noise > 0.0:
                action = action + rng.normal(0.0, args.rl_action_noise, size=np.asarray(action).shape)
            action = np.clip(action, low, high).astype(np.float32)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            act_seq.append(action)
            rew_seq.append(float(reward))
            ep_actions.append(action.copy())
            ep_rewards.append(float(reward))
            obs = next_obs
            obs_seq.append(obs.copy())
            if len(act_seq) == args.chunk_horizon:
                chunks.append(make_chunk(obs_seq, act_seq, rew_seq, args))
                obs_seq, act_seq, rew_seq = [obs.copy()], [], []
            if terminated or truncated:
                break
        ep_return = float(np.sum(ep_rewards))
        if ep_actions and is_elite_episode(args.env, obs, ep_return, args.elite_min_return):
            elites.append(
                EliteEpisode(
                    start_obs=start_obs.astype(np.float32),
                    final_obs=np.asarray(obs, dtype=np.float32),
                    actions=np.asarray(ep_actions, dtype=np.float32),
                    total_return=ep_return,
                    length=len(ep_actions),
                )
            )
    return chunks, elites


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Gymnasium online skill/action-set control.")
    p.add_argument("--env", type=str, default="Pendulum-v1")
    p.add_argument("--source_policy", type=str, choices=["online", "rl"], default="online")
    p.add_argument("--online_iterations", type=int, default=5)
    p.add_argument("--episodes_per_iter", type=int, default=20)
    p.add_argument("--max_episode_steps", type=int, default=200)
    p.add_argument("--chunk_horizon", type=int, default=16)
    p.add_argument("--max_skills", type=int, default=8)
    p.add_argument("--max_archive_size", type=int, default=4000)
    p.add_argument("--random_action_prob", type=float, default=0.35)
    p.add_argument("--skill_action_noise", type=float, default=0.08)
    p.add_argument("--bandit_temperature", type=float, default=8.0)
    p.add_argument("--discriminator_epochs", type=int, default=4)
    p.add_argument("--outcome_epochs", type=int, default=8)
    p.add_argument("--mpc_horizon", type=int, default=2)
    p.add_argument("--mpc_beam_width", type=int, default=16)
    p.add_argument("--eval_episodes", type=int, default=10)
    p.add_argument("--skip_skill_mpc", action="store_true")
    p.add_argument("--skip_baselines", action="store_true")
    p.add_argument("--lambda_discriminator", type=float, default=0.5)
    p.add_argument("--lambda_success", type=float, default=0.5)
    p.add_argument("--lambda_reliability", type=float, default=0.2)
    p.add_argument("--lambda_task_score", type=float, default=0.5)
    p.add_argument("--lambda_task_progress", type=float, default=3.0)
    p.add_argument("--utility_progress_weight", type=float, default=10.0)
    p.add_argument("--utility_state_weight", type=float, default=0.5)
    p.add_argument("--elite_min_return", type=float, default=50.0)
    p.add_argument("--nn_utility_weight", type=float, default=0.0)
    p.add_argument("--rl_algo", type=str, default="SAC", choices=["SAC", "TD3", "PPO"])
    p.add_argument("--rl_train_steps", type=int, default=50000)
    p.add_argument("--rl_eval_freq", type=int, default=5000)
    p.add_argument("--rl_eval_episodes", type=int, default=5)
    p.add_argument("--rl_target_reward", type=float, default=None)
    p.add_argument("--rl_min_steps_before_stop", type=int, default=10000)
    p.add_argument("--rl_patience_evals", type=int, default=2)
    p.add_argument("--rl_collect_episodes", type=int, default=50)
    p.add_argument("--rl_learning_rate", type=float, default=3e-4)
    p.add_argument("--rl_learning_starts", type=int, default=1000)
    p.add_argument("--rl_batch_size", type=int, default=256)
    p.add_argument("--rl_action_noise", type=float, default=0.0)
    p.add_argument("--rl_stochastic_collect", action="store_true")
    p.add_argument("--rl_model_path", type=str, default="")
    p.add_argument("--rl_load_if_exists", action="store_true")
    p.add_argument("--allow_unconverged_rl", action="store_true")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=str, default="outputs/gym_online_skill_control/summary.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    env = make_gym_env(args.env, args)
    action_set = GymActionSet(max_skills=args.max_skills, max_archive_size=args.max_archive_size)
    history = []
    disc = None
    model = None
    rl_model = None
    rl_training = None

    if args.source_policy == "online":
        for it in range(args.online_iterations):
            chunks, elites = collect_chunks(env, action_set, args, rng, it)
            stats = action_set.update(chunks, elites)
            if len(action_set.skills) >= 2:
                disc = train_discriminator(action_set, args)
                model = GymOutcomeModel(device=args.device, seed=args.seed)
                outcome_stats = model.fit(action_set, epochs=args.outcome_epochs)
            else:
                outcome_stats = None
            history.append(
                {
                    "iteration": it,
                    "source_policy": "online",
                    **stats,
                    "skill_returns": {str(sid): action_set.skills[sid].mean_return for sid in action_set.skill_ids},
                    "skill_utilities": {str(sid): action_set.skills[sid].mean_utility for sid in action_set.skill_ids},
                    "skill_progress": {str(sid): action_set.skills[sid].mean_progress for sid in action_set.skill_ids},
                    "outcome": outcome_stats,
                }
            )
    else:
        rl_model, rl_training = train_or_load_rl_policy(args)
        if (
            not args.allow_unconverged_rl
            and not bool(rl_training.get("loaded"))
            and rl_training.get("target_reward") is not None
            and not bool(rl_training.get("converged"))
        ):
            raise RuntimeError(
                "RL policy did not reach the convergence target. "
                "Increase --rl_train_steps/lower --rl_target_reward, or pass --allow_unconverged_rl."
            )
        chunks, elites = collect_chunks_from_rl_policy(env, rl_model, args, rng)
        stats = action_set.update(chunks, elites)
        if len(action_set.skills) >= 2:
            disc = train_discriminator(action_set, args)
            model = GymOutcomeModel(device=args.device, seed=args.seed)
            outcome_stats = model.fit(action_set, epochs=args.outcome_epochs)
        else:
            outcome_stats = None
        history.append(
            {
                "iteration": 0,
                "source_policy": "rl",
                **stats,
                "rl_collected_episodes": args.rl_collect_episodes,
                "skill_returns": {str(sid): action_set.skills[sid].mean_return for sid in action_set.skill_ids},
                "skill_utilities": {str(sid): action_set.skills[sid].mean_utility for sid in action_set.skill_ids},
                "skill_progress": {str(sid): action_set.skills[sid].mean_progress for sid in action_set.skill_ids},
                "outcome": outcome_stats,
            }
        )

    assert disc is not None and model is not None
    evals: Dict[str, List[float]] = {}
    if not args.skip_baselines:
        evals.update({"random_action": [], "zero_action": []})
    if not args.skip_skill_mpc:
        evals["skill_mpc"] = []
    if rl_model is not None:
        evals["rl_policy"] = []
        evals["nearest_chunk"] = []
    if action_set.elite_episodes:
        evals["elite_replay"] = []
    for i in range(args.eval_episodes):
        if "skill_mpc" in evals:
            evals["skill_mpc"].append(skill_mpc_episode(env, action_set, disc, model, args, args.seed + 10000 + i))
        if rl_model is not None:
            evals["rl_policy"].append(rl_policy_episode(env, rl_model, args, args.seed + 12000 + i))
            evals["nearest_chunk"].append(nearest_chunk_episode(env, action_set, args, args.seed + 13000 + i))
        if "elite_replay" in evals:
            evals["elite_replay"].append(elite_replay_episode(env, action_set, args, args.seed + 15000 + i))
        if "random_action" in evals:
            evals["random_action"].append(baseline_episode(env, "random", args, args.seed + 20000 + i))
        if "zero_action" in evals:
            evals["zero_action"].append(baseline_episode(env, "zero", args, args.seed + 30000 + i))

    summary = {
        "env": args.env,
        "source_policy": args.source_policy,
        "num_skills": len(action_set.skills),
        "elite_episodes": len(action_set.elite_episodes),
        "rl_training": rl_training,
        "history": history,
        "eval": {
            k: {
                "mean_return": float(np.mean(v)),
                "std_return": float(np.std(v)),
                "returns": [float(x) for x in v],
            }
            for k, v in evals.items()
        },
        "skills": {
            str(sid): {
                "num_segments": s.num_segments,
                "mean_return": s.mean_return,
                "mean_utility": s.mean_utility,
                "mean_progress": s.mean_progress,
                "mean_energy": s.mean_energy,
            }
            for sid, s in action_set.skills.items()
        },
        "elite_episode_returns": [float(ep.total_return) for ep in action_set.elite_episodes],
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    env.close()


if __name__ == "__main__":
    main()
