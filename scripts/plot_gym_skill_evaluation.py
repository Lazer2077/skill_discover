"""Plot Gymnasium skill/action-set evaluation from random initial states.

This script is intentionally focused on the RL-first workflow:

1. load a converged SB3 policy,
2. collect rollouts from that policy,
3. build a state-conditioned action archive,
4. compare nearest-chunk skill control against the original RL policy.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

from scripts.gym_online_skill_control import (
    GymActionSet,
    collect_chunks_from_rl_policy,
    make_gym_env,
)


def load_sb3_model(algo: str, path: str, device: str) -> Any:
    from stable_baselines3 import PPO, SAC, TD3

    algos = {"SAC": SAC, "TD3": TD3, "PPO": PPO}
    key = algo.upper()
    if key not in algos:
        raise ValueError(f"Unsupported algo {algo!r}. Choose from {sorted(algos)}.")
    return algos[key].load(path, device=device)


def build_eval_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        env=args.env,
        max_episode_steps=args.max_episode_steps,
        chunk_horizon=args.chunk_horizon,
        max_skills=args.max_skills,
        max_archive_size=args.max_archive_size,
        utility_progress_weight=args.utility_progress_weight,
        utility_state_weight=args.utility_state_weight,
        elite_min_return=args.elite_min_return,
        rl_collect_episodes=args.rl_collect_episodes,
        rl_action_noise=args.rl_action_noise,
        rl_stochastic_collect=args.rl_stochastic_collect,
        seed=args.seed,
    )


def action_set_policy(action_set: GymActionSet, utility_weight: float) -> Callable[[np.ndarray], np.ndarray]:
    starts = np.stack([np.asarray(c["start_obs"], dtype=np.float32) for c in action_set.archive])
    actions = [np.asarray(c["actions"], dtype=np.float32) for c in action_set.archive]
    utilities = np.asarray([float(c["utility"]) for c in action_set.archive], dtype=np.float32)
    utilities = (utilities - float(utilities.mean())) / float(utilities.std() + 1e-6)
    scale = starts.std(axis=0, keepdims=True) + 0.25
    pending: List[np.ndarray] = []

    def policy(obs: np.ndarray) -> np.ndarray:
        if pending:
            return pending.pop(0)
        dists = np.linalg.norm((starts - obs.reshape(1, -1)) / scale, axis=-1) / np.sqrt(starts.shape[-1])
        idx = int(np.argmax(-dists + utility_weight * utilities))
        seq = actions[idx]
        pending.extend([a.copy() for a in seq[1:]])
        return seq[0].copy()

    return policy


def rollout(
    env_id: str,
    max_steps: int,
    seed: int,
    policy: Callable[[np.ndarray], np.ndarray],
) -> Dict[str, Any]:
    env = gym.make(env_id, max_episode_steps=max_steps)
    obs, _ = env.reset(seed=seed)
    obs_list = [np.asarray(obs, dtype=np.float32)]
    action_list: List[np.ndarray] = []
    reward_list: List[float] = []
    done = False
    for _ in range(max_steps):
        action = np.asarray(policy(obs), dtype=np.float32)
        action = np.clip(action, env.action_space.low, env.action_space.high)
        obs, reward, terminated, truncated, _ = env.step(action)
        obs_list.append(np.asarray(obs, dtype=np.float32))
        action_list.append(np.asarray(action, dtype=np.float32))
        reward_list.append(float(reward))
        done = terminated or truncated
        if done:
            break
    env.close()
    obs_arr = np.asarray(obs_list, dtype=np.float32)
    action_arr = np.asarray(action_list, dtype=np.float32)
    reward_arr = np.asarray(reward_list, dtype=np.float32)
    action_2d = action_arr.reshape(len(action_arr), -1) if len(action_arr) else np.zeros((0, 1), dtype=np.float32)
    return {
        "seed": seed,
        "return": float(np.sum(reward_arr)),
        "done": done,
        "obs": obs_arr,
        "actions": action_2d,
        "rewards": reward_arr,
        "cum_reward": np.cumsum(reward_arr),
    }


def env_metrics(env_id: str, sample_obs: np.ndarray) -> List[tuple[str, Callable[[np.ndarray], np.ndarray]]]:
    if env_id.startswith("Pendulum"):
        return [
            ("angle", lambda obs: np.arctan2(obs[:, 1], obs[:, 0])),
            ("angular velocity", lambda obs: obs[:, 2]),
        ]
    if env_id.startswith("MountainCarContinuous"):
        return [
            ("position", lambda obs: obs[:, 0]),
            ("velocity", lambda obs: obs[:, 1]),
        ]
    if env_id.startswith("LunarLander"):
        return [
            ("x position", lambda obs: obs[:, 0]),
            ("y position", lambda obs: obs[:, 1]),
            ("angle", lambda obs: obs[:, 4]),
        ]
    names = []
    for idx in range(min(3, sample_obs.shape[1])):
        names.append((f"state[{idx}]", lambda obs, i=idx: obs[:, i]))
    return names


def plot_results(results: Dict[str, List[Dict[str, Any]]], env_id: str, output: Path) -> None:
    controllers = list(results)
    num_states = len(next(iter(results.values())))
    sample_obs = next(iter(results.values()))[0]["obs"]
    metrics = env_metrics(env_id, sample_obs)
    num_cols = len(metrics) + 2
    fig, axes = plt.subplots(num_states, num_cols, figsize=(4.2 * num_cols, 3.8 * num_states), squeeze=False)
    colors = {"skill_nearest": "#1f77b4", "rl_policy": "#2ca02c", "random": "#d62728"}

    for row in range(num_states):
        seed = results[controllers[0]][row]["seed"]
        axes[row, 0].set_ylabel(f"seed {seed}")
        for name in controllers:
            r = results[name][row]
            t_obs = np.arange(len(r["obs"]))
            t_act = np.arange(len(r["actions"]))
            label = f"{name} ({r['return']:.1f})"
            color = colors.get(name)
            for col, (_metric_name, metric_fn) in enumerate(metrics):
                axes[row, col].plot(t_obs, metric_fn(r["obs"]), label=label, color=color)
            axes[row, len(metrics)].plot(t_act, r["cum_reward"], label=label, color=color)
            if r["actions"].shape[1] == 1:
                axes[row, len(metrics) + 1].plot(t_act, r["actions"][:, 0], label=label, color=color, alpha=0.9)
            else:
                axes[row, len(metrics) + 1].plot(t_act, r["actions"][:, 0], label=f"{label} a0", color=color, alpha=0.9)
                axes[row, len(metrics) + 1].plot(t_act, r["actions"][:, 1], linestyle="--", label=f"{label} a1", color=color, alpha=0.7)
        for col, (metric_name, _metric_fn) in enumerate(metrics):
            axes[row, col].axhline(0.0, color="black", linewidth=0.8, alpha=0.35)
            axes[row, col].set_title(metric_name)
        axes[row, len(metrics)].set_title("cumulative reward")
        axes[row, len(metrics) + 1].set_title("action")
        for col in range(num_cols):
            axes[row, col].grid(alpha=0.25)
            axes[row, col].set_xlabel("step")
        axes[row, 0].legend(loc="lower right", fontsize=8)

    fig.suptitle("Skill/action-set control from random initial states", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot Gym skill-control evaluation.")
    p.add_argument("--env", type=str, default="Pendulum-v1")
    p.add_argument("--model_path", type=str, default="outputs/gym_online_skill_control/pendulum_rl_converged_summary.sac.zip")
    p.add_argument("--rl_algo", type=str, default="SAC")
    p.add_argument("--max_episode_steps", type=int, default=200)
    p.add_argument("--chunk_horizon", type=int, default=1)
    p.add_argument("--max_archive_size", type=int, default=20000)
    p.add_argument("--max_skills", type=int, default=16)
    p.add_argument("--rl_collect_episodes", type=int, default=100)
    p.add_argument("--utility_progress_weight", type=float, default=10.0)
    p.add_argument("--utility_state_weight", type=float, default=0.5)
    p.add_argument("--elite_min_return", type=float, default=50.0)
    p.add_argument("--rl_action_noise", type=float, default=0.0)
    p.add_argument("--rl_stochastic_collect", action="store_true")
    p.add_argument("--nn_utility_weight", type=float, default=0.0)
    p.add_argument("--num_initial_states", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--output", type=str, default="outputs/gym_online_skill_control/skill_eval_3_states.png")
    p.add_argument("--summary_output", type=str, default="outputs/gym_online_skill_control/skill_eval_3_states.json")
    p.add_argument("--include_random", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    eval_args = build_eval_args(args)
    model = load_sb3_model(args.rl_algo, args.model_path, args.device)

    collect_env = make_gym_env(args.env, eval_args)
    action_set = GymActionSet(max_skills=args.max_skills, max_archive_size=args.max_archive_size)
    chunks, elites = collect_chunks_from_rl_policy(collect_env, model, eval_args, rng)
    stats = action_set.update(chunks, elites)
    collect_env.close()

    seeds = [args.seed + 70000 + i for i in range(args.num_initial_states)]
    skill_policy = action_set_policy(action_set, args.nn_utility_weight)
    results: Dict[str, List[Dict[str, Any]]] = {"skill_nearest": [], "rl_policy": []}
    if args.include_random:
        results["random"] = []

    for seed in seeds:
        results["skill_nearest"].append(rollout(args.env, args.max_episode_steps, seed, skill_policy))
        results["rl_policy"].append(
            rollout(
                args.env,
                args.max_episode_steps,
                seed,
                lambda obs, m=model: m.predict(obs, deterministic=True)[0],
            )
        )
        if args.include_random:
            local_rng = np.random.default_rng(seed)
            tmp_env = gym.make(args.env)
            low, high = tmp_env.action_space.low, tmp_env.action_space.high
            tmp_env.close()
            results["random"].append(
                rollout(
                    args.env,
                    args.max_episode_steps,
                    seed,
                    lambda _obs, r=local_rng, lo=low, hi=high: r.uniform(lo, hi).astype(np.float32),
                )
            )

    output = Path(args.output)
    plot_results(results, args.env, output)

    summary = {
        "env": args.env,
        "model_path": args.model_path,
        "action_set_stats": stats,
        "chunk_horizon": args.chunk_horizon,
        "num_initial_states": args.num_initial_states,
        "seeds": seeds,
        "returns": {
            name: [float(r["return"]) for r in runs]
            for name, runs in results.items()
        },
        "mean_returns": {
            name: float(np.mean([r["return"] for r in runs]))
            for name, runs in results.items()
        },
        "plot": str(output),
    }
    summary_path = Path(args.summary_output)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
