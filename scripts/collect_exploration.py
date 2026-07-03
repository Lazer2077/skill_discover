"""Collect exploration rollouts from an Isaac Lab locomotion environment.

Usage (inside Isaac Lab):
    ./isaaclab.sh -p scripts/collect_exploration.py \
        --task Isaac-Ant-v0 --num_envs 256 --num_steps 50000 \
        --headless --output outputs/rollouts_ant.pkl

Add --list_envs to print registered Isaac tasks and exit (useful when task
names differ across Isaac Lab versions).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the package importable when running via `./isaaclab.sh -p scripts/...`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect exploration rollouts.")
    parser.add_argument("--task", type=str, default="Isaac-Ant-v0", help="Isaac Lab task name.")
    parser.add_argument("--num_envs", type=int, default=256, help="Parallel environments.")
    parser.add_argument("--num_steps", type=int, default=50000, help="Total env-steps to collect.")
    parser.add_argument("--headless", action="store_true", help="Run without rendering.")
    parser.add_argument("--output", type=str, default="outputs/rollouts.pkl", help="Output pickle path.")
    parser.add_argument("--config", type=str, default=None, help="Optional YAML config; CLI overrides it.")
    parser.add_argument("--policy", type=str, default=None, choices=["random", "latent"], help="Exploration policy.")
    parser.add_argument("--action_std", type=float, default=None)
    parser.add_argument("--action_smoothing", type=float, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--list_envs", action="store_true", help="List registered Isaac tasks and exit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # The simulation app must exist before ANY isaaclab/gym-task import.
    from skill_discovery.envs.isaac_env_wrapper import IsaacEnvWrapper, launch_app, list_locomotion_tasks

    sim_app = launch_app(headless=args.headless)

    try:
        import yaml

        from skill_discovery.exploration.latent_policy import LatentExplorationPolicy
        from skill_discovery.exploration.random_policy import RandomExplorationPolicy
        from skill_discovery.exploration.rollout_collector import RolloutCollector
        from skill_discovery.utils.buffers import save_pickle
        from skill_discovery.utils.logging import get_logger
        from skill_discovery.utils.math_utils import set_global_seed

        logger = get_logger("collect")

        if args.list_envs:
            for name in list_locomotion_tasks():
                print(name)
            return

        cfg = {}
        if args.config:
            with open(args.config) as f:
                cfg = yaml.safe_load(f) or {}
        exp_cfg = cfg.get("exploration", {})
        policy_name = args.policy or exp_cfg.get("policy", "random")
        action_std = args.action_std if args.action_std is not None else exp_cfg.get("action_std", 0.5)
        action_smoothing = (
            args.action_smoothing if args.action_smoothing is not None else exp_cfg.get("action_smoothing", 0.2)
        )

        set_global_seed(args.seed)
        env = IsaacEnvWrapper.create(args.task, num_envs=args.num_envs, device=args.device)

        policy_cls = LatentExplorationPolicy if policy_name == "latent" else RandomExplorationPolicy
        policy = policy_cls(
            num_envs=env.num_envs,
            action_dim=env.action_dim,
            action_std=action_std,
            action_smoothing=action_smoothing,
            seed=args.seed,
        )
        logger.info("Exploration policy: %s (std=%.2f, smoothing=%.2f)", policy_name, action_std, action_smoothing)

        collector = RolloutCollector(env, policy)
        trajectories = collector.collect(args.num_steps)

        save_pickle({"task": args.task, "seed": args.seed, "trajectories": trajectories}, args.output)
        logger.info("Saved %d trajectories to %s", len(trajectories), args.output)
        env.close()
    finally:
        sim_app.close()


if __name__ == "__main__":
    main()
