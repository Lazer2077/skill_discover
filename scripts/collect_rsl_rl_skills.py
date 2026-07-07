"""Collect an online action set from an Isaac Lab RSL-RL policy.

This is the Isaac-side analogue of the Gym RL-first path: first use a trained
controller to generate competent locomotion rollouts, then compute the skill /
action archive from those rollouts.

Run through Isaac Lab:

    ./isaaclab.sh -p scripts/collect_rsl_rl_skills.py \
      --task Isaac-Velocity-Flat-Unitree-Go2-v0 \
      --use_pretrained_checkpoint --num_envs 64 --num_steps 32768 \
      --headless --output_dir outputs/go2_rsl_rl_skills
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect skills from an RSL-RL policy.")
    parser.add_argument("--task", type=str, default="Isaac-Velocity-Flat-Unitree-Go2-v0")
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--num_steps", type=int, default=32768, help="Total env-steps to collect.")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--use_pretrained_checkpoint", action="store_true")
    parser.add_argument("--segment_horizon", type=int, default=32)
    parser.add_argument("--segment_stride", type=int, default=8)
    parser.add_argument("--min_segment_length", type=int, default=16)
    parser.add_argument("--max_skills", type=int, default=16)
    parser.add_argument("--max_archive_size", type=int, default=5000)
    parser.add_argument("--discriminator_epochs", type=int, default=3)
    parser.add_argument("--outcome_epochs", type=int, default=3)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--output_dir", type=str, default="outputs/rsl_rl_skills")
    return parser.parse_args()


def _to_numpy(x: Any):
    import numpy as np
    import torch

    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _obs_for_buffer(obs: Any):
    """Convert RSL-RL TensorDict observations to a plain array for segmentation."""
    if hasattr(obs, "keys"):
        if "policy" in obs.keys():
            return _to_numpy(obs["policy"])
        first_key = next(iter(obs.keys()))
        return _to_numpy(obs[first_key])
    return _to_numpy(obs)


def _jsonable(obj: Any) -> Any:
    if hasattr(obj, "__dict__"):
        return {k: _jsonable(v) for k, v in obj.__dict__.items() if not k.startswith("_")}
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    try:
        import numpy as np

        if isinstance(obj, np.generic):
            return obj.item()
    except Exception:
        pass
    return obj


def main() -> None:
    args = parse_args()

    from skill_discovery.envs.isaac_env_wrapper import IsaacEnvWrapper, launch_app

    sim_app = launch_app(headless=args.headless)
    try:
        import gymnasium as gym
        import numpy as np
        import torch
        from rsl_rl.runners import DistillationRunner, OnPolicyRunner

        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
        from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
        from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

        import isaaclab_tasks  # noqa: F401

        from skill_discovery.descriptors.locomotion_descriptors import LocomotionDescriptorExtractor
        from skill_discovery.learning.state_skill_discriminator import StateSkillDiscriminator
        from skill_discovery.learning.skill_outcome_model import SkillOutcomeModel
        from skill_discovery.online.online_action_set import OnlineActionSet, OnlineActionSetConfig
        from skill_discovery.segmentation.fixed_horizon_segmenter import FixedHorizonSegmenter
        from skill_discovery.utils.buffers import TrajectoryBuffer, save_pickle
        from skill_discovery.utils.logging import get_logger
        from skill_discovery.utils.math_utils import set_global_seed

        logger = get_logger("rsl_rl_skills")
        set_global_seed(args.seed)
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        task_name = args.task.split(":")[-1]
        train_task_name = task_name.replace("-Play", "")
        env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
        agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
        agent_cfg.seed = args.seed
        agent_cfg.device = args.device
        env_cfg.seed = args.seed
        env_cfg.sim.device = args.device

        if args.use_pretrained_checkpoint:
            resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
            if resume_path is None:
                raise RuntimeError(f"No published RSL-RL checkpoint is available for {train_task_name}.")
        elif args.checkpoint:
            resume_path = args.checkpoint
        else:
            log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

        raw_env = gym.make(args.task, cfg=env_cfg)
        state_reader = IsaacEnvWrapper(raw_env, device=args.device)
        rl_env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)

        logger.info("Loading RSL-RL checkpoint: %s", resume_path)
        if agent_cfg.class_name == "OnPolicyRunner":
            runner = OnPolicyRunner(rl_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        elif agent_cfg.class_name == "DistillationRunner":
            runner = DistillationRunner(rl_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        else:
            raise ValueError(f"Unsupported RSL-RL runner class: {agent_cfg.class_name}")
        runner.load(resume_path)
        policy = runner.get_inference_policy(device=rl_env.unwrapped.device)
        policy_nn = getattr(runner.alg, "policy", getattr(runner.alg, "actor_critic", None))

        sim_steps = int(np.ceil(args.num_steps / args.num_envs))
        buffer = TrajectoryBuffer(args.num_envs)
        obs = rl_env.get_observations()
        logger.info("Collecting %d env-steps from RSL-RL policy (%d sim steps).", args.num_steps, sim_steps)
        for _ in range(sim_steps):
            with torch.inference_mode():
                actions = policy(obs)
                if agent_cfg.clip_actions is not None:
                    actions = torch.clamp(actions, -agent_cfg.clip_actions, agent_cfg.clip_actions)
            state = state_reader.get_robot_state()
            next_obs, rewards, dones, _ = rl_env.step(actions)
            buffer.add(
                {
                    "obs": _obs_for_buffer(obs),
                    "actions": _to_numpy(actions),
                    "rewards": _to_numpy(rewards).reshape(args.num_envs),
                    "dones": _to_numpy(dones).astype(bool).reshape(args.num_envs),
                    **state,
                }
            )
            obs = next_obs
            if policy_nn is not None and hasattr(policy_nn, "reset"):
                policy_nn.reset(dones)

        buffer.flush_all()
        trajectories = buffer.trajectories
        save_pickle({"task": args.task, "seed": args.seed, "trajectories": trajectories}, out_dir / "rollouts.pkl")
        logger.info("Collected %d trajectories, %d total steps.", buffer.num_trajectories, buffer.total_steps)

        segmenter = FixedHorizonSegmenter(
            segment_horizon=args.segment_horizon,
            segment_stride=args.segment_stride,
            min_segment_length=args.min_segment_length,
        )
        segments = segmenter.segment_all(trajectories)
        extractor = LocomotionDescriptorExtractor()
        descriptor_matrix, _ = extractor.compute_matrix(segments)
        save_pickle({"segments": segments, "descriptors": descriptor_matrix}, out_dir / "segments_descriptors.pkl")

        action_set = OnlineActionSet(
            OnlineActionSetConfig(max_skills=args.max_skills, max_archive_size=args.max_archive_size)
        )
        action_set.metadata = {
            "task": args.task,
            "seed": args.seed,
            "source_policy": "rsl_rl",
            "checkpoint": str(resume_path),
            "num_steps": args.num_steps,
        }
        update_stats = action_set.update(segments, descriptor_matrix, iteration=0)
        action_set.save(str(out_dir / "online_action_set.pkl"))
        action_set.to_skill_library().save(str(out_dir / "skill_library_v2.pkl"))

        discriminator = StateSkillDiscriminator(device="cpu", seed=args.seed)
        disc_stats = None
        if action_set.state_feature_dim is not None and action_set.skills:
            disc_stats = discriminator.fit(action_set, epochs=args.discriminator_epochs)
            discriminator.save(out_dir / "state_skill_discriminator.pt")

        outcome_model = SkillOutcomeModel(device="cpu", seed=args.seed)
        outcome_stats = None
        if action_set.skills:
            outcome_stats = outcome_model.fit(action_set, epochs=args.outcome_epochs)
            outcome_model.save(out_dir / "skill_outcome_model.pt")

        summary: Dict[str, Any] = {
            "task": args.task,
            "source_policy": "rsl_rl",
            "checkpoint": str(resume_path),
            "num_steps": args.num_steps,
            "num_trajectories": len(trajectories),
            "num_segments": len(segments),
            "num_skills": len(action_set.skills),
            "archive_size": len(action_set._archive_segments),
            "update": _jsonable(update_stats),
            "discriminator": _jsonable(disc_stats),
            "outcome_model": _jsonable(outcome_stats),
            "skill_summary": action_set.summary(),
        }
        (out_dir / "rsl_rl_skill_summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))
        rl_env.close()
    finally:
        sim_app.close()


if __name__ == "__main__":
    main()
