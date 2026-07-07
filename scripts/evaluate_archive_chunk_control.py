"""Evaluate closed-loop archive-chunk control in Isaac Lab.

Run through Isaac Lab:

    ./isaaclab.sh -p scripts/evaluate_archive_chunk_control.py \
      --task Isaac-Velocity-Flat-Unitree-Go2-v0 \
      --online_action_set outputs/isaac_current_go2_smoke/online_action_set.pkl \
      --targets "0.5,0;0.5,0.5;-0.5,0" --headless
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_targets(raw: str) -> List[List[float]]:
    targets: List[List[float]] = []
    for item in raw.split(";"):
        item = item.strip()
        if not item:
            continue
        x, y = item.split(",")
        targets.append([float(x), float(y)])
    if not targets:
        raise ValueError("At least one target is required.")
    return targets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate archive-chunk control.")
    parser.add_argument("--task", type=str, default="Isaac-Ant-v0")
    parser.add_argument("--online_action_set", type=str, required=True)
    parser.add_argument("--targets", type=str, default="0.5,0;0.5,0.5;-0.5,0")
    parser.add_argument("--num_trials", type=int, default=3)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--target_threshold", type=float, default=0.3)
    parser.add_argument("--max_high_level_steps", type=int, default=12)
    parser.add_argument("--execution_horizon", type=int, default=8)
    parser.add_argument("--lambda_state", type=float, default=0.25)
    parser.add_argument("--lambda_energy", type=float, default=0.05)
    parser.add_argument("--lambda_stability", type=float, default=0.5)
    parser.add_argument("--lambda_no_progress", type=float, default=1.5)
    parser.add_argument("--lambda_utility", type=float, default=0.1)
    parser.add_argument("--lambda_progress", type=float, default=0.25)
    parser.add_argument("--applicability_temperature", type=float, default=1.5)
    parser.add_argument("--k_nearest", type=int, default=0)
    parser.add_argument("--feature_zero_slices", type=str, default="")
    parser.add_argument("--relative_targets", action="store_true")
    parser.add_argument("--no_scale_motion_to_execution_horizon", action="store_true")
    parser.add_argument("--min_final_height_fraction", type=float, default=0.0)
    parser.add_argument("--random_action_std", type=float, default=0.5)
    parser.add_argument("--action_smoothing", type=float, default=0.2)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="outputs/archive_chunk_eval.json")
    return parser.parse_args()


def _distance_to_target(env: Any, target_xy: Any) -> float:
    import numpy as np

    pos = env.get_robot_state()["base_pos"][0]
    return float(np.linalg.norm(np.asarray(target_xy, dtype=np.float64) - pos[:2]))


def _energy_step(env: Any, action: Any) -> float:
    import numpy as np

    joint_vel = env.get_robot_state()["joint_vel"][0]
    action = np.asarray(action)
    min_dim = min(len(action), len(joint_vel))
    return float(np.mean(np.abs(action[:min_dim] * joint_vel[:min_dim])))


def _run_random_archive(
    env: Any,
    action_set: Any,
    target_xy: Any,
    args: argparse.Namespace,
    seed: int,
) -> Dict[str, Any]:
    import numpy as np

    rng = np.random.default_rng(seed)
    env.reset()
    start_height = float(env.get_robot_state()["base_pos"][0][2])
    if args.relative_targets:
        start_pos = env.get_robot_state()["base_pos"][0]
        target_xy = start_pos[:2] + np.asarray(target_xy, dtype=np.float64)
    archive = action_set._archive_segments
    selected: List[int] = []
    energy_total, energy_steps = 0.0, 0
    terminated_early = False

    for _ in range(args.max_high_level_steps):
        if _distance_to_target(env, target_xy) < args.target_threshold:
            break
        idx = int(rng.integers(len(archive)))
        selected.append(idx)
        actions = np.asarray(archive[idx]["action_seq"], dtype=np.float32)[: args.execution_horizon]
        for action in actions:
            result = env.step(np.tile(action, (env.num_envs, 1)))
            energy_total += _energy_step(env, action)
            energy_steps += 1
            if result.dones[0]:
                terminated_early = True
                break
        if terminated_early:
            break

    final_distance = _distance_to_target(env, target_xy)
    final_height = float(env.get_robot_state()["base_pos"][0][2])
    height_ok = final_height >= args.min_final_height_fraction * max(start_height, 1e-6)
    return {
        "success": (not terminated_early) and height_ok and final_distance < args.target_threshold,
        "final_distance": final_distance,
        "final_height": final_height,
        "energy_proxy": energy_total / max(energy_steps, 1),
        "num_chunks_used": len(selected),
        "archive_indices": selected,
        "terminated_early": terminated_early,
    }


def _run_low_level_baseline(
    env: Any,
    target_xy: Any,
    args: argparse.Namespace,
    mode: str,
    seed: int,
) -> Dict[str, Any]:
    import numpy as np

    from skill_discovery.exploration.random_policy import RandomExplorationPolicy

    obs = env.reset()
    start_height = float(env.get_robot_state()["base_pos"][0][2])
    if args.relative_targets:
        start_pos = env.get_robot_state()["base_pos"][0]
        target_xy = start_pos[:2] + np.asarray(target_xy, dtype=np.float64)
    total_steps = args.max_high_level_steps * args.execution_horizon
    energy_total, energy_steps = 0.0, 0
    terminated_early = False
    if mode == "random_action":
        policy = RandomExplorationPolicy(
            num_envs=env.num_envs,
            action_dim=env.action_dim,
            action_std=args.random_action_std,
            action_smoothing=args.action_smoothing,
            seed=seed,
        )
    elif mode == "zero_action":
        policy = None
    else:
        raise ValueError(f"Unknown low-level baseline: {mode}")

    for _ in range(total_steps):
        if _distance_to_target(env, target_xy) < args.target_threshold:
            break
        if policy is None:
            actions = np.zeros((env.num_envs, env.action_dim), dtype=np.float32)
        else:
            actions = policy.act(obs)
        result = env.step(actions)
        obs = result.obs
        energy_total += _energy_step(env, actions[0])
        energy_steps += 1
        if result.dones[0]:
            terminated_early = True
            break

    final_distance = _distance_to_target(env, target_xy)
    final_height = float(env.get_robot_state()["base_pos"][0][2])
    height_ok = final_height >= args.min_final_height_fraction * max(start_height, 1e-6)
    return {
        "success": (not terminated_early) and height_ok and final_distance < args.target_threshold,
        "final_distance": final_distance,
        "final_height": final_height,
        "energy_proxy": energy_total / max(energy_steps, 1),
        "num_chunks_used": 0,
        "archive_indices": [],
        "terminated_early": terminated_early,
    }


def _summarize(records: List[Dict[str, Any]]) -> Dict[str, float]:
    import numpy as np

    return {
        "success_rate": float(np.mean([r["success"] for r in records])),
        "average_final_distance": float(np.mean([r["final_distance"] for r in records])),
        "average_energy_proxy": float(np.mean([r["energy_proxy"] for r in records])),
        "average_num_chunks_used": float(np.mean([r["num_chunks_used"] for r in records])),
    }


def main() -> None:
    args = parse_args()

    from skill_discovery.envs.isaac_env_wrapper import IsaacEnvWrapper, launch_app

    sim_app = launch_app(headless=args.headless)
    try:
        import numpy as np

        from skill_discovery.control.archive_chunk_composer import ArchiveChunkComposer
        from skill_discovery.learning.skill_policy import parse_obs_slices
        from skill_discovery.online.online_action_set import OnlineActionSet
        from skill_discovery.utils.math_utils import set_global_seed

        set_global_seed(args.seed)
        targets = parse_targets(args.targets)
        action_set = OnlineActionSet.load(args.online_action_set)
        feature_zero_slices = parse_obs_slices(args.feature_zero_slices)
        env = IsaacEnvWrapper.create(args.task, num_envs=args.num_envs, device=args.device, seed=args.seed)

        composer = ArchiveChunkComposer(
            action_set=action_set,
            target_threshold=args.target_threshold,
            max_high_level_steps=args.max_high_level_steps,
            execution_horizon=args.execution_horizon,
            lambda_state=args.lambda_state,
            lambda_energy=args.lambda_energy,
            lambda_stability=args.lambda_stability,
            lambda_no_progress=args.lambda_no_progress,
            lambda_utility=args.lambda_utility,
            lambda_progress=args.lambda_progress,
            applicability_temperature=args.applicability_temperature,
            k_nearest=args.k_nearest,
            feature_zero_slices=feature_zero_slices,
            relative_target=args.relative_targets,
            scale_motion_to_execution_horizon=not args.no_scale_motion_to_execution_horizon,
            min_final_height_fraction=args.min_final_height_fraction,
        )

        methods = ["archive_chunk", "random_archive", "random_action", "zero_action"]
        all_records: Dict[str, Dict[str, List[Dict[str, Any]]]] = {
            f"{target[0]:g},{target[1]:g}": {method: [] for method in methods}
            for target in targets
        }

        for target_index, target in enumerate(targets):
            target_xy = np.asarray(target, dtype=np.float64)
            target_key = f"{target[0]:g},{target[1]:g}"
            for trial in range(args.num_trials):
                trial_seed = args.seed + 1000 * target_index + trial

                result = composer.rollout(env, target_xy)
                all_records[target_key]["archive_chunk"].append(
                    {
                        "trial": trial,
                        "success": bool(result.success),
                        "final_distance": float(result.final_distance),
                        "energy_proxy": float(result.energy_proxy),
                        "num_chunks_used": int(result.num_skills_used),
                        "archive_indices": [int(i) for i in result.skill_sequence],
                        "terminated_early": bool(result.terminated_early),
                        "final_height": float(result.final_height),
                        "min_height": float(result.min_height),
                        "decisions": [asdict(d) for d in composer.last_decisions],
                    }
                )
                all_records[target_key]["random_archive"].append(
                    _run_random_archive(env, action_set, target_xy, args, seed=trial_seed)
                )
                all_records[target_key]["random_action"].append(
                    _run_low_level_baseline(env, target_xy, args, "random_action", seed=trial_seed)
                )
                all_records[target_key]["zero_action"].append(
                    _run_low_level_baseline(env, target_xy, args, "zero_action", seed=trial_seed)
                )

        summary = {
            target_key: {method: _summarize(records) for method, records in method_records.items()}
            for target_key, method_records in all_records.items()
        }
        output = {
            "task": args.task,
            "online_action_set": args.online_action_set,
            "archive_size": composer.archive_size,
            "targets": targets,
            "num_trials": args.num_trials,
            "target_threshold": args.target_threshold,
            "max_high_level_steps": args.max_high_level_steps,
            "execution_horizon": args.execution_horizon,
            "controller": {
                "lambda_state": args.lambda_state,
                "lambda_energy": args.lambda_energy,
                "lambda_stability": args.lambda_stability,
                "lambda_no_progress": args.lambda_no_progress,
                "lambda_utility": args.lambda_utility,
                "lambda_progress": args.lambda_progress,
                "applicability_temperature": args.applicability_temperature,
                "k_nearest": args.k_nearest,
                "feature_zero_slices": feature_zero_slices,
                "relative_targets": args.relative_targets,
                "scale_motion_to_execution_horizon": not args.no_scale_motion_to_execution_horizon,
                "min_final_height_fraction": args.min_final_height_fraction,
            },
            "summary": summary,
            "records": all_records,
        }
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(output, indent=2))
        print(json.dumps({"summary": summary}, indent=2))
        env.close()
    finally:
        sim_app.close()


if __name__ == "__main__":
    main()
