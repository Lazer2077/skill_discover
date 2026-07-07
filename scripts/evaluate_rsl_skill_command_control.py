"""Evaluate skill-selected velocity commands with a pretrained RSL-RL policy.

This is the safer locomotion variant for humanoids: the discovered action set
is used as a high-level library of local motion commands, while the pretrained
RSL-RL policy remains the low-level stabilizing controller.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_targets(raw: str) -> List[List[float]]:
    targets: List[List[float]] = []
    for item in raw.split(";"):
        item = item.strip()
        if item:
            x, y = item.split(",")
            targets.append([float(x), float(y)])
    if not targets:
        raise ValueError("At least one target is required.")
    return targets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate RSL-RL low-level control with skill-selected commands.")
    parser.add_argument("--task", type=str, default="Isaac-Velocity-Rough-H1-v0")
    parser.add_argument("--online_action_set", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--use_pretrained_checkpoint", action="store_true")
    parser.add_argument("--targets", type=str, default="0.5,0;0.5,0.5;1.0,0")
    parser.add_argument("--num_trials", type=int, default=5)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument(
        "--methods",
        type=str,
        default="skill_command,direct_target_command,random_command,zero_command",
        help=(
            "Comma-separated methods to evaluate. Valid methods: skill_command, "
            "guarded_skill_command, learned_guarded_skill_command, direct_target_command, "
            "random_command, zero_command."
        ),
    )
    parser.add_argument("--target_threshold", type=float, default=0.3)
    parser.add_argument("--max_high_level_steps", type=int, default=48)
    parser.add_argument("--execution_horizon", type=int, default=4)
    parser.add_argument("--lambda_state", type=float, default=2.0)
    parser.add_argument("--lambda_energy", type=float, default=0.05)
    parser.add_argument("--lambda_stability", type=float, default=0.5)
    parser.add_argument("--lambda_no_progress", type=float, default=1.5)
    parser.add_argument("--lambda_utility", type=float, default=0.1)
    parser.add_argument("--lambda_progress", type=float, default=0.25)
    parser.add_argument("--applicability_temperature", type=float, default=1.5)
    parser.add_argument("--k_nearest", type=int, default=16)
    parser.add_argument("--feature_zero_slices", type=str, default="9:12")
    parser.add_argument("--command_slice", type=str, default="9:12")
    parser.add_argument("--command_gain", type=float, default=1.0)
    parser.add_argument("--command_max", type=float, default=1.0)
    parser.add_argument("--yaw_command_gain", type=float, default=1.0)
    parser.add_argument(
        "--skill_target_command_blend",
        type=float,
        default=1.0,
        help="1.0 uses pure selected skill command; 0.0 uses pure direct target command.",
    )
    parser.add_argument("--guard_min_alignment", type=float, default=0.35)
    parser.add_argument("--guard_min_progress", type=float, default=0.0)
    parser.add_argument("--guard_min_stability", type=float, default=0.7)
    parser.add_argument("--guard_max_energy_z", type=float, default=0.75)
    parser.add_argument("--guard_max_skill_blend", type=float, default=0.5)
    parser.add_argument("--guard_blend_distance", type=float, default=0.75)
    parser.add_argument("--guard_max_speed_gain", type=float, default=0.25)
    parser.add_argument(
        "--guard_low_height_fraction",
        type=float,
        default=0.0,
        help="If >0, guarded control disables skill residuals below this fraction of the rollout start height.",
    )
    parser.add_argument("--guard_low_height_command_scale", type=float, default=0.5)
    parser.add_argument("--viability_model", type=str, default=None)
    parser.add_argument("--viability_threshold", type=float, default=0.5)
    parser.add_argument(
        "--viability_filter_mode",
        type=str,
        default="intersect",
        choices=["intersect", "replace", "union"],
        help="How learned viability combines with the heuristic guard.",
    )
    parser.add_argument(
        "--guard_recovery_height_fraction",
        type=float,
        default=0.0,
        help="If >0, guarded control keeps issuing recovery commands inside the target radius until this height fraction is restored.",
    )
    parser.add_argument("--guard_recovery_command_scale", type=float, default=0.0)
    parser.add_argument("--random_command_std", type=float, default=0.5)
    parser.add_argument("--min_final_height_fraction", type=float, default=0.7)
    parser.add_argument("--relative_targets", action="store_true")
    parser.add_argument("--store_positions", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="outputs/rsl_skill_command_eval.json")
    return parser.parse_args()


def _to_numpy(x: Any):
    import numpy as np
    import torch

    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _obs_for_feature(obs: Any):
    if hasattr(obs, "keys"):
        if "policy" in obs.keys():
            return _to_numpy(obs["policy"])
        first_key = next(iter(obs.keys()))
        return _to_numpy(obs[first_key])
    return _to_numpy(obs)


def _clone_obs_with_command(obs: Any, command: Any, command_slice: tuple[int, int]) -> Any:
    import torch

    start, end = command_slice
    command_t = torch.as_tensor(command, dtype=torch.float32)
    if hasattr(obs, "keys"):
        out = obs.clone()
        key = "policy" if "policy" in obs.keys() else next(iter(obs.keys()))
        policy_obs = out[key].clone()
        command_t = command_t.to(device=policy_obs.device)
        policy_obs[:, start : min(end, policy_obs.shape[-1])] = command_t[:, : max(0, min(end, policy_obs.shape[-1]) - start)]
        out[key] = policy_obs
        return out
    out = obs.clone()
    command_t = command_t.to(device=out.device)
    out[:, start : min(end, out.shape[-1])] = command_t[:, : max(0, min(end, out.shape[-1]) - start)]
    return out


def _target_command(
    local_target: Any,
    command_dim: int,
    command_gain: float,
    command_max: float,
    yaw_command_gain: float,
):
    import numpy as np

    command = np.zeros(command_dim, dtype=np.float32)
    if command_dim >= 2:
        command[:2] = np.clip(np.asarray(local_target[:2]) * command_gain, -command_max, command_max)
    elif command_dim == 1:
        command[0] = float(np.clip(local_target[0] * command_gain, -command_max, command_max))
    if command_dim >= 3:
        heading_error = float(np.arctan2(local_target[1], max(local_target[0], 1e-6)))
        command[2] = float(np.clip(heading_error * yaw_command_gain, -command_max, command_max))
    return command


def _summarize(records: List[Dict[str, Any]]) -> Dict[str, float]:
    import numpy as np

    return {
        "success_rate": float(np.mean([r["success"] for r in records])),
        "average_final_distance": float(np.mean([r["final_distance"] for r in records])),
        "average_energy_proxy": float(np.mean([r["energy_proxy"] for r in records])),
        "average_num_commands_used": float(np.mean([r["num_commands_used"] for r in records])),
        "termination_rate": float(np.mean([r["terminated_early"] for r in records])),
        "average_final_height": float(np.mean([r["final_height"] for r in records])),
    }


def _float_feature(value: Any, default: float = 0.0) -> float:
    import numpy as np

    try:
        if value is None:
            return default
        out = float(value)
        if not np.isfinite(out):
            return default
        return out
    except Exception:
        return default


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

        from skill_discovery.control.archive_chunk_composer import ArchiveChunkComposer
        from skill_discovery.descriptors.locomotion_descriptors import DESCRIPTOR_NAMES
        from skill_discovery.learning.skill_policy import parse_obs_slices
        from skill_discovery.learning.state_features import env_state_feature
        from skill_discovery.online.online_action_set import OnlineActionSet
        from skill_discovery.utils.math_utils import quat_to_yaw, set_global_seed, world_to_body_2d

        set_global_seed(args.seed)
        targets = parse_targets(args.targets)
        methods = [item.strip() for item in args.methods.split(",") if item.strip()]
        valid_methods = {
            "skill_command",
            "guarded_skill_command",
            "learned_guarded_skill_command",
            "direct_target_command",
            "random_command",
            "zero_command",
        }
        unknown_methods = sorted(set(methods) - valid_methods)
        if unknown_methods:
            raise ValueError(f"Unknown methods requested: {unknown_methods}. Valid methods: {sorted(valid_methods)}")
        if not methods:
            raise ValueError("--methods must include at least one method.")
        if "learned_guarded_skill_command" in methods and not args.viability_model:
            raise ValueError("--viability_model is required for learned_guarded_skill_command.")
        action_set = OnlineActionSet.load(args.online_action_set)
        command_slices = parse_obs_slices(args.command_slice)
        if not command_slices:
            raise ValueError("--command_slice is required, for example '9:12'.")
        command_slice = command_slices[0]
        command_dim = max(command_slice[1] - command_slice[0], 0)
        feature_zero_slices = parse_obs_slices(args.feature_zero_slices)

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

        if agent_cfg.class_name == "OnPolicyRunner":
            runner = OnPolicyRunner(rl_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        elif agent_cfg.class_name == "DistillationRunner":
            runner = DistillationRunner(rl_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        else:
            raise ValueError(f"Unsupported RSL-RL runner class: {agent_cfg.class_name}")
        runner.load(resume_path)
        policy = runner.get_inference_policy(device=rl_env.unwrapped.device)
        policy_nn = getattr(runner.alg, "policy", getattr(runner.alg, "actor_critic", None))

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
            relative_target=False,
            scale_motion_to_execution_horizon=True,
            min_final_height_fraction=args.min_final_height_fraction,
        )
        idx = {name: i for i, name in enumerate(DESCRIPTOR_NAMES)}
        archive_energy = composer._descriptors[:, idx["energy_proxy"]]
        archive_energy_mean = float(np.mean(archive_energy))
        archive_energy_std = float(max(np.std(archive_energy), 1e-6))
        viability_model = None
        viability_feature_names: List[str] = []
        viability_mean = None
        viability_std = None
        if args.viability_model:
            try:
                viability_data = torch.load(args.viability_model, map_location="cpu", weights_only=False)
            except TypeError:
                viability_data = torch.load(args.viability_model, map_location="cpu")
            viability_feature_names = list(viability_data["feature_names"])
            viability_model = torch.nn.Sequential(
                torch.nn.Linear(len(viability_feature_names), int(viability_data["hidden_dim"])),
                torch.nn.ReLU(),
                torch.nn.Linear(int(viability_data["hidden_dim"]), int(viability_data["hidden_dim"])),
                torch.nn.ReLU(),
                torch.nn.Linear(int(viability_data["hidden_dim"]), 1),
            )
            viability_model.load_state_dict(viability_data["state_dict"])
            viability_model.eval()
            viability_mean = np.asarray(viability_data["mean"], dtype=np.float32)
            viability_std = np.asarray(viability_data["std"], dtype=np.float32)

        def predict_viability(feature_values: Dict[str, float]) -> float:
            if viability_model is None or viability_mean is None or viability_std is None:
                return 1.0
            values = np.asarray(
                [_float_feature(feature_values.get(name, 0.0)) for name in viability_feature_names],
                dtype=np.float32,
            ).reshape(1, -1)
            values = ((values - viability_mean) / viability_std).astype(np.float32)
            with torch.inference_mode():
                logits = viability_model(torch.from_numpy(values)).squeeze()
                return float(torch.sigmoid(logits).item())

        rng = np.random.default_rng(args.seed)

        def rollout(method: str, target_offset: np.ndarray, trial_seed: int) -> Dict[str, Any]:
            nonlocal rng
            obs, _ = rl_env.reset()
            if policy_nn is not None and hasattr(policy_nn, "reset"):
                try:
                    policy_nn.reset(torch.ones(args.num_envs, dtype=torch.bool, device=rl_env.unwrapped.device))
                except TypeError:
                    policy_nn.reset()

            start_state = state_reader.get_robot_state()
            start_pos = start_state["base_pos"][0]
            start_height = float(start_pos[2])
            target = np.asarray(target_offset, dtype=np.float64)
            if args.relative_targets:
                target = start_pos[:2].astype(np.float64) + target

            selected: List[int] = []
            decisions: List[Dict[str, Any]] = []
            positions: List[List[float]] = []
            energy_total, energy_steps = 0.0, 0
            terminated_early = False
            rng = np.random.default_rng(trial_seed)

            for _ in range(args.max_high_level_steps):
                state = state_reader.get_robot_state()
                pos = state["base_pos"][0]
                yaw = float(quat_to_yaw(state["base_quat"][0]))
                positions.append(pos.tolist())
                distance = float(np.linalg.norm(target - pos[:2]))
                recovery_height = args.guard_recovery_height_fraction * max(start_height, 1e-6)
                needs_recovery = (
                    method in {"guarded_skill_command", "learned_guarded_skill_command"}
                    and args.guard_recovery_height_fraction > 0.0
                    and distance < args.target_threshold
                    and float(pos[2]) < recovery_height
                )
                if distance < args.target_threshold and not needs_recovery:
                    break

                if needs_recovery:
                    local_target = world_to_body_2d(target - pos[:2], yaw)
                    target_command = _target_command(
                        local_target,
                        command_dim,
                        args.command_gain,
                        args.command_max,
                        args.yaw_command_gain,
                    )
                    command = float(np.clip(args.guard_recovery_command_scale, 0.0, 1.0)) * target_command
                    decisions.append(
                        {
                            "recovery": True,
                            "current_distance": distance,
                            "current_height": float(pos[2]),
                            "target_recovery_height": recovery_height,
                        }
                    )
                elif method in {"skill_command", "guarded_skill_command", "learned_guarded_skill_command"}:
                    local_target = world_to_body_2d(target - pos[:2], yaw)
                    target_command = _target_command(
                        local_target,
                        command_dim,
                        args.command_gain,
                        args.command_max,
                        args.yaw_command_gain,
                    )
                    obs_np = _obs_for_feature(obs)
                    feature = env_state_feature(
                        obs_np,
                        state,
                        env_id=0,
                        include_obs=action_set.config.include_obs_in_state_feature,
                        max_obs_dim=action_set.config.max_obs_dim,
                    )
                    decision = composer.select_chunk((float(pos[0]), float(pos[1]), yaw), target, feature)
                    desc = composer._descriptors[decision.archive_index]
                    skill_command = np.zeros(command_dim, dtype=np.float32)
                    if command_dim >= 1:
                        skill_command[0] = float(desc[idx["average_forward_velocity"]])
                    if command_dim >= 2:
                        skill_command[1] = float(desc[idx["average_lateral_velocity"]])
                    if command_dim >= 3:
                        skill_command[2] = float(desc[idx["average_yaw_rate"]])
                    skill_command = np.clip(skill_command, -args.command_max, args.command_max)
                    blend = float(np.clip(args.skill_target_command_blend, 0.0, 1.0))
                    guard_info: Dict[str, Any] = {"accepted": True, "blend": blend}
                    if method in {"guarded_skill_command", "learned_guarded_skill_command"}:
                        target_xy = np.asarray(local_target[:2], dtype=np.float64)
                        skill_xy = np.asarray(skill_command[: min(2, command_dim)], dtype=np.float64)
                        target_norm = float(np.linalg.norm(target_xy))
                        skill_norm = float(np.linalg.norm(skill_xy))
                        if target_norm > 1e-6 and skill_norm > 1e-6:
                            alignment = float(np.dot(skill_xy, target_xy / target_norm) / skill_norm)
                        else:
                            alignment = 0.0
                        energy_z = float((decision.energy - archive_energy_mean) / archive_energy_std)
                        accepted = (
                            alignment >= args.guard_min_alignment
                            and decision.predicted_progress >= args.guard_min_progress
                            and decision.stability >= args.guard_min_stability
                            and energy_z <= args.guard_max_energy_z
                        )
                        viability_prob = None
                        if method == "learned_guarded_skill_command":
                            heuristic_accepted = bool(accepted)
                            feature_values = {
                                "current_distance": decision.current_distance,
                                "predicted_distance": decision.predicted_distance,
                                "predicted_progress": decision.predicted_progress,
                                "state_distance": decision.state_distance,
                                "applicability": decision.applicability,
                                "utility": decision.utility,
                                "energy": decision.energy,
                                "stability": decision.stability,
                                "guard_alignment": alignment,
                                "guard_energy_z": energy_z,
                                "guard_target_norm": target_norm,
                                "guard_skill_norm": skill_norm,
                                "guard_blend": blend,
                                "guard_accepted": float(accepted),
                                "guard_low_height": 0.0,
                            }
                            viability_prob = predict_viability(feature_values)
                            model_accepted = viability_prob >= float(np.clip(args.viability_threshold, 0.0, 1.0))
                            if args.viability_filter_mode == "replace":
                                accepted = model_accepted
                            elif args.viability_filter_mode == "union":
                                accepted = heuristic_accepted or model_accepted
                            else:
                                accepted = heuristic_accepted and model_accepted
                        low_height_threshold = args.guard_low_height_fraction * max(start_height, 1e-6)
                        low_height = args.guard_low_height_fraction > 0.0 and float(pos[2]) < low_height_threshold
                        if command_dim >= 2 and skill_norm > 1e-6:
                            target_command_norm = float(np.linalg.norm(target_command[:2]))
                            max_skill_norm = max(
                                target_command_norm * (1.0 + max(args.guard_max_speed_gain, 0.0)),
                                1e-6,
                            )
                            if skill_norm > max_skill_norm:
                                skill_command[:2] *= max_skill_norm / skill_norm
                                skill_norm = max_skill_norm
                        distance_blend = float(np.clip(distance / max(args.guard_blend_distance, 1e-6), 0.0, 1.0))
                        blend = float(np.clip(args.guard_max_skill_blend, 0.0, 1.0)) * distance_blend
                        if low_height:
                            target_command = (
                                float(np.clip(args.guard_low_height_command_scale, 0.0, 1.0)) * target_command
                            )
                            accepted = False
                            if method == "learned_guarded_skill_command" and viability_prob is None:
                                feature_values = {
                                    "current_distance": decision.current_distance,
                                    "predicted_distance": decision.predicted_distance,
                                    "predicted_progress": decision.predicted_progress,
                                    "state_distance": decision.state_distance,
                                    "applicability": decision.applicability,
                                    "utility": decision.utility,
                                    "energy": decision.energy,
                                    "stability": decision.stability,
                                    "guard_alignment": alignment,
                                    "guard_energy_z": energy_z,
                                    "guard_target_norm": target_norm,
                                    "guard_skill_norm": skill_norm,
                                    "guard_blend": blend,
                                    "guard_accepted": 0.0,
                                    "guard_low_height": 1.0,
                                }
                                viability_prob = predict_viability(feature_values)
                        if not accepted:
                            blend = 0.0
                        guard_info = {
                            "accepted": bool(accepted),
                            "blend": blend,
                            "alignment": alignment,
                            "energy_z": energy_z,
                            "target_norm": target_norm,
                            "skill_norm": skill_norm,
                            "low_height": bool(low_height),
                            "low_height_threshold": low_height_threshold,
                        }
                        if viability_prob is not None:
                            guard_info["viability_prob"] = viability_prob
                            guard_info["viability_threshold"] = float(np.clip(args.viability_threshold, 0.0, 1.0))
                            guard_info["viability_filter_mode"] = args.viability_filter_mode
                    command = blend * skill_command + (1.0 - blend) * target_command
                    command = np.clip(command, -args.command_max, args.command_max)
                    if blend > 0.0:
                        selected.append(int(decision.archive_index))
                    decision_record = asdict(decision)
                    decision_record["guard"] = guard_info
                    decisions.append(decision_record)
                elif method == "direct_target_command":
                    local_target = world_to_body_2d(target - pos[:2], yaw)
                    command = _target_command(
                        local_target,
                        command_dim,
                        args.command_gain,
                        args.command_max,
                        args.yaw_command_gain,
                    )
                elif method == "random_command":
                    command = rng.normal(0.0, args.random_command_std, size=command_dim).astype(np.float32)
                    command = np.clip(command, -args.command_max, args.command_max)
                elif method == "zero_command":
                    command = np.zeros(command_dim, dtype=np.float32)
                else:
                    raise ValueError(f"Unknown method: {method}")

                command_batch = np.tile(command, (args.num_envs, 1))
                for _low in range(args.execution_horizon):
                    with torch.inference_mode():
                        policy_obs = _clone_obs_with_command(obs, command_batch, command_slice)
                        actions = policy(policy_obs)
                        if agent_cfg.clip_actions is not None:
                            actions = torch.clamp(actions, -agent_cfg.clip_actions, agent_cfg.clip_actions)
                    obs, _rewards, dones, _extras = rl_env.step(actions)
                    action_np = _to_numpy(actions)[0]
                    joint_vel = state_reader.get_robot_state()["joint_vel"][0]
                    min_dim = min(len(action_np), len(joint_vel))
                    energy_total += float(np.mean(np.abs(action_np[:min_dim] * joint_vel[:min_dim])))
                    energy_steps += 1
                    if bool(_to_numpy(dones).reshape(-1)[0]):
                        terminated_early = True
                        break
                if terminated_early:
                    break

            final_state = state_reader.get_robot_state()
            final_pos = final_state["base_pos"][0]
            positions.append(final_pos.tolist())
            final_distance = float(np.linalg.norm(target - final_pos[:2]))
            final_height = float(final_pos[2])
            min_height = float(np.min(np.asarray(positions)[:, 2])) if positions else final_height
            height_ok = final_height >= args.min_final_height_fraction * max(start_height, 1e-6)
            result = {
                "success": (not terminated_early) and height_ok and final_distance < args.target_threshold,
                "final_distance": final_distance,
                "final_height": final_height,
                "min_height": min_height,
                "energy_proxy": energy_total / max(energy_steps, 1),
                "num_commands_used": (
                    len(selected)
                    if method in {"skill_command", "guarded_skill_command", "learned_guarded_skill_command"}
                    else 0
                ),
                "selected_archive_indices": selected,
                "decisions": decisions,
                "terminated_early": terminated_early,
            }
            if args.store_positions:
                result["positions"] = positions
                result["target_position"] = target.tolist()
            return result

        records: Dict[str, Dict[str, List[Dict[str, Any]]]] = {
            f"{target[0]:g},{target[1]:g}": {method: [] for method in methods}
            for target in targets
        }
        for target_index, target in enumerate(targets):
            target_arr = np.asarray(target, dtype=np.float64)
            key = f"{target[0]:g},{target[1]:g}"
            for trial in range(args.num_trials):
                trial_seed = args.seed + 1000 * target_index + trial
                for method in methods:
                    rec = rollout(method, target_arr, trial_seed)
                    rec["trial"] = trial
                    records[key][method].append(rec)

        summary = {
            target_key: {method: _summarize(method_records) for method, method_records in target_records.items()}
            for target_key, target_records in records.items()
        }
        output = {
            "task": args.task,
            "online_action_set": args.online_action_set,
            "checkpoint": str(resume_path),
            "targets": targets,
            "num_trials": args.num_trials,
            "methods": methods,
            "controller": {
                "target_threshold": args.target_threshold,
                "max_high_level_steps": args.max_high_level_steps,
                "execution_horizon": args.execution_horizon,
                "lambda_state": args.lambda_state,
                "lambda_energy": args.lambda_energy,
                "lambda_stability": args.lambda_stability,
                "lambda_no_progress": args.lambda_no_progress,
                "lambda_utility": args.lambda_utility,
                "lambda_progress": args.lambda_progress,
                "applicability_temperature": args.applicability_temperature,
                "k_nearest": args.k_nearest,
                "feature_zero_slices": feature_zero_slices,
                "command_slice": command_slice,
                "command_gain": args.command_gain,
                "command_max": args.command_max,
                "yaw_command_gain": args.yaw_command_gain,
                "skill_target_command_blend": args.skill_target_command_blend,
                "guard_min_alignment": args.guard_min_alignment,
                "guard_min_progress": args.guard_min_progress,
                "guard_min_stability": args.guard_min_stability,
                "guard_max_energy_z": args.guard_max_energy_z,
                "guard_max_skill_blend": args.guard_max_skill_blend,
                "guard_blend_distance": args.guard_blend_distance,
                "guard_max_speed_gain": args.guard_max_speed_gain,
                "guard_low_height_fraction": args.guard_low_height_fraction,
                "guard_low_height_command_scale": args.guard_low_height_command_scale,
                "viability_model": args.viability_model,
                "viability_threshold": args.viability_threshold,
                "viability_filter_mode": args.viability_filter_mode,
                "viability_feature_names": viability_feature_names,
                "guard_recovery_height_fraction": args.guard_recovery_height_fraction,
                "guard_recovery_command_scale": args.guard_recovery_command_scale,
                "min_final_height_fraction": args.min_final_height_fraction,
                "relative_targets": args.relative_targets,
            },
            "summary": summary,
            "records": records,
        }
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(output, indent=2))
        print(json.dumps({"summary": summary}, indent=2))
        rl_env.close()
    finally:
        sim_app.close()


if __name__ == "__main__":
    main()
