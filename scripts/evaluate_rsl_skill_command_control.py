"""Evaluate skill-selected velocity commands with a pretrained RSL-RL policy.

This is the safer locomotion variant for humanoids: the discovered action set
is used as a high-level library of local motion commands, while the pretrained
RSL-RL policy remains the low-level stabilizing controller.
"""

from __future__ import annotations

import argparse
import faulthandler
import itertools
import json
import os
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

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
    parser.add_argument("--fixed_command_scale", type=float, default=0.75,
                        help="Scale used by the fixed scaled-command baseline.")
    parser.add_argument(
        "--vgs_nominal_scale",
        type=float,
        default=0.75,
        help=(
            "Nominal command scale for viability_gated_scaling_command. "
            "The controller executes this scale only when the response model "
            "certifies progress and posture relative to direct control."
        ),
    )
    parser.add_argument("--reactive_height_fraction", type=float, default=0.9,
                        help="Reactive governor activates below this fraction of initial relative height.")
    parser.add_argument("--reactive_tilt_rad", type=float, default=0.3,
                        help="Reactive governor activates above this measured base tilt.")
    parser.add_argument("--reactive_command_scale", type=float, default=0.5,
                        help="Command scale applied while the reactive governor is active.")
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
    parser.add_argument("--response_model", type=str, default=None,
                        help="Command-response model for model_feedforward_command.")
    parser.add_argument("--task_value_model", type=str, default=None,
                        help="Task-level remaining work/time/success model for task_value_feedforward_command.")
    parser.add_argument(
        "--scale_selector_model",
        type=str,
        default=None,
        help="Episode-level paired-ranking model for paired_scale_selector_command.",
    )
    parser.add_argument(
        "--macro_transition_model",
        type=str,
        default=None,
        help="Recursive macro-transition ensemble for completion_mpc_command.",
    )
    parser.add_argument("--mpc_scales", type=str, default="0.75,0.90,1.00")
    parser.add_argument("--mpc_planning_steps", type=int, default=4)
    parser.add_argument("--mpc_uncertainty_k", type=float, default=1.0)
    parser.add_argument("--mpc_work_margin_frac", type=float, default=0.02)
    parser.add_argument(
        "--mpc_prefix_calibrator",
        type=str,
        default=None,
        help="Exact-replay ensemble calibrator for recursive prefix outcomes.",
    )
    parser.add_argument("--mpc_prefix_uncertainty_k", type=float, default=1.0)
    parser.add_argument(
        "--mpc_progress_ratio",
        type=float,
        default=0.0,
        help="If positive, require calibrated prefix progress LCB relative to all-direct.",
    )
    parser.add_argument(
        "--collect_mpc_dagger",
        action="store_true",
        help=(
            "Collect completion-value correction labels by deterministically replaying "
            "MPC prefixes, executing candidate plans, and switching to direct control."
        ),
    )
    parser.add_argument(
        "--dagger_queries_per_episode",
        type=int,
        default=2,
        help="Number of interior MPC-visited states queried per episode.",
    )
    parser.add_argument(
        "--dagger_candidate_sequences",
        type=str,
        default="selected;0.75,0.75,0.75,0.75;0.90,0.90,0.90,0.90;1.00,1.00,1.00,1.00",
        help=(
            "Semicolon-separated plan library for B1 collection. 'selected' uses the "
            "on-policy MPC plan at the query state."
        ),
    )
    parser.add_argument("--tv_success_margin", type=float, default=0.02,
                        help="Candidate success probability may be at most this below the direct command.")
    parser.add_argument("--tv_time_ratio", type=float, default=1.10,
                        help="Candidate predicted remaining time must not exceed this times direct.")
    parser.add_argument(
        "--tv_command_scales",
        type=str,
        default="0.60,0.75,0.90,1.00",
        help="Comma-separated in-support command scales considered by task_value_scaling_command.",
    )
    parser.add_argument("--ff_progress_floor", type=float, default=0.9,
                        help="Candidate must retain at least this fraction of the direct command's predicted progress.")
    parser.add_argument("--ff_min_height_fraction", type=float, default=0.85,
                        help="Fraction of start height below which predicted min height is considered unsafe.")
    parser.add_argument("--ff_rescue_height_fraction", type=float, default=0.8,
                        help="Rescue mode triggers only if the direct command's predicted min height falls below this fraction of nominal.")
    parser.add_argument("--ff_energy_margin_frac", type=float, default=0.1,
                        help="Required fractional predicted energy saving over the direct command before compensating.")
    parser.add_argument("--ff_current_height_fraction", type=float, default=0.9,
                        help="If the base is below this fraction of nominal relative height, fall back to the direct command to restore posture.")
    parser.add_argument("--ff_max_height_drop", type=float, default=0.02,
                        help="Candidates whose predicted mean relative height falls more than this below the current one are rejected.")
    parser.add_argument(
        "--ff_max_tilt_rad",
        type=float,
        default=0.0,
        help="If positive and predicted by the model, reject candidates above this horizon-max tilt.",
    )
    parser.add_argument(
        "--ff_max_ang_speed",
        type=float,
        default=0.0,
        help="If positive and predicted by the model, reject candidates above this horizon-max base angular speed.",
    )
    parser.add_argument("--ff_blend", type=float, default=0.5,
                        help="Feedforward correction gain: executed command is c_goal + blend * (c_best - c_goal).")
    parser.add_argument("--ff_anneal_distance", type=float, default=0.5,
                        help="Correction gain is scaled by clip(distance/this, 0, 1); near the goal the direct command regains full authority. 0 disables annealing.")
    parser.add_argument("--ff_height_scan_slice", type=str, default="48:235",
                        help="Obs slice holding the height scan used for terrain-relative posture height.")
    parser.add_argument("--ff_height_scan_offset", type=float, default=0.5)
    parser.add_argument(
        "--ff_height_uncertainty_k",
        type=float,
        default=0.0,
        help="Subtract k times ensemble std from predicted min height before viability gating.",
    )
    parser.add_argument(
        "--ff_use_model_conformal_k",
        action="store_true",
        help="Use the one-sided height scale calibrated and stored in an ensemble response model.",
    )
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
    parser.add_argument(
        "--freeze_terrain_curriculum",
        action="store_true",
        help=(
            "Disable episode-to-episode terrain-level updates. This is automatically "
            "enabled for MPC-DAgger counterfactual replay."
        ),
    )
    parser.add_argument(
        "--paired_method_resets",
        action="store_true",
        help="Re-seed the environment before every rollout so methods sharing a target/trial start identically.",
    )
    parser.add_argument("--store_positions", action="store_true")
    parser.add_argument(
        "--store_value_samples",
        action="store_true",
        help="Store high-level state/command snapshots labeled with remaining work, time, and success.",
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--stress_static_friction",
        type=float,
        default=-1.0,
        help="If non-negative, fix robot-body static friction to this value for a stress test.",
    )
    parser.add_argument(
        "--stress_dynamic_friction",
        type=float,
        default=-1.0,
        help="If non-negative, fix robot-body dynamic friction to this value for a stress test.",
    )
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
        "average_max_tilt": float(np.mean([r.get("max_tilt", 0.0) for r in records])),
        "average_max_ang_speed": float(np.mean([r.get("max_ang_speed", 0.0) for r in records])),
        "average_mechanical_energy_j": float(
            np.mean([r.get("mechanical_energy_j", float("nan")) for r in records])
        ),
        "average_mechanical_power_w": float(
            np.mean([r.get("mean_mechanical_power_w", float("nan")) for r in records])
        ),
        "average_energy_per_meter_j_m": float(
            np.mean([r.get("energy_per_meter_j_m", float("nan")) for r in records])
        ),
        "average_cost_of_transport": float(
            np.mean([r.get("cost_of_transport", float("nan")) for r in records])
        ),
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
        mpc_debug_timing = bool(os.environ.get("MPC_DEBUG_TIMING"))
        if mpc_debug_timing:
            faulthandler.dump_traceback_later(20.0, repeat=True)
        targets = parse_targets(args.targets)
        methods = [item.strip() for item in args.methods.split(",") if item.strip()]
        valid_methods = {
            "skill_command",
            "guarded_skill_command",
            "learned_guarded_skill_command",
            "model_feedforward_command",
            "mechanical_feedforward_command",
            "mechanical_efficiency_feedforward_command",
            "aligned_efficiency_feedforward_command",
            "aligned_proxy_feedforward_command",
            "proxy_efficiency_feedforward_command",
            "viability_gated_scaling_command",
            "task_value_feedforward_command",
            "task_value_scaling_command",
            "paired_scale_selector_command",
            "completion_mpc_command",
            "direct_target_command",
            "scaled_target_command",
            "scaled_target_command_60",
            "scaled_target_command_75",
            "scaled_target_command_90",
            "reactive_governor_command",
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
        ff_methods = {
            "model_feedforward_command",
            "mechanical_feedforward_command",
            "mechanical_efficiency_feedforward_command",
            "aligned_efficiency_feedforward_command",
            "aligned_proxy_feedforward_command",
            "proxy_efficiency_feedforward_command",
            "viability_gated_scaling_command",
            "task_value_feedforward_command",
            "task_value_scaling_command",
        }
        if set(methods) & ff_methods and not args.response_model:
            raise ValueError("--response_model is required for feedforward command methods.")
        if set(methods) & {"task_value_feedforward_command", "task_value_scaling_command"} and not args.task_value_model:
            raise ValueError("--task_value_model is required for task-value command methods.")
        if "paired_scale_selector_command" in methods and not args.scale_selector_model:
            raise ValueError("--scale_selector_model is required for paired_scale_selector_command.")
        if "completion_mpc_command" in methods:
            if not args.macro_transition_model:
                raise ValueError("--macro_transition_model is required for completion_mpc_command.")
            if not args.task_value_model:
                raise ValueError("--task_value_model must provide the direct terminal value for completion_mpc_command.")
        if args.collect_mpc_dagger:
            if "completion_mpc_command" not in methods:
                raise ValueError("--collect_mpc_dagger requires completion_mpc_command in --methods.")
            if not args.paired_method_resets:
                raise ValueError("--collect_mpc_dagger requires --paired_method_resets for exact prefix replay.")
        if args.mpc_prefix_calibrator and "completion_mpc_command" not in methods:
            raise ValueError("--mpc_prefix_calibrator is only used by completion_mpc_command.")
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
        freeze_terrain_curriculum = bool(
            args.freeze_terrain_curriculum or args.collect_mpc_dagger
        )
        if freeze_terrain_curriculum and hasattr(env_cfg, "curriculum"):
            if hasattr(env_cfg.curriculum, "terrain_levels"):
                env_cfg.curriculum.terrain_levels = None
        agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
        agent_cfg.seed = args.seed
        agent_cfg.device = args.device
        env_cfg.seed = args.seed
        env_cfg.sim.device = args.device
        if args.stress_static_friction >= 0.0:
            env_cfg.events.physics_material.params["static_friction_range"] = (
                args.stress_static_friction,
                args.stress_static_friction,
            )
        if args.stress_dynamic_friction >= 0.0:
            env_cfg.events.physics_material.params["dynamic_friction_range"] = (
                args.stress_dynamic_friction,
                args.stress_dynamic_friction,
            )

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
        control_dt = float(getattr(raw_env.unwrapped, "step_dt", 0.02))
        robot_mass = state_reader.get_robot_mass()

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

        response_model = None
        response_models = []
        response_stats = None
        if args.response_model:
            try:
                rm_data = torch.load(args.response_model, map_location="cpu", weights_only=False)
            except TypeError:
                rm_data = torch.load(args.response_model, map_location="cpu")
            rm_hidden = int(rm_data["hidden_dim"])
            rm_dropout = float(rm_data.get("dropout", 0.0))
            def make_response_model():
                return torch.nn.Sequential(
                    torch.nn.Linear(int(rm_data["obs_dim"]), rm_hidden),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(rm_dropout),
                    torch.nn.Linear(rm_hidden, rm_hidden),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(rm_dropout),
                    torch.nn.Linear(rm_hidden, len(rm_data["output_names"])),
                )

            state_dicts = rm_data.get("state_dicts", [rm_data.get("state_dict")])
            for state_dict in state_dicts:
                if state_dict is None:
                    raise ValueError("Response model contains no state_dict.")
                member_model = make_response_model()
                member_model.load_state_dict(state_dict)
                member_model.eval()
                response_models.append(member_model)
            response_model = response_models[0]
            response_stats = {
                "x_mean": np.asarray(rm_data["x_mean"], dtype=np.float32),
                "x_std": np.asarray(rm_data["x_std"], dtype=np.float32),
                "y_mean": np.asarray(rm_data["y_mean"], dtype=np.float32),
                "y_std": np.asarray(rm_data["y_std"], dtype=np.float32),
                "output_names": list(rm_data["output_names"]),
                "height_conformal_k": float(rm_data.get("height_conformal_k", 0.0)),
            }

        task_value_model = None
        task_value_stats = None
        task_value_device = torch.device("cpu")
        if args.task_value_model:
            try:
                tv_data = torch.load(args.task_value_model, map_location="cpu", weights_only=False)
            except TypeError:
                tv_data = torch.load(args.task_value_model, map_location="cpu")
            tv_hidden = int(tv_data["hidden_dim"])
            task_value_model = torch.nn.Sequential(
                torch.nn.Linear(int(tv_data["input_dim"]), tv_hidden),
                torch.nn.ReLU(),
                torch.nn.Linear(tv_hidden, tv_hidden),
                torch.nn.ReLU(),
                torch.nn.Linear(tv_hidden, 3),
            )
            task_value_model.load_state_dict(tv_data["state_dict"])
            task_value_model.eval()
            task_value_stats = {
                "obs_dim": int(tv_data["obs_dim"]),
                "command_slice": tuple(tv_data["command_slice"]),
                "x_mean": np.asarray(tv_data["x_mean"], dtype=np.float32),
                "x_std": np.asarray(tv_data["x_std"], dtype=np.float32),
                "y_mean": np.asarray(tv_data["y_mean"], dtype=np.float32),
                "y_std": np.asarray(tv_data["y_std"], dtype=np.float32),
            }

        scale_selector = None
        if args.scale_selector_model:
            selector_data = json.loads(Path(args.scale_selector_model).read_text())
            if selector_data.get("model_type") != "paired_scale_selector":
                raise ValueError("Scale selector model has an unsupported model_type.")
            scale_selector = {
                "x_mean": np.asarray(selector_data["x_mean"], dtype=np.float32),
                "x_std": np.asarray(selector_data["x_std"], dtype=np.float32),
                "coef": np.asarray(selector_data["coef"], dtype=np.float32),
                "intercept": float(selector_data["intercept"]),
                "threshold": float(selector_data["threshold"]),
                "selected_scale": float(selector_data["selected_scale"]),
                "fallback_scale": float(selector_data["fallback_scale"]),
                "height_scan_slice": tuple(selector_data["height_scan_slice"]),
            }

        macro_transition_models = []
        macro_transition_stats = None
        macro_transition_params = None
        macro_transition_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.macro_transition_model:
            macro_load_start = time.perf_counter()
            try:
                macro_data = torch.load(
                    args.macro_transition_model, map_location="cpu", weights_only=False
                )
            except TypeError:
                macro_data = torch.load(args.macro_transition_model, map_location="cpu")
            if macro_data.get("model_type") != "macro_transition_ensemble":
                raise ValueError("Macro transition model has an unsupported model_type.")
            macro_hidden = int(macro_data["hidden_dim"])
            for state_dict in macro_data["state_dicts"]:
                model = torch.nn.Sequential(
                    torch.nn.Linear(int(macro_data["input_dim"]), macro_hidden),
                    torch.nn.ReLU(),
                    torch.nn.Linear(macro_hidden, macro_hidden),
                    torch.nn.ReLU(),
                    torch.nn.Linear(macro_hidden, int(macro_data["output_dim"])),
                )
                model.load_state_dict(state_dict)
                model.eval()
                model.to(macro_transition_device)
                macro_transition_models.append(model)
            macro_transition_stats = {
                "input_dim": int(macro_data["input_dim"]),
                "response_names": list(macro_data["response_names"]),
                "x_mean": np.asarray(macro_data["x_mean"], dtype=np.float32),
                "x_std": np.asarray(macro_data["x_std"], dtype=np.float32),
                "y_mean": np.asarray(macro_data["y_mean"], dtype=np.float32),
                "y_std": np.asarray(macro_data["y_std"], dtype=np.float32),
                "horizon": int(macro_data["horizon"]),
                "control_dt": float(macro_data["control_dt"]),
                "command_slice": tuple(macro_data["command_slice"]),
            }
            macro_transition_params = {
                key: torch.stack(
                    [model.state_dict()[key].detach() for model in macro_transition_models]
                ).to(macro_transition_device)
                for key in (
                    "0.weight",
                    "0.bias",
                    "2.weight",
                    "2.bias",
                    "4.weight",
                    "4.bias",
                )
            }
            if task_value_model is not None:
                task_value_device = macro_transition_device
                task_value_model.to(task_value_device)
            if mpc_debug_timing:
                print(
                    f"[MPC timing] model load: {time.perf_counter() - macro_load_start:.3f}s",
                    file=sys.stderr,
                    flush=True,
                )

        prefix_calibrator_models = []
        prefix_calibrator_stats = None
        if args.mpc_prefix_calibrator:
            try:
                prefix_data = torch.load(
                    args.mpc_prefix_calibrator, map_location="cpu", weights_only=False
                )
            except TypeError:
                prefix_data = torch.load(args.mpc_prefix_calibrator, map_location="cpu")
            if prefix_data.get("model_type") != "mpc_prefix_calibrator_ensemble":
                raise ValueError("Prefix calibrator has an unsupported model_type.")
            for state_dict in prefix_data["state_dicts"]:
                model = torch.nn.Sequential(
                    torch.nn.Linear(
                        int(prefix_data["input_dim"]), int(prefix_data["hidden_dim"])
                    ),
                    torch.nn.ReLU(),
                    torch.nn.Linear(
                        int(prefix_data["hidden_dim"]), int(prefix_data["hidden_dim"])
                    ),
                    torch.nn.ReLU(),
                    torch.nn.Linear(
                        int(prefix_data["hidden_dim"]), len(prefix_data["output_names"])
                    ),
                )
                model.load_state_dict(state_dict)
                model.eval()
                model.to(macro_transition_device)
                prefix_calibrator_models.append(model)
            prefix_calibrator_stats = {
                "x_mean": np.asarray(prefix_data["x_mean"], dtype=np.float32),
                "x_std": np.asarray(prefix_data["x_std"], dtype=np.float32),
                "y_mean": np.asarray(prefix_data["y_mean"], dtype=np.float32),
                "y_std": np.asarray(prefix_data["y_std"], dtype=np.float32),
                "output_names": list(prefix_data["output_names"]),
                "height_scan_slice": tuple(prefix_data["height_scan_slice"]),
                "target_mode": prefix_data.get("target_mode", "absolute"),
            }

        def predict_scale_selection(
            obs_vec: np.ndarray, local_target: np.ndarray, distance: float
        ) -> float:
            if scale_selector is None:
                raise RuntimeError("Scale selector model is not loaded.")
            scan_lo, scan_hi = scale_selector["height_scan_slice"]
            scan = obs_vec[scan_lo:scan_hi]
            scan_stats = np.asarray(
                [
                    scan.mean(),
                    scan.std(),
                    scan.min(),
                    scan.max(),
                    *np.quantile(scan, [0.10, 0.25, 0.50, 0.75, 0.90]),
                ],
                dtype=np.float32,
            )
            features = np.concatenate(
                [
                    np.asarray(local_target[:2], dtype=np.float32),
                    np.asarray([distance], dtype=np.float32),
                    obs_vec[:9].astype(np.float32),
                    obs_vec[12:48].astype(np.float32),
                    scan_stats,
                ]
            )
            normalized = (features - scale_selector["x_mean"]) / scale_selector["x_std"]
            logit = float(normalized @ scale_selector["coef"] + scale_selector["intercept"])
            return float(1.0 / (1.0 + np.exp(-np.clip(logit, -40.0, 40.0))))

        def predict_task_values(
            obs_vec: np.ndarray,
            local_target: np.ndarray,
            distance: float,
            candidates: np.ndarray,
        ):
            if task_value_model is None or task_value_stats is None:
                raise RuntimeError("Task value model is not loaded.")
            Xobs = np.tile(obs_vec[None, :], (len(candidates), 1)).astype(np.float32)
            lo_tv, hi_tv = task_value_stats["command_slice"]
            Xobs[:, lo_tv:hi_tv] = candidates[:, : hi_tv - lo_tv]
            target_part = np.tile(
                np.asarray([local_target[0], local_target[1], distance], dtype=np.float32)[None, :],
                (len(candidates), 1),
            )
            X = np.concatenate([Xobs, target_part], axis=1)
            Xn = (X - task_value_stats["x_mean"]) / task_value_stats["x_std"]
            with torch.inference_mode():
                raw = (
                    task_value_model(
                        torch.from_numpy(Xn.astype(np.float32)).to(task_value_device)
                    )
                    .cpu()
                    .numpy()
                )
            reg = raw[:, :2] * task_value_stats["y_std"] + task_value_stats["y_mean"]
            prob = 1.0 / (1.0 + np.exp(-raw[:, 2]))
            return reg[:, 0], reg[:, 1], prob

        mpc_sequences = None
        dagger_sequence_specs: List[tuple[str, Optional[np.ndarray]]] = []
        if macro_transition_stats is not None:
            mpc_scales = sorted(
                {
                    float(np.clip(float(value), 0.0, 1.0))
                    for value in args.mpc_scales.split(",")
                    if value.strip()
                }
            )
            if 1.0 not in mpc_scales:
                mpc_scales.append(1.0)
            planning_steps = max(1, int(args.mpc_planning_steps))
            if len(mpc_scales) ** planning_steps > 1000:
                raise ValueError("MPC candidate count exceeds 1000; reduce scales or planning steps.")
            mpc_sequences = np.asarray(
                list(itertools.product(mpc_scales, repeat=planning_steps)), dtype=np.float32
            )
            if args.collect_mpc_dagger:
                for raw_spec in args.dagger_candidate_sequences.split(";"):
                    raw_spec = raw_spec.strip()
                    if not raw_spec:
                        continue
                    if raw_spec.lower() == "selected":
                        dagger_sequence_specs.append(("selected", None))
                        continue
                    sequence = np.asarray(
                        [float(value) for value in raw_spec.split(",") if value.strip()],
                        dtype=np.float32,
                    )
                    if len(sequence) != planning_steps:
                        raise ValueError(
                            f"DAgger sequence '{raw_spec}' has {len(sequence)} values; "
                            f"expected {planning_steps}."
                        )
                    if not np.any(
                        np.all(np.isclose(mpc_sequences, sequence[None, :]), axis=1)
                    ):
                        raise ValueError(
                            f"DAgger sequence '{raw_spec}' is outside --mpc_scales."
                        )
                    dagger_sequence_specs.append((raw_spec, sequence))
                if not dagger_sequence_specs:
                    raise ValueError("--dagger_candidate_sequences contains no candidates.")

        def predict_prefix_calibration(
            obs_vec: np.ndarray,
            local_target: np.ndarray,
            start_relative_height: float,
            predicted_prefix: Dict[str, np.ndarray],
        ):
            if not prefix_calibrator_models or prefix_calibrator_stats is None:
                return None
            scan_lo, scan_hi = prefix_calibrator_stats["height_scan_slice"]
            scan = np.asarray(obs_vec[scan_lo:scan_hi], dtype=np.float32)
            scan_stats = np.asarray(
                [
                    scan.mean(),
                    scan.std(),
                    scan.min(),
                    scan.max(),
                    *np.quantile(scan, [0.10, 0.25, 0.50, 0.75, 0.90]),
                ],
                dtype=np.float32,
            )
            prefix_stats = []
            for name in prefix_calibrator_stats["output_names"]:
                values = predicted_prefix[name]
                prefix_stats.extend([values.mean(0), values.std(0)])
            prefix_stats = np.stack(prefix_stats, axis=1).astype(np.float32)
            sequence_count = len(mpc_sequences)
            shared = np.concatenate(
                [
                    np.asarray(obs_vec[:48], dtype=np.float32),
                    scan_stats,
                    np.asarray(local_target[:2], dtype=np.float32),
                    np.asarray(
                        [np.linalg.norm(local_target[:2]), start_relative_height],
                        dtype=np.float32,
                    ),
                ]
            )
            features = np.concatenate(
                [
                    np.tile(shared[None, :], (sequence_count, 1)),
                    mpc_sequences.astype(np.float32),
                    prefix_stats,
                ],
                axis=1,
            )
            normalized = (
                features - prefix_calibrator_stats["x_mean"]
            ) / prefix_calibrator_stats["x_std"]
            predictions = []
            with torch.inference_mode():
                tensor = torch.from_numpy(normalized.astype(np.float32)).to(
                    macro_transition_device
                )
                for model in prefix_calibrator_models:
                    raw = model(tensor).cpu().numpy()
                    prediction = (
                        raw * prefix_calibrator_stats["y_std"]
                        + prefix_calibrator_stats["y_mean"]
                    )
                    if prefix_calibrator_stats["target_mode"] == "residual":
                        prediction = prediction + np.stack(
                            [
                                predicted_prefix[name].mean(0)
                                for name in prefix_calibrator_stats["output_names"]
                            ],
                            axis=1,
                        )
                    predictions.append(prediction)
            predictions = np.stack(predictions)
            return predictions.mean(0), predictions.std(0)

        def score_completion_mpc(
            obs_vec: np.ndarray,
            local_target: np.ndarray,
            start_relative_height: float,
            return_terminal_inputs: bool = False,
        ):
            if not macro_transition_models or macro_transition_stats is None or mpc_sequences is None:
                raise RuntimeError("Macro transition ensemble is not loaded.")
            if task_value_model is None or task_value_stats is None:
                raise RuntimeError("Direct terminal value model is not loaded.")
            debug_start = time.perf_counter()
            member_count = len(macro_transition_models)
            sequence_count, planning_steps = mpc_sequences.shape
            obs_dim = macro_transition_stats["input_dim"]
            states = np.tile(
                np.asarray(obs_vec, dtype=np.float32)[None, None, :],
                (member_count, sequence_count, 1),
            )
            targets = np.tile(
                np.asarray(local_target[:2], dtype=np.float32)[None, None, :],
                (member_count, sequence_count, 1),
            )
            initial_target_distance = float(np.linalg.norm(local_target[:2]))
            cumulative_work = np.zeros((member_count, sequence_count), dtype=np.float32)
            min_height = np.full((member_count, sequence_count), np.inf, dtype=np.float32)
            max_tilt = np.zeros((member_count, sequence_count), dtype=np.float32)
            max_ang_speed = np.zeros((member_count, sequence_count), dtype=np.float32)
            response_offset = obs_dim
            response_map = {
                name: response_offset + index
                for index, name in enumerate(macro_transition_stats["response_names"])
            }
            macro_duration = (
                macro_transition_stats["horizon"] * macro_transition_stats["control_dt"]
            )
            lo_macro, hi_macro = macro_transition_stats["command_slice"]
            for step in range(planning_steps):
                commands = np.zeros((member_count, sequence_count, command_dim), dtype=np.float32)
                if command_dim >= 2:
                    commands[:, :, :2] = np.clip(
                        targets * args.command_gain, -args.command_max, args.command_max
                    )
                elif command_dim == 1:
                    commands[:, :, 0] = np.clip(
                        targets[:, :, 0] * args.command_gain,
                        -args.command_max,
                        args.command_max,
                    )
                if command_dim >= 3:
                    heading = np.arctan2(targets[:, :, 1], np.maximum(targets[:, :, 0], 1e-6))
                    commands[:, :, 2] = np.clip(
                        heading * args.yaw_command_gain, -args.command_max, args.command_max
                    )
                commands *= mpc_sequences[None, :, step, None]
                states[:, :, lo_macro:hi_macro] = commands[:, :, : hi_macro - lo_macro]
                normalized = (
                    states - macro_transition_stats["x_mean"][None, None, :]
                ) / macro_transition_stats["x_std"][None, None, :]
                with torch.inference_mode():
                    tensor = torch.from_numpy(normalized.astype(np.float32)).to(
                        macro_transition_device
                    )
                    hidden = torch.relu(
                        torch.bmm(
                            tensor, macro_transition_params["0.weight"].transpose(1, 2)
                        )
                        + macro_transition_params["0.bias"][:, None, :]
                    )
                    hidden = torch.relu(
                        torch.bmm(
                            hidden, macro_transition_params["2.weight"].transpose(1, 2)
                        )
                        + macro_transition_params["2.bias"][:, None, :]
                    )
                    raw = (
                        torch.bmm(
                            hidden, macro_transition_params["4.weight"].transpose(1, 2)
                        )
                        + macro_transition_params["4.bias"][:, None, :]
                    )
                    member_outputs = raw.cpu().numpy()
                member_outputs = (
                    member_outputs * macro_transition_stats["y_std"][None, None, :]
                    + macro_transition_stats["y_mean"][None, None, :]
                )
                states += member_outputs[:, :, :obs_dim]
                dx = member_outputs[:, :, response_map["delta_x"]]
                dy = member_outputs[:, :, response_map["delta_y"]]
                dyaw = member_outputs[:, :, response_map["delta_yaw"]]
                power = np.maximum(
                    member_outputs[:, :, response_map["mechanical_power"]], 0.0
                )
                cumulative_work += power * macro_duration
                min_height = np.minimum(
                    min_height, member_outputs[:, :, response_map["min_height"]]
                )
                if "max_tilt" in response_map:
                    max_tilt = np.maximum(
                        max_tilt, member_outputs[:, :, response_map["max_tilt"]]
                    )
                if "max_ang_speed" in response_map:
                    max_ang_speed = np.maximum(
                        max_ang_speed, member_outputs[:, :, response_map["max_ang_speed"]]
                    )
                remaining_x = targets[:, :, 0] - dx
                remaining_y = targets[:, :, 1] - dy
                c, s = np.cos(-dyaw), np.sin(-dyaw)
                targets = np.stack(
                    [c * remaining_x - s * remaining_y, s * remaining_x + c * remaining_y],
                    axis=2,
                ).astype(np.float32)
            if mpc_debug_timing:
                print(
                    f"[MPC timing] recursive rollout: {time.perf_counter() - debug_start:.3f}s",
                    file=sys.stderr,
                    flush=True,
                )

            flat_states = states.reshape(member_count * sequence_count, obs_dim).copy()
            flat_targets = targets.reshape(member_count * sequence_count, 2)
            flat_distances = np.linalg.norm(flat_targets, axis=1)
            terminal_commands = np.zeros(
                (member_count * sequence_count, command_dim), dtype=np.float32
            )
            if command_dim >= 2:
                terminal_commands[:, :2] = np.clip(
                    flat_targets * args.command_gain, -args.command_max, args.command_max
                )
            elif command_dim == 1:
                terminal_commands[:, 0] = np.clip(
                    flat_targets[:, 0] * args.command_gain,
                    -args.command_max,
                    args.command_max,
                )
            if command_dim >= 3:
                terminal_commands[:, 2] = np.clip(
                    np.arctan2(flat_targets[:, 1], np.maximum(flat_targets[:, 0], 1e-6))
                    * args.yaw_command_gain,
                    -args.command_max,
                    args.command_max,
                )
            lo_tv, hi_tv = task_value_stats["command_slice"]
            flat_states[:, lo_tv:hi_tv] = terminal_commands[:, : hi_tv - lo_tv]
            terminal_input = np.concatenate(
                [flat_states, flat_targets, flat_distances[:, None]], axis=1
            )
            terminal_normalized = (
                terminal_input - task_value_stats["x_mean"]
            ) / task_value_stats["x_std"]
            with torch.inference_mode():
                terminal_raw = task_value_model(
                    torch.from_numpy(terminal_normalized.astype(np.float32)).to(task_value_device)
                ).cpu().numpy()
            if mpc_debug_timing:
                print(
                    f"[MPC timing] plus terminal: {time.perf_counter() - debug_start:.3f}s",
                    file=sys.stderr,
                    flush=True,
                )
            terminal_reg = (
                terminal_raw[:, :2] * task_value_stats["y_std"]
                + task_value_stats["y_mean"]
            ).reshape(member_count, sequence_count, 2)
            terminal_success = (
                1.0 / (1.0 + np.exp(-terminal_raw[:, 2]))
            ).reshape(member_count, sequence_count)
            total_work_members = cumulative_work + np.maximum(terminal_reg[:, :, 0], 0.0)
            total_time_members = planning_steps * macro_duration + np.maximum(
                terminal_reg[:, :, 1], 0.0
            )
            calibrated_prefix = predict_prefix_calibration(
                obs_vec,
                local_target,
                start_relative_height,
                {
                    "work_j": cumulative_work,
                    "progress_m": (
                        initial_target_distance - np.linalg.norm(targets, axis=2)
                    ).astype(np.float32),
                    "min_height": min_height,
                    "max_tilt": max_tilt,
                    "max_ang_speed": max_ang_speed,
                },
            )
            if calibrated_prefix is None:
                work_mean = total_work_members.mean(0)
                work_std = total_work_members.std(0)
                progress_mean = (
                    initial_target_distance - np.linalg.norm(targets, axis=2)
                ).mean(0)
                progress_std = (
                    initial_target_distance - np.linalg.norm(targets, axis=2)
                ).std(0)
                height_mean = min_height.mean(0)
                height_std = min_height.std(0)
                tilt_mean = max_tilt.mean(0)
                ang_speed_mean = max_ang_speed.mean(0)
            else:
                calibrated_mean, calibrated_std = calibrated_prefix
                cj = {
                    name: index
                    for index, name in enumerate(prefix_calibrator_stats["output_names"])
                }
                terminal_work = np.maximum(terminal_reg[:, :, 0], 0.0)
                work_mean = calibrated_mean[:, cj["work_j"]] + terminal_work.mean(0)
                work_std = np.sqrt(
                    calibrated_std[:, cj["work_j"]] ** 2 + terminal_work.std(0) ** 2
                )
                progress_mean = calibrated_mean[:, cj["progress_m"]]
                progress_std = calibrated_std[:, cj["progress_m"]]
                height_mean = calibrated_mean[:, cj["min_height"]]
                height_std = calibrated_std[:, cj["min_height"]]
                tilt_mean = calibrated_mean[:, cj["max_tilt"]]
                ang_speed_mean = calibrated_mean[:, cj["max_ang_speed"]]
            time_mean = total_time_members.mean(0)
            time_std = total_time_members.std(0)
            success_mean = terminal_success.mean(0)
            success_std = terminal_success.std(0)
            uncertainty_k = max(0.0, args.mpc_uncertainty_k)
            work_ucb = work_mean + uncertainty_k * work_std
            time_ucb = time_mean + uncertainty_k * time_std
            success_lcb = success_mean - uncertainty_k * success_std
            height_lcb = height_mean - uncertainty_k * height_std
            progress_lcb = progress_mean - max(
                0.0, args.mpc_prefix_uncertainty_k
            ) * progress_std
            direct_idx = int(
                np.flatnonzero(np.all(np.isclose(mpc_sequences, 1.0), axis=1))[0]
            )
            eligible = (
                (success_lcb >= success_lcb[direct_idx] - args.tv_success_margin)
                & (time_ucb <= args.tv_time_ratio * time_ucb[direct_idx])
                & (height_lcb >= args.ff_min_height_fraction * max(start_relative_height, 1e-6))
            )
            if args.mpc_progress_ratio > 0.0:
                eligible &= progress_lcb >= args.mpc_progress_ratio * max(
                    progress_lcb[direct_idx], 0.0
                )
            if args.ff_max_tilt_rad > 0.0:
                eligible &= tilt_mean <= args.ff_max_tilt_rad
            if args.ff_max_ang_speed > 0.0:
                eligible &= ang_speed_mean <= args.ff_max_ang_speed
            eligible[direct_idx] = True
            best_idx = int(np.argmin(np.where(eligible, work_ucb, np.inf)))
            active = bool(
                work_mean[best_idx]
                < work_mean[direct_idx] * (1.0 - args.mpc_work_margin_frac)
            )
            if not active:
                best_idx = direct_idx
            result = {
                "sequence": mpc_sequences[best_idx],
                "active": active,
                "best_index": best_idx,
                "direct_index": direct_idx,
                "predicted_work_best": float(work_mean[best_idx]),
                "predicted_work_direct": float(work_mean[direct_idx]),
                "predicted_work_std_best": float(work_std[best_idx]),
                "predicted_time_best": float(time_mean[best_idx]),
                "predicted_time_direct": float(time_mean[direct_idx]),
                "predicted_success_best": float(success_mean[best_idx]),
                "predicted_success_direct": float(success_mean[direct_idx]),
                "predicted_min_height_lcb_best": float(height_lcb[best_idx]),
                "eligible_count": int(eligible.sum()),
                "prefix_calibrated": bool(calibrated_prefix is not None),
                "predicted_progress_best": float(progress_mean[best_idx]),
                "predicted_progress_direct": float(progress_mean[direct_idx]),
            }
            if return_terminal_inputs:
                result["terminal_value_inputs"] = terminal_input.reshape(
                    member_count, sequence_count, -1
                )
                result["prefix_predictions"] = {
                    "work_j": cumulative_work.copy(),
                    "progress_m": (
                        initial_target_distance - np.linalg.norm(targets, axis=2)
                    ).astype(np.float32),
                    "min_height": min_height.copy(),
                    "max_tilt": max_tilt.copy(),
                    "max_ang_speed": max_ang_speed.copy(),
                }
            return result

        def _ff_candidate_commands(target_command: np.ndarray) -> np.ndarray:
            cands = [np.asarray(target_command, dtype=np.float32)]
            for s in (0.4, 0.6, 0.8, 1.2):
                cands.append(np.clip(target_command * s, -args.command_max, args.command_max))
            for dyaw in (-0.4, -0.2, 0.2, 0.4):
                c = np.array(target_command, dtype=np.float32)
                if command_dim >= 3:
                    c[2] = float(np.clip(c[2] + dyaw, -args.command_max, args.command_max))
                cands.append(c)
            base_ang = float(np.arctan2(target_command[1], target_command[0])) if command_dim >= 2 else 0.0
            base_yaw = float(target_command[2]) if command_dim >= 3 else 0.0
            for speed in (0.25, 0.5, 0.75, 1.0):
                for dang in (-0.6, -0.3, 0.0, 0.3, 0.6):
                    for yaw_scale in (0.0, 0.5, 1.0):
                        c = np.zeros(command_dim, dtype=np.float32)
                        ang = base_ang + dang
                        if command_dim >= 2:
                            c[0] = speed * np.cos(ang)
                            c[1] = speed * np.sin(ang)
                        if command_dim >= 3:
                            c[2] = base_yaw * yaw_scale
                        cands.append(np.clip(c, -args.command_max, args.command_max))
            return np.unique(np.stack(cands).astype(np.float32), axis=0)

        def predict_response_stats(obs_vec: np.ndarray, candidates: np.ndarray):
            X = np.tile(obs_vec[None, :], (candidates.shape[0], 1)).astype(np.float32)
            lo, hi = command_slice
            X[:, lo : min(hi, X.shape[1])] = candidates[:, : max(0, min(hi, X.shape[1]) - lo)]
            Xn = (X - response_stats["x_mean"]) / response_stats["x_std"]
            with torch.inference_mode():
                member_preds = [
                    model(torch.from_numpy(Xn.astype(np.float32))).numpy()
                    for model in response_models
                ]
            pred = np.stack(member_preds) * response_stats["y_std"] + response_stats["y_mean"]
            mean = pred.mean(axis=0)
            std = pred.std(axis=0, ddof=1) if len(response_models) > 1 else np.zeros_like(mean)
            return mean, std

        def predict_responses(obs_vec: np.ndarray, candidates: np.ndarray) -> np.ndarray:
            return predict_response_stats(obs_vec, candidates)[0]

        ff_scan_slice = parse_obs_slices(args.ff_height_scan_slice)[0]

        def relative_height(obs_vec: np.ndarray) -> float:
            lo, hi = ff_scan_slice
            return float(np.mean(obs_vec[lo:hi]) + args.ff_height_scan_offset)

        rng = np.random.default_rng(args.seed)

        def rollout(
            method: str,
            target_offset: np.ndarray,
            trial_seed: int,
            forced_scale_schedule: Optional[List[float]] = None,
            dagger_query_step: Optional[int] = None,
            dagger_sequence: Optional[np.ndarray] = None,
            reference_query_observation: Optional[np.ndarray] = None,
        ) -> Dict[str, Any]:
            nonlocal rng
            if mpc_debug_timing:
                print(f"[MPC timing] rollout start: {method}", file=sys.stderr, flush=True)
            if args.paired_method_resets:
                rl_env.seed(trial_seed)
            obs, _ = rl_env.reset()
            if mpc_debug_timing:
                print(f"[MPC timing] reset complete: {method}", file=sys.stderr, flush=True)
            if policy_nn is not None and hasattr(policy_nn, "reset"):
                try:
                    policy_nn.reset(torch.ones(args.num_envs, dtype=torch.bool, device=rl_env.unwrapped.device))
                except TypeError:
                    policy_nn.reset()

            start_state = state_reader.get_robot_state()
            start_pos = start_state["base_pos"][0]
            start_height = float(start_pos[2])
            start_rel_height = relative_height(_obs_for_feature(obs)[0])
            target = np.asarray(target_offset, dtype=np.float64)
            if args.relative_targets:
                target = start_pos[:2].astype(np.float64) + target

            selected: List[int] = []
            decisions: List[Dict[str, Any]] = []
            value_samples: List[Dict[str, Any]] = []
            positions: List[List[float]] = []
            ff_activations = 0
            mpc_activations = 0
            selected_policy_scale = None
            energy_total, energy_steps = 0.0, 0
            mechanical_energy_j = 0.0
            positive_work_j = 0.0
            negative_work_j = 0.0
            elapsed_time_s = 0.0
            path_length_m = 0.0
            previous_low_pos = np.asarray(start_pos[:2], dtype=np.float64)
            q0 = np.asarray(start_state["base_quat"][0], dtype=np.float64)
            max_tilt = float(np.arccos(np.clip(1.0 - 2.0 * (q0[1] ** 2 + q0[2] ** 2), -1.0, 1.0)))
            max_ang_speed = float(np.linalg.norm(start_state["base_ang_vel"][0]))
            terminated_early = False
            dagger_predicted_inputs = None
            dagger_query_observation = None
            dagger_query_local_target = None
            dagger_actual_terminal_input = None
            dagger_terminal_work_start = None
            dagger_terminal_time_start = None
            dagger_replay_observation_l2 = None
            dagger_predicted_prefix = None
            dagger_query_work_start = None
            dagger_query_time_start = None
            dagger_query_distance = None
            dagger_prefix_min_relative_height = float("inf")
            dagger_prefix_max_tilt = 0.0
            dagger_prefix_max_ang_speed = 0.0
            rng = np.random.default_rng(trial_seed)

            for high_step in range(args.max_high_level_steps):
                state = state_reader.get_robot_state()
                pos = state["base_pos"][0]
                yaw = float(quat_to_yaw(state["base_quat"][0]))
                positions.append(pos.tolist())
                distance = float(np.linalg.norm(target - pos[:2]))
                recovery_height = args.guard_recovery_height_fraction * max(start_height, 1e-6)
                needs_recovery = (
                    method
                    in {
                        "guarded_skill_command",
                        "learned_guarded_skill_command",
                        "model_feedforward_command",
                        "mechanical_feedforward_command",
                        "proxy_efficiency_feedforward_command",
                        "direct_target_command",
                        "scaled_target_command",
                        "reactive_governor_command",
                    }
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
                    if method in ff_methods:
                        # Predictive terminal recovery: among commands whose predicted end
                        # position stays inside the target radius, pick the one the response
                        # model expects to raise posture most. Reactive standing does not
                        # restore posture; model-selected stepping motion can.
                        obs_vec = _obs_for_feature(obs)[0]
                        candidates = _ff_candidate_commands(target_command)
                        preds = predict_responses(obs_vec, candidates)
                        j = {n: i for i, n in enumerate(response_stats["output_names"])}
                        local_xy = np.asarray(local_target[:2], dtype=np.float64)
                        pred_dist = np.sqrt(
                            (local_xy[0] - preds[:, j["delta_x"]]) ** 2
                            + (local_xy[1] - preds[:, j["delta_y"]]) ** 2
                        )
                        stay = pred_dist <= args.target_threshold
                        if not bool(stay.any()):
                            stay[:] = True
                        best_idx = int(np.argmax(np.where(stay, preds[:, j["min_height"]], -np.inf)))
                        command = candidates[best_idx]
                        ff_activations += 1
                        decisions.append(
                            {
                                "ff_mode": "terminal_recovery",
                                "ff_active": True,
                                "current_distance": distance,
                                "current_height": float(pos[2]),
                                "target_recovery_height": recovery_height,
                                "command": [float(v) for v in command],
                            }
                        )
                    else:
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
                elif method in ff_methods:
                    local_target = world_to_body_2d(target - pos[:2], yaw)
                    target_command = _target_command(
                        local_target,
                        command_dim,
                        args.command_gain,
                        args.command_max,
                        args.yaw_command_gain,
                    )
                    obs_vec = _obs_for_feature(obs)[0]
                    if method == "viability_gated_scaling_command":
                        nominal_scale = float(np.clip(args.vgs_nominal_scale, 0.0, 1.0))
                        candidates = np.unique(
                            np.clip(
                                np.stack(
                                    [
                                        nominal_scale * target_command,
                                        target_command,
                                    ]
                                ),
                                -args.command_max,
                                args.command_max,
                            ).astype(np.float32),
                            axis=0,
                        )
                    elif method == "task_value_scaling_command":
                        tv_scales = np.asarray(
                            [float(value) for value in args.tv_command_scales.split(",") if value.strip()],
                            dtype=np.float32,
                        )
                        if tv_scales.size == 0:
                            raise ValueError("--tv_command_scales must contain at least one scale.")
                        if not np.any(np.isclose(tv_scales, 1.0)):
                            tv_scales = np.append(tv_scales, 1.0)
                        tv_scales = np.unique(np.clip(tv_scales, 0.0, 1.0))
                        candidates = np.unique(
                            np.clip(
                                tv_scales[:, None] * target_command[None, :],
                                -args.command_max,
                                args.command_max,
                            ).astype(np.float32),
                            axis=0,
                        )
                    else:
                        candidates = _ff_candidate_commands(target_command)
                    execution_aligned = method in {
                        "aligned_efficiency_feedforward_command",
                        "aligned_proxy_feedforward_command",
                        "proxy_efficiency_feedforward_command",
                        "task_value_feedforward_command",
                        "task_value_scaling_command",
                        "viability_gated_scaling_command",
                    }
                    if execution_aligned and method not in {
                        "task_value_scaling_command",
                        "viability_gated_scaling_command",
                    }:
                        aligned_gain = float(np.clip(args.ff_blend, 0.0, 1.0))
                        if args.ff_anneal_distance > 0.0:
                            aligned_gain *= float(
                                np.clip(distance / args.ff_anneal_distance, 0.0, 1.0)
                            )
                        # Score exactly the bounded commands that can be executed,
                        # rather than scoring full candidates and blending afterward.
                        candidates = np.clip(
                            target_command[None, :]
                            + aligned_gain * (candidates - target_command[None, :]),
                            -args.command_max,
                            args.command_max,
                        ).astype(np.float32)
                        candidates = np.unique(candidates, axis=0)
                    preds, pred_stds = predict_response_stats(obs_vec, candidates)
                    names = response_stats["output_names"]
                    j = {n: i for i, n in enumerate(names)}
                    pred_dx = preds[:, j["delta_x"]]
                    pred_dy = preds[:, j["delta_y"]]
                    cost_output = (
                        "mechanical_power"
                        if method in {
                            "mechanical_feedforward_command",
                            "mechanical_efficiency_feedforward_command",
                            "aligned_efficiency_feedforward_command",
                            "viability_gated_scaling_command",
                        }
                        else "energy"
                    )
                    if cost_output not in j:
                        raise ValueError(
                            f"Response model does not provide required cost output '{cost_output}'."
                        )
                    pred_energy = preds[:, j[cost_output]]
                    pred_min_h = preds[:, j["min_height"]]
                    pred_min_h_std = pred_stds[:, j["min_height"]]
                    uncertainty_k = (
                        response_stats["height_conformal_k"]
                        if args.ff_use_model_conformal_k
                        else max(0.0, args.ff_height_uncertainty_k)
                    )
                    pred_min_h_gate = pred_min_h - uncertainty_k * pred_min_h_std
                    pred_tilt = preds[:, j["max_tilt"]] if "max_tilt" in j else np.zeros(len(candidates))
                    pred_ang_speed = (
                        preds[:, j["max_ang_speed"]] if "max_ang_speed" in j else np.zeros(len(candidates))
                    )
                    local_xy = np.asarray(local_target[:2], dtype=np.float64)
                    pred_dist = np.sqrt(
                        (local_xy[0] - pred_dx) ** 2 + (local_xy[1] - pred_dy) ** 2
                    )
                    pred_mean_h = preds[:, j["mean_height"]]
                    pred_progress = distance - pred_dist
                    if method in {
                            "mechanical_efficiency_feedforward_command",
                            "aligned_efficiency_feedforward_command",
                            "proxy_efficiency_feedforward_command",
                            "viability_gated_scaling_command",
                    }:
                        # Horizon duration is constant across candidates, so
                        # power/progress ranks predicted mechanical work per meter.
                        pred_energy = pred_energy / np.maximum(pred_progress, 0.02)
                        cost_output = f"{cost_output}_per_progress"
                    direct_idx = int(
                        np.argmin(np.linalg.norm(candidates - target_command[None, :], axis=1))
                    )
                    task_constraint = np.ones(len(candidates), dtype=bool)
                    task_work = np.zeros(len(candidates), dtype=np.float32)
                    task_time = np.zeros(len(candidates), dtype=np.float32)
                    task_success = np.ones(len(candidates), dtype=np.float32)
                    if method in {"task_value_feedforward_command", "task_value_scaling_command"}:
                        task_work, task_time, task_success = predict_task_values(
                            obs_vec, local_xy, distance, candidates
                        )
                        pred_energy = task_work
                        cost_output = "task_remaining_work"
                        task_constraint = (
                            task_success >= task_success[direct_idx] - args.tv_success_margin
                        ) & (task_time <= task_time[direct_idx] * args.tv_time_ratio)
                    height_floor = args.ff_min_height_fraction * max(start_rel_height, 1e-6)
                    rescue_floor = args.ff_rescue_height_fraction * max(start_rel_height, 1e-6)
                    tilt_safe = (args.ff_max_tilt_rad <= 0.0) | (pred_tilt <= args.ff_max_tilt_rad)
                    ang_speed_safe = (args.ff_max_ang_speed <= 0.0) | (
                        pred_ang_speed <= args.ff_max_ang_speed
                    )
                    stability_safe = tilt_safe & ang_speed_safe
                    height_safe = (pred_min_h_gate >= height_floor) & stability_safe
                    direct_progress = float(pred_progress[direct_idx])
                    direct_safe = bool(
                        pred_min_h_gate[direct_idx] >= rescue_floor and stability_safe[direct_idx]
                    )
                    current_height = relative_height(obs_vec)
                    posture_low = current_height < args.ff_current_height_fraction * max(start_rel_height, 1e-6)
                    mode = "follow"
                    if posture_low and direct_safe:
                        # Posture restoration: the base has sagged below the operating band.
                        # The aggressive direct tracking command restores height reliably, so
                        # compensation is suspended until posture recovers.
                        mode = "restore"
                        best_idx = direct_idx
                        ff_active = False
                    elif not direct_safe:
                        # Predictive height rescue: the model expects the direct command to
                        # drive the base below the safety floor within the horizon. Pick the
                        # candidate with the highest predicted min height among those that
                        # still make non-negative progress.
                        mode = "rescue"
                        rescue_mask = (pred_progress >= 0.0) & stability_safe
                        if not bool(rescue_mask.any()):
                            rescue_mask = pred_progress >= 0.0
                        if not bool(rescue_mask.any()):
                            rescue_mask = np.ones(len(candidates), dtype=bool)
                        best_idx = int(np.argmax(np.where(rescue_mask, pred_min_h_gate, -np.inf)))
                        ff_active = best_idx != direct_idx
                    else:
                        # Feedforward energy compensation: keep at least ff_progress_floor of
                        # the direct command's predicted progress, stay height-safe, and pick
                        # the lowest predicted energy. Activate only if the saving clears the
                        # margin.
                        required = args.ff_progress_floor * max(direct_progress, 0.0)
                        no_height_decline = pred_mean_h >= current_height - args.ff_max_height_drop
                        eligible = (
                            height_safe
                            & no_height_decline
                            & (pred_progress >= required)
                            & task_constraint
                        )
                        eligible[direct_idx] = True
                        cand_energy = np.where(eligible, pred_energy, np.inf)
                        best_idx = int(np.argmin(cand_energy))
                        ff_active = bool(
                            pred_energy[best_idx]
                            < pred_energy[direct_idx] * (1.0 - args.ff_energy_margin_frac)
                        )
                        if not ff_active:
                            best_idx = direct_idx
                    if ff_active:
                        if execution_aligned:
                            command = candidates[best_idx].astype(np.float32)
                        else:
                            blend = float(np.clip(args.ff_blend, 0.0, 1.0))
                            if args.ff_anneal_distance > 0.0:
                                blend *= float(np.clip(distance / args.ff_anneal_distance, 0.0, 1.0))
                            command = np.clip(
                                target_command + blend * (candidates[best_idx] - target_command),
                                -args.command_max,
                                args.command_max,
                            ).astype(np.float32)
                        ff_activations += 1
                    else:
                        command = target_command
                    decisions.append(
                        {
                            "ff_active": ff_active,
                            "ff_mode": mode,
                            "current_distance": distance,
                            "command": [float(v) for v in command],
                            "pred_progress_best": float(pred_progress[best_idx]),
                            "pred_progress_direct": direct_progress,
                            "pred_energy_best": float(pred_energy[best_idx]),
                            "pred_energy_direct": float(pred_energy[direct_idx]),
                            "cost_output": cost_output,
                            "pred_min_height_best": float(pred_min_h[best_idx]),
                            "pred_min_height_direct": float(pred_min_h[direct_idx]),
                            "pred_min_height_std_best": float(pred_min_h_std[best_idx]),
                            "pred_min_height_std_direct": float(pred_min_h_std[direct_idx]),
                            "pred_min_height_lcb_best": float(pred_min_h_gate[best_idx]),
                            "pred_min_height_lcb_direct": float(pred_min_h_gate[direct_idx]),
                            "height_uncertainty_k": float(uncertainty_k),
                            "pred_max_tilt_best": float(pred_tilt[best_idx]),
                            "pred_max_tilt_direct": float(pred_tilt[direct_idx]),
                            "pred_max_ang_speed_best": float(pred_ang_speed[best_idx]),
                            "pred_max_ang_speed_direct": float(pred_ang_speed[direct_idx]),
                            "pred_remaining_work_best": float(task_work[best_idx]),
                            "pred_remaining_work_direct": float(task_work[direct_idx]),
                            "pred_remaining_time_best": float(task_time[best_idx]),
                            "pred_remaining_time_direct": float(task_time[direct_idx]),
                            "pred_success_best": float(task_success[best_idx]),
                            "pred_success_direct": float(task_success[direct_idx]),
                        }
                    )
                elif method == "completion_mpc_command":
                    forced_scale = None
                    if forced_scale_schedule is not None:
                        forced_scale = float(
                            forced_scale_schedule[high_step]
                            if high_step < len(forced_scale_schedule)
                            else 1.0
                        )
                    if mpc_debug_timing and not decisions and forced_scale is None:
                        print("[MPC timing] entering first score", file=sys.stderr, flush=True)
                    local_target = world_to_body_2d(target - pos[:2], yaw)
                    target_command = _target_command(
                        local_target,
                        command_dim,
                        args.command_gain,
                        args.command_max,
                        args.yaw_command_gain,
                    )
                    obs_vec = _obs_for_feature(obs)[0]
                    if forced_scale is None:
                        mpc_start_time = time.perf_counter()
                        mpc_result = score_completion_mpc(
                            obs_vec, local_target[:2], start_rel_height
                        )
                        mpc_inference_ms = 1000.0 * (time.perf_counter() - mpc_start_time)
                        if not decisions:
                            print(
                                f"completion MPC first-decision inference: {mpc_inference_ms:.2f} ms "
                                f"for {len(mpc_sequences)} sequences",
                                flush=True,
                            )
                        executed_scale = float(mpc_result["sequence"][0])
                        if mpc_result["active"]:
                            mpc_activations += 1
                        decision_record = {
                            "mpc_active": bool(mpc_result["active"]),
                            "selected_sequence": [
                                float(value) for value in mpc_result["sequence"]
                            ],
                            "predicted_work_best": mpc_result["predicted_work_best"],
                            "predicted_work_direct": mpc_result["predicted_work_direct"],
                            "predicted_work_std_best": mpc_result["predicted_work_std_best"],
                            "predicted_time_best": mpc_result["predicted_time_best"],
                            "predicted_time_direct": mpc_result["predicted_time_direct"],
                            "predicted_success_best": mpc_result["predicted_success_best"],
                            "predicted_success_direct": mpc_result["predicted_success_direct"],
                            "predicted_min_height_lcb_best": mpc_result[
                                "predicted_min_height_lcb_best"
                            ],
                            "prefix_calibrated": mpc_result["prefix_calibrated"],
                            "predicted_progress_best": mpc_result[
                                "predicted_progress_best"
                            ],
                            "predicted_progress_direct": mpc_result[
                                "predicted_progress_direct"
                            ],
                            "eligible_count": mpc_result["eligible_count"],
                            "inference_ms": mpc_inference_ms,
                        }
                        if args.collect_mpc_dagger:
                            decision_record["observation"] = obs_vec.astype(float).tolist()
                    else:
                        executed_scale = float(np.clip(forced_scale, 0.0, 1.0))
                        decision_record = {
                            "forced_replay": True,
                            "selected_sequence": [executed_scale],
                        }
                        if dagger_query_step is not None and high_step == dagger_query_step:
                            if dagger_sequence is None:
                                raise ValueError("A DAgger query requires dagger_sequence.")
                            query_result = score_completion_mpc(
                                obs_vec,
                                local_target[:2],
                                start_rel_height,
                                return_terminal_inputs=True,
                            )
                            matches = np.flatnonzero(
                                np.all(np.isclose(mpc_sequences, dagger_sequence[None, :]), axis=1)
                            )
                            if not len(matches):
                                raise ValueError(
                                    f"DAgger sequence {dagger_sequence.tolist()} is outside the MPC library."
                                )
                            dagger_predicted_inputs = query_result[
                                "terminal_value_inputs"
                            ][:, int(matches[0]), :].copy()
                            dagger_predicted_prefix = {
                                name: values[:, int(matches[0])].copy()
                                for name, values in query_result["prefix_predictions"].items()
                            }
                            dagger_query_observation = obs_vec.copy()
                            dagger_query_local_target = np.asarray(
                                local_target[:2], dtype=np.float32
                            ).copy()
                            dagger_query_work_start = mechanical_energy_j
                            dagger_query_time_start = elapsed_time_s
                            dagger_query_distance = distance
                            dagger_prefix_min_relative_height = relative_height(obs_vec)
                            if reference_query_observation is not None:
                                dagger_replay_observation_l2 = float(
                                    np.linalg.norm(
                                        dagger_query_observation
                                        - np.asarray(reference_query_observation, dtype=np.float32)
                                    )
                                )
                    command = np.clip(
                        target_command * executed_scale,
                        -args.command_max,
                        args.command_max,
                    ).astype(np.float32)
                    decision_record["executed_scale"] = executed_scale
                    decision_record["command"] = [float(value) for value in command]
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
                elif method == "paired_scale_selector_command":
                    local_target = world_to_body_2d(target - pos[:2], yaw)
                    target_command = _target_command(
                        local_target,
                        command_dim,
                        args.command_gain,
                        args.command_max,
                        args.yaw_command_gain,
                    )
                    if selected_policy_scale is None:
                        selection_probability = predict_scale_selection(
                            _obs_for_feature(obs)[0], local_target[:2], distance
                        )
                        selected_policy_scale = (
                            scale_selector["selected_scale"]
                            if selection_probability >= scale_selector["threshold"]
                            else scale_selector["fallback_scale"]
                        )
                        decisions.append(
                            {
                                "selection_probability": selection_probability,
                                "selection_threshold": scale_selector["threshold"],
                                "selected_scale": selected_policy_scale,
                                "committed_for_episode": True,
                            }
                        )
                    command = target_command * selected_policy_scale
                elif method == "scaled_target_command" or method.startswith("scaled_target_command_"):
                    local_target = world_to_body_2d(target - pos[:2], yaw)
                    command = _target_command(
                        local_target, command_dim, args.command_gain, args.command_max, args.yaw_command_gain
                    )
                    if method.startswith("scaled_target_command_"):
                        scale = float(method.rsplit("_", 1)[1]) / 100.0
                    else:
                        scale = args.fixed_command_scale
                    command *= float(np.clip(scale, 0.0, 1.0))
                elif method == "reactive_governor_command":
                    local_target = world_to_body_2d(target - pos[:2], yaw)
                    command = _target_command(
                        local_target, command_dim, args.command_gain, args.command_max, args.yaw_command_gain
                    )
                    obs_vec = _obs_for_feature(obs)[0]
                    current_rel_height = relative_height(obs_vec)
                    q_now = np.asarray(state["base_quat"][0], dtype=np.float64)
                    current_tilt = float(
                        np.arccos(np.clip(1.0 - 2.0 * (q_now[1] ** 2 + q_now[2] ** 2), -1.0, 1.0))
                    )
                    reactive = (
                        current_rel_height < args.reactive_height_fraction * max(start_rel_height, 1e-6)
                        or current_tilt > args.reactive_tilt_rad
                    )
                    if reactive:
                        command *= float(np.clip(args.reactive_command_scale, 0.0, 1.0))
                    decisions.append(
                        {
                            "reactive_active": bool(reactive),
                            "current_relative_height": current_rel_height,
                            "current_tilt": current_tilt,
                        }
                    )
                elif method == "random_command":
                    command = rng.normal(0.0, args.random_command_std, size=command_dim).astype(np.float32)
                    command = np.clip(command, -args.command_max, args.command_max)
                elif method == "zero_command":
                    command = np.zeros(command_dim, dtype=np.float32)
                else:
                    raise ValueError(f"Unknown method: {method}")

                if args.store_value_samples:
                    sample_local_target = world_to_body_2d(target - pos[:2], yaw)
                    value_samples.append(
                        {
                            "observation": _obs_for_feature(obs)[0].astype(float).tolist(),
                            "local_target": np.asarray(sample_local_target[:2], dtype=float).tolist(),
                            "command": np.asarray(command, dtype=float).tolist(),
                            "distance": distance,
                            "work_so_far_j": mechanical_energy_j,
                            "time_so_far_s": elapsed_time_s,
                        }
                    )

                command_batch = np.tile(command, (args.num_envs, 1))
                for _low in range(args.execution_horizon):
                    with torch.inference_mode():
                        policy_obs = _clone_obs_with_command(obs, command_batch, command_slice)
                        actions = policy(policy_obs)
                        if agent_cfg.clip_actions is not None:
                            actions = torch.clamp(actions, -agent_cfg.clip_actions, agent_cfg.clip_actions)
                    obs, _rewards, dones, _extras = rl_env.step(actions)
                    action_np = _to_numpy(actions)[0]
                    low_state = state_reader.get_robot_state()
                    joint_vel = low_state["joint_vel"][0]
                    joint_effort = low_state["joint_effort"][0]
                    q = np.asarray(low_state["base_quat"][0], dtype=np.float64)
                    tilt = float(np.arccos(np.clip(1.0 - 2.0 * (q[1] ** 2 + q[2] ** 2), -1.0, 1.0)))
                    max_tilt = max(max_tilt, tilt)
                    max_ang_speed = max(
                        max_ang_speed, float(np.linalg.norm(low_state["base_ang_vel"][0]))
                    )
                    if (
                        dagger_query_step is not None
                        and dagger_query_step
                        <= high_step
                        < dagger_query_step + int(args.mpc_planning_steps)
                    ):
                        dagger_prefix_min_relative_height = min(
                            dagger_prefix_min_relative_height,
                            relative_height(_obs_for_feature(obs)[0]),
                        )
                        dagger_prefix_max_tilt = max(dagger_prefix_max_tilt, tilt)
                        dagger_prefix_max_ang_speed = max(
                            dagger_prefix_max_ang_speed,
                            float(np.linalg.norm(low_state["base_ang_vel"][0])),
                        )
                    torque_dim = min(len(joint_effort), len(joint_vel))
                    joint_power = np.asarray(joint_effort[:torque_dim]) * np.asarray(
                        joint_vel[:torque_dim]
                    )
                    mechanical_energy_j += float(np.sum(np.abs(joint_power))) * control_dt
                    positive_work_j += float(np.sum(np.maximum(joint_power, 0.0))) * control_dt
                    negative_work_j += float(np.sum(np.maximum(-joint_power, 0.0))) * control_dt
                    elapsed_time_s += control_dt
                    current_low_pos = np.asarray(low_state["base_pos"][0, :2], dtype=np.float64)
                    path_length_m += float(np.linalg.norm(current_low_pos - previous_low_pos))
                    previous_low_pos = current_low_pos
                    min_dim = min(len(action_np), len(joint_vel))
                    energy_total += float(np.mean(np.abs(action_np[:min_dim] * joint_vel[:min_dim])))
                    energy_steps += 1
                    if bool(_to_numpy(dones).reshape(-1)[0]):
                        terminated_early = True
                        break
                if (
                    dagger_query_step is not None
                    and high_step == dagger_query_step + int(args.mpc_planning_steps) - 1
                ):
                    dagger_terminal_work_start = mechanical_energy_j
                    dagger_terminal_time_start = elapsed_time_s
                    terminal_state = state_reader.get_robot_state()
                    terminal_pos = terminal_state["base_pos"][0]
                    terminal_yaw = float(quat_to_yaw(terminal_state["base_quat"][0]))
                    terminal_local_target = world_to_body_2d(
                        target - terminal_pos[:2], terminal_yaw
                    )
                    terminal_distance = float(np.linalg.norm(target - terminal_pos[:2]))
                    terminal_obs = _obs_for_feature(obs)[0].copy()
                    terminal_command = _target_command(
                        terminal_local_target,
                        command_dim,
                        args.command_gain,
                        args.command_max,
                        args.yaw_command_gain,
                    )
                    lo_tv, hi_tv = task_value_stats["command_slice"]
                    terminal_obs[lo_tv:hi_tv] = terminal_command[: hi_tv - lo_tv]
                    dagger_actual_terminal_input = np.concatenate(
                        [
                            terminal_obs,
                            np.asarray(terminal_local_target[:2], dtype=np.float32),
                            np.asarray([terminal_distance], dtype=np.float32),
                        ]
                    )
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
                "mechanical_energy_j": mechanical_energy_j,
                "positive_mechanical_work_j": positive_work_j,
                "negative_mechanical_work_j": negative_work_j,
                "mean_mechanical_power_w": mechanical_energy_j / max(elapsed_time_s, 1e-9),
                "elapsed_time_s": elapsed_time_s,
                "path_length_m": path_length_m,
                "energy_per_meter_j_m": mechanical_energy_j / max(path_length_m, 1e-9),
                "robot_mass_kg": robot_mass,
                "cost_of_transport": mechanical_energy_j
                / max(robot_mass * 9.81 * path_length_m, 1e-9),
                "max_tilt": max_tilt,
                "max_ang_speed": max_ang_speed,
                "num_commands_used": (
                    len(selected)
                    if method in {"skill_command", "guarded_skill_command", "learned_guarded_skill_command"}
                    else (
                        ff_activations
                        if method in ff_methods
                        else (mpc_activations if method == "completion_mpc_command" else 0)
                    )
                ),
                "selected_archive_indices": selected,
                "decisions": decisions,
                "terminated_early": terminated_early,
            }
            if args.store_positions:
                result["positions"] = positions
                result["target_position"] = target.tolist()
            if args.store_value_samples:
                for sample in value_samples:
                    sample["remaining_work_j"] = mechanical_energy_j - sample.pop("work_so_far_j")
                    sample["remaining_time_s"] = elapsed_time_s - sample.pop("time_so_far_s")
                    sample["success"] = bool(result["success"])
                result["value_samples"] = value_samples
            if dagger_predicted_inputs is not None:
                if dagger_terminal_work_start is None:
                    dagger_terminal_work_start = mechanical_energy_j
                    dagger_terminal_time_start = elapsed_time_s
                result["dagger_sample"] = {
                    "predicted_terminal_inputs": dagger_predicted_inputs.astype(float).tolist(),
                    "actual_terminal_input": (
                        dagger_actual_terminal_input.astype(float).tolist()
                        if dagger_actual_terminal_input is not None
                        else None
                    ),
                    "query_observation": dagger_query_observation.astype(float).tolist(),
                    "query_local_target": dagger_query_local_target.astype(float).tolist(),
                    "query_start_relative_height": start_rel_height,
                    "replay_observation_l2": dagger_replay_observation_l2,
                    "predicted_prefix": {
                        name: values.astype(float).tolist()
                        for name, values in dagger_predicted_prefix.items()
                    },
                    "prefix_work_j": (
                        dagger_terminal_work_start - dagger_query_work_start
                    ),
                    "prefix_time_s": (
                        dagger_terminal_time_start - dagger_query_time_start
                    ),
                    "prefix_progress_m": (
                        dagger_query_distance
                        - float(dagger_actual_terminal_input[-1])
                        if dagger_actual_terminal_input is not None
                        else 0.0
                    ),
                    "prefix_min_relative_height": dagger_prefix_min_relative_height,
                    "prefix_max_tilt": dagger_prefix_max_tilt,
                    "prefix_max_ang_speed": dagger_prefix_max_ang_speed,
                    "remaining_work_j": mechanical_energy_j - dagger_terminal_work_start,
                    "remaining_time_s": elapsed_time_s - dagger_terminal_time_start,
                    "success": bool(result["success"]),
                    "prefix_terminated": bool(
                        terminated_early
                        and elapsed_time_s <= dagger_terminal_time_start + 1e-9
                    ),
                }
            return result

        records: Dict[str, Dict[str, List[Dict[str, Any]]]] = {
            f"{target[0]:g},{target[1]:g}": {method: [] for method in methods}
            for target in targets
        }
        if args.paired_method_resets:
            # A bare reset is insufficient to match the observation/action
            # history seen after ordinary episodes.  Discard one full rollout
            # so the first evaluated method has the same reset path and history
            # lifecycle as every subsequent method in a paired comparison.
            rollout("direct_target_command", np.asarray(targets[0], dtype=np.float64), args.seed)
        for target_index, target in enumerate(targets):
            target_arr = np.asarray(target, dtype=np.float64)
            key = f"{target[0]:g},{target[1]:g}"
            for trial in range(args.num_trials):
                trial_seed = args.seed + 1000 * target_index + trial
                for method in methods:
                    rec = rollout(method, target_arr, trial_seed)
                    rec["trial"] = trial
                    records[key][method].append(rec)

        mpc_dagger_samples: List[Dict[str, Any]] = []
        if args.collect_mpc_dagger:
            planning_steps = int(args.mpc_planning_steps)
            query_count = max(1, int(args.dagger_queries_per_episode))
            for target_index, target in enumerate(targets):
                target_arr = np.asarray(target, dtype=np.float64)
                key = f"{target[0]:g},{target[1]:g}"
                for trial, on_policy in enumerate(records[key]["completion_mpc_command"]):
                    trial_seed = args.seed + 1000 * target_index + trial
                    decisions = on_policy.get("decisions", [])
                    scale_history = [
                        float(decision["executed_scale"])
                        for decision in decisions
                        if "executed_scale" in decision
                    ]
                    if not scale_history:
                        continue
                    upper = max(0, len(scale_history) - 1)
                    query_steps = sorted(
                        {
                            int(round((index + 1) * upper / (query_count + 1)))
                            for index in range(query_count)
                        }
                    )
                    for query_step in query_steps:
                        reference_observation = np.asarray(
                            decisions[query_step]["observation"], dtype=np.float32
                        )
                        candidates: List[tuple[str, np.ndarray]] = []
                        for label, fixed_sequence in dagger_sequence_specs:
                            sequence = (
                                np.asarray(
                                    decisions[query_step]["selected_sequence"],
                                    dtype=np.float32,
                                )
                                if fixed_sequence is None
                                else fixed_sequence.copy()
                            )
                            if len(sequence) != planning_steps:
                                continue
                            if any(np.all(np.isclose(sequence, prior)) for _, prior in candidates):
                                continue
                            candidates.append((label, sequence))
                        for candidate_label, sequence in candidates:
                            schedule = scale_history[:query_step] + sequence.astype(float).tolist()
                            branch = rollout(
                                "completion_mpc_command",
                                target_arr,
                                trial_seed,
                                forced_scale_schedule=schedule,
                                dagger_query_step=query_step,
                                dagger_sequence=sequence,
                                reference_query_observation=reference_observation,
                            )
                            sample = branch.get("dagger_sample")
                            if sample is None:
                                continue
                            sample.update(
                                {
                                    "target": [float(value) for value in target],
                                    "trial": trial,
                                    "trial_seed": trial_seed,
                                    "query_step": query_step,
                                    "candidate_label": candidate_label,
                                    "candidate_sequence": sequence.astype(float).tolist(),
                                }
                            )
                            mpc_dagger_samples.append(sample)

            replay_errors = np.asarray(
                [
                    sample["replay_observation_l2"]
                    for sample in mpc_dagger_samples
                    if sample["replay_observation_l2"] is not None
                ],
                dtype=np.float64,
            )
            print(
                json.dumps(
                    {
                        "mpc_dagger_samples": len(mpc_dagger_samples),
                        "replay_observation_l2_mean": (
                            float(replay_errors.mean()) if len(replay_errors) else None
                        ),
                        "replay_observation_l2_max": (
                            float(replay_errors.max()) if len(replay_errors) else None
                        ),
                    },
                    indent=2,
                ),
                flush=True,
            )

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
                "stress_static_friction": args.stress_static_friction,
                "stress_dynamic_friction": args.stress_dynamic_friction,
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
                "fixed_command_scale": args.fixed_command_scale,
                "vgs_nominal_scale": args.vgs_nominal_scale,
                "reactive_height_fraction": args.reactive_height_fraction,
                "reactive_tilt_rad": args.reactive_tilt_rad,
                "reactive_command_scale": args.reactive_command_scale,
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
                "freeze_terrain_curriculum": freeze_terrain_curriculum,
                "paired_method_resets": args.paired_method_resets,
                "store_value_samples": args.store_value_samples,
                "response_model": args.response_model,
                "task_value_model": args.task_value_model,
                "scale_selector_model": args.scale_selector_model,
                "macro_transition_model": args.macro_transition_model,
                "mpc_scales": args.mpc_scales,
                "mpc_planning_steps": args.mpc_planning_steps,
                "mpc_uncertainty_k": args.mpc_uncertainty_k,
                "mpc_work_margin_frac": args.mpc_work_margin_frac,
                "mpc_prefix_calibrator": args.mpc_prefix_calibrator,
                "mpc_prefix_uncertainty_k": args.mpc_prefix_uncertainty_k,
                "mpc_progress_ratio": args.mpc_progress_ratio,
                "collect_mpc_dagger": args.collect_mpc_dagger,
                "dagger_queries_per_episode": args.dagger_queries_per_episode,
                "dagger_candidate_sequences": args.dagger_candidate_sequences,
                "tv_success_margin": args.tv_success_margin,
                "tv_time_ratio": args.tv_time_ratio,
                "tv_command_scales": args.tv_command_scales,
                "ff_progress_floor": args.ff_progress_floor,
                "ff_min_height_fraction": args.ff_min_height_fraction,
                "ff_rescue_height_fraction": args.ff_rescue_height_fraction,
                "ff_energy_margin_frac": args.ff_energy_margin_frac,
                "ff_current_height_fraction": args.ff_current_height_fraction,
                "ff_max_height_drop": args.ff_max_height_drop,
                "ff_max_tilt_rad": args.ff_max_tilt_rad,
                "ff_max_ang_speed": args.ff_max_ang_speed,
                "ff_blend": args.ff_blend,
                "ff_anneal_distance": args.ff_anneal_distance,
                "ff_height_scan_slice": args.ff_height_scan_slice,
                "ff_height_scan_offset": args.ff_height_scan_offset,
                "ff_height_uncertainty_k": args.ff_height_uncertainty_k,
                "ff_use_model_conformal_k": args.ff_use_model_conformal_k,
            },
            "summary": summary,
            "records": records,
        }
        if args.collect_mpc_dagger:
            output["mpc_dagger_samples"] = mpc_dagger_samples
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(output, indent=2))
        print(json.dumps({"summary": summary}, indent=2), flush=True)
        rl_env.close()
        # Isaac Sim can hang in SimulationApp.close() after headless evaluation,
        # retaining several GB of GPU memory even though the result is complete.
        # The output is durable at this point; terminate the standalone worker.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    except Exception:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    finally:
        sim_app.close()


if __name__ == "__main__":
    main()
