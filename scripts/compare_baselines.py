"""Compare skill composition against simple target-reaching baselines in Isaac Lab.

This script is intended to run through the Isaac Lab launcher:

    ./isaaclab.sh -p /path/to/skill_discover/scripts/compare_baselines.py \
        --task Isaac-Ant-v0 \
        --skill_library /path/to/skill_discover/outputs/skill_library_ant.pkl \
        --targets "3,0;3,2;0,3;-2,0" \
        --num_trials 5 --headless

Baselines are deliberately simple and V1-appropriate:
    * skill_greedy: discovered skills selected by one-step outcome prediction.
    * skill_mpc: discovered skills selected by receding-horizon skill MPC.
    * random_skill: random discovered skill replay.
    * random_action: smoothed Gaussian low-level actions for the same step budget.
    * zero_action: no-op actions for the same step budget.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_targets(raw: str) -> List[List[float]]:
    """Parse 'x,y;x,y;...' into target coordinate pairs."""
    targets: List[List[float]] = []
    for item in raw.split(";"):
        item = item.strip()
        if not item:
            continue
        parts = [p.strip() for p in item.split(",")]
        if len(parts) != 2:
            raise ValueError(f"Invalid target '{item}'. Expected 'x,y'.")
        targets.append([float(parts[0]), float(parts[1])])
    if not targets:
        raise ValueError("At least one target is required.")
    return targets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare skill composition with simple baselines.")
    parser.add_argument("--task", type=str, default="Isaac-Ant-v0")
    parser.add_argument("--skill_library", type=str, required=True)
    parser.add_argument("--targets", type=str, default="3,0;3,2;0,3;-2,0")
    parser.add_argument("--num_trials", type=int, default=5)
    parser.add_argument("--num_envs", type=int, default=1, help="Only env 0 is evaluated.")
    parser.add_argument("--target_threshold", type=float, default=0.5)
    parser.add_argument("--max_high_level_steps", type=int, default=20)
    parser.add_argument("--lambda_energy", type=float, default=0.05)
    parser.add_argument("--lambda_yaw", type=float, default=0.1)
    parser.add_argument("--mpc_horizon", type=int, default=3)
    parser.add_argument("--random_action_std", type=float, default=0.5)
    parser.add_argument("--action_smoothing", type=float, default=0.2)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="outputs/baseline_comparison.json")
    return parser.parse_args()


def _state_xy_yaw(env: Any) -> tuple[Any, Any, float]:
    import numpy as np

    from skill_discovery.utils.math_utils import quat_to_yaw

    state = env.get_robot_state()
    pos = state["base_pos"][0]
    yaw = float(quat_to_yaw(state["base_quat"][0]))
    return pos, state, yaw


def _distance_to_target(env: Any, target_xy: Any) -> float:
    import numpy as np

    pos, _, _ = _state_xy_yaw(env)
    return float(np.linalg.norm(target_xy - pos[:2]))


def _run_skill_mpc(env: Any, library: Any, target_xy: Any, args: argparse.Namespace) -> Dict[str, Any]:
    import numpy as np

    from skill_discovery.control.skill_mpc import SkillMPC

    env.reset()
    planner = SkillMPC(
        library,
        horizon=args.mpc_horizon,
        lambda_energy=args.lambda_energy,
        lambda_yaw=args.lambda_yaw,
    )
    skill_sequence: List[int] = []
    energy_total, energy_steps = 0.0, 0

    for _ in range(args.max_high_level_steps):
        pos, state, yaw = _state_xy_yaw(env)
        distance = float(np.linalg.norm(target_xy - pos[:2]))
        if distance < args.target_threshold:
            break

        seq, _ = planner.plan(
            {"x": float(pos[0]), "y": float(pos[1]), "yaw": yaw},
            {"x": float(target_xy[0]), "y": float(target_xy[1])},
        )
        if not seq:
            break
        skill = library.get_skill(seq[0])
        skill_sequence.append(skill.skill_id)

        for action in skill.action_sequence:
            result = env.step(np.tile(action, (env.num_envs, 1)))
            joint_vel = env.get_robot_state()["joint_vel"][0]
            min_dim = min(len(action), len(joint_vel))
            energy_total += float(np.mean(np.abs(action[:min_dim] * joint_vel[:min_dim])))
            energy_steps += 1
            if result.dones[0]:
                break

    final_distance = _distance_to_target(env, target_xy)
    return {
        "success": final_distance < args.target_threshold,
        "final_distance": final_distance,
        "energy_proxy": energy_total / max(energy_steps, 1),
        "num_skills_used": len(skill_sequence),
        "skill_sequence": skill_sequence,
    }


def _run_random_skill(env: Any, library: Any, target_xy: Any, args: argparse.Namespace, seed: int) -> Dict[str, Any]:
    import numpy as np

    rng = np.random.default_rng(seed)
    env.reset()
    skill_ids = library.skill_ids
    skill_sequence: List[int] = []
    energy_total, energy_steps = 0.0, 0

    for _ in range(args.max_high_level_steps):
        if _distance_to_target(env, target_xy) < args.target_threshold:
            break
        sid = int(rng.choice(skill_ids))
        skill = library.get_skill(sid)
        skill_sequence.append(sid)
        for action in skill.action_sequence:
            result = env.step(np.tile(action, (env.num_envs, 1)))
            joint_vel = env.get_robot_state()["joint_vel"][0]
            min_dim = min(len(action), len(joint_vel))
            energy_total += float(np.mean(np.abs(action[:min_dim] * joint_vel[:min_dim])))
            energy_steps += 1
            if result.dones[0]:
                break

    final_distance = _distance_to_target(env, target_xy)
    return {
        "success": final_distance < args.target_threshold,
        "final_distance": final_distance,
        "energy_proxy": energy_total / max(energy_steps, 1),
        "num_skills_used": len(skill_sequence),
        "skill_sequence": skill_sequence,
    }


def _run_low_level_baseline(
    env: Any,
    target_xy: Any,
    args: argparse.Namespace,
    total_low_level_steps: int,
    mode: str,
    seed: int,
) -> Dict[str, Any]:
    import numpy as np

    from skill_discovery.exploration.random_policy import RandomExplorationPolicy

    env.reset()
    energy_total, energy_steps = 0.0, 0

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

    obs = np.zeros((env.num_envs, 1), dtype=np.float32)
    for _ in range(total_low_level_steps):
        if _distance_to_target(env, target_xy) < args.target_threshold:
            break
        if policy is None:
            actions = np.zeros((env.num_envs, env.action_dim), dtype=np.float32)
        else:
            actions = policy.act(obs)
        result = env.step(actions)
        obs = result.obs
        joint_vel = env.get_robot_state()["joint_vel"][0]
        action0 = actions[0]
        min_dim = min(len(action0), len(joint_vel))
        energy_total += float(np.mean(np.abs(action0[:min_dim] * joint_vel[:min_dim])))
        energy_steps += 1
        if result.dones[0]:
            break

    final_distance = _distance_to_target(env, target_xy)
    return {
        "success": final_distance < args.target_threshold,
        "final_distance": final_distance,
        "energy_proxy": energy_total / max(energy_steps, 1),
        "num_skills_used": 0,
        "skill_sequence": [],
    }


def _summarize(records: List[Dict[str, Any]]) -> Dict[str, float]:
    import numpy as np

    return {
        "success_rate": float(np.mean([r["success"] for r in records])),
        "average_final_distance": float(np.mean([r["final_distance"] for r in records])),
        "average_energy_proxy": float(np.mean([r["energy_proxy"] for r in records])),
        "average_num_skills_used": float(np.mean([r["num_skills_used"] for r in records])),
    }


def main() -> None:
    args = parse_args()

    from skill_discovery.envs.isaac_env_wrapper import IsaacEnvWrapper, launch_app

    sim_app = launch_app(headless=args.headless)

    try:
        import numpy as np

        from skill_discovery.control.skill_composer import GreedySkillComposer
        from skill_discovery.library.skill_library import SkillLibrary
        from skill_discovery.utils.logging import get_logger
        from skill_discovery.utils.math_utils import set_global_seed

        logger = get_logger("compare")
        set_global_seed(args.seed)
        targets = parse_targets(args.targets)

        library = SkillLibrary.load(args.skill_library)
        skill_horizon = int(np.median([len(library.get_skill(sid).action_sequence) for sid in library.skill_ids]))
        total_low_level_steps = args.max_high_level_steps * skill_horizon

        env = IsaacEnvWrapper.create(args.task, num_envs=args.num_envs, device=args.device)
        greedy = GreedySkillComposer(
            library,
            target_threshold=args.target_threshold,
            max_high_level_steps=args.max_high_level_steps,
            lambda_energy=args.lambda_energy,
        )

        methods = ["skill_greedy", "skill_mpc", "random_skill", "random_action", "zero_action"]
        all_records: Dict[str, Dict[str, List[Dict[str, Any]]]] = {
            f"{target[0]:g},{target[1]:g}": {method: [] for method in methods} for target in targets
        }

        for target_index, target in enumerate(targets):
            target_xy = np.asarray(target, dtype=np.float64)
            target_key = f"{target[0]:g},{target[1]:g}"
            logger.info("Target %s, %d trials", target_key, args.num_trials)
            for trial in range(args.num_trials):
                trial_seed = args.seed + 1000 * target_index + trial

                res = greedy.rollout(env, target_xy)
                all_records[target_key]["skill_greedy"].append(
                    {
                        "success": bool(res.success),
                        "final_distance": float(res.final_distance),
                        "energy_proxy": float(res.energy_proxy),
                        "num_skills_used": int(res.num_skills_used),
                        "skill_sequence": [int(s) for s in res.skill_sequence],
                    }
                )
                all_records[target_key]["skill_mpc"].append(_run_skill_mpc(env, library, target_xy, args))
                all_records[target_key]["random_skill"].append(
                    _run_random_skill(env, library, target_xy, args, seed=trial_seed)
                )
                all_records[target_key]["random_action"].append(
                    _run_low_level_baseline(
                        env, target_xy, args, total_low_level_steps, "random_action", seed=trial_seed
                    )
                )
                all_records[target_key]["zero_action"].append(
                    _run_low_level_baseline(
                        env, target_xy, args, total_low_level_steps, "zero_action", seed=trial_seed
                    )
                )

        summary = {
            target_key: {method: _summarize(records) for method, records in method_records.items()}
            for target_key, method_records in all_records.items()
        }
        output = {
            "task": args.task,
            "skill_library": args.skill_library,
            "targets": targets,
            "num_trials": args.num_trials,
            "target_threshold": args.target_threshold,
            "max_high_level_steps": args.max_high_level_steps,
            "skill_horizon": skill_horizon,
            "total_low_level_step_budget": total_low_level_steps,
            "summary": summary,
            "records": all_records,
        }
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(output, indent=2))
        print(json.dumps({"summary": summary}, indent=2))
        logger.info("Wrote comparison to %s", out)
        env.close()
    finally:
        sim_app.close()


if __name__ == "__main__":
    main()
