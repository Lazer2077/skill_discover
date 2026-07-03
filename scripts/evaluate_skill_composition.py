"""Evaluate skill composition on a 2D target-reaching task in Isaac Lab.

Usage (inside Isaac Lab):
    ./isaaclab.sh -p scripts/evaluate_skill_composition.py \
        --task Isaac-Ant-v0 \
        --skill_library outputs/skill_library_ant.pkl \
        --target_x 3.0 --target_y 2.0 --num_trials 10 --headless
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate greedy skill composition.")
    parser.add_argument("--task", type=str, default="Isaac-Ant-v0")
    parser.add_argument("--skill_library", type=str, required=True)
    parser.add_argument("--target_x", type=float, default=3.0)
    parser.add_argument("--target_y", type=float, default=2.0)
    parser.add_argument("--num_trials", type=int, default=10)
    parser.add_argument("--num_envs", type=int, default=1, help="Only env 0 is evaluated.")
    parser.add_argument("--target_threshold", type=float, default=0.5)
    parser.add_argument("--max_high_level_steps", type=int, default=20)
    parser.add_argument("--lambda_energy", type=float, default=0.05)
    parser.add_argument("--planner", type=str, default="greedy", choices=["greedy", "mpc"],
                        help="'mpc' logs multi-step plans from SkillMPC before each greedy trial (diagnostic).")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="outputs/composition_eval.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from skill_discovery.envs.isaac_env_wrapper import IsaacEnvWrapper, launch_app

    sim_app = launch_app(headless=args.headless)

    try:
        import numpy as np

        from skill_discovery.control.skill_composer import GreedySkillComposer
        from skill_discovery.control.skill_mpc import SkillMPC
        from skill_discovery.library.skill_library import SkillLibrary
        from skill_discovery.utils.logging import get_logger
        from skill_discovery.utils.math_utils import set_global_seed
        from skill_discovery.utils.plotting import plot_composition_trajectories

        logger = get_logger("evaluate")
        set_global_seed(args.seed)

        library = SkillLibrary.load(args.skill_library)
        logger.info("Loaded skill library with %d skills: %s",
                    len(library.skills), [s["interpretation"] for s in library.summary()])

        env = IsaacEnvWrapper.create(args.task, num_envs=args.num_envs, device=args.device)
        composer = GreedySkillComposer(
            library,
            target_threshold=args.target_threshold,
            max_high_level_steps=args.max_high_level_steps,
            lambda_energy=args.lambda_energy,
        )
        target = np.array([args.target_x, args.target_y])

        if args.planner == "mpc":
            mpc = SkillMPC(library, horizon=3, lambda_energy=args.lambda_energy)
            seq, cost = mpc.plan({"x": 0.0, "y": 0.0, "yaw": 0.0}, {"x": args.target_x, "y": args.target_y})
            logger.info("SkillMPC 3-step plan from origin: %s (predicted cost %.3f)", seq, cost)

        results = []
        for trial in range(args.num_trials):
            res = composer.rollout(env, target)
            results.append(res)
            logger.info(
                "Trial %d: success=%s final_dist=%.3f skills=%s",
                trial, res.success, res.final_distance, res.skill_sequence,
            )

        metrics = {
            "task": args.task,
            "target": [args.target_x, args.target_y],
            "num_trials": args.num_trials,
            "success_rate": float(np.mean([r.success for r in results])),
            "average_final_distance": float(np.mean([r.final_distance for r in results])),
            "average_energy_proxy": float(np.mean([r.energy_proxy for r in results])),
            "average_num_skills_used": float(np.mean([r.num_skills_used for r in results])),
            "skill_sequence_per_trial": [r.skill_sequence for r in results],
        }
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(metrics, indent=2))
        plot_composition_trajectories(
            [r.base_positions[:, :2] for r in results], target,
            out.parent / "plots" / "composition_trajectories.png",
            threshold=args.target_threshold,
        )
        print(json.dumps(metrics, indent=2))
        env.close()
    finally:
        sim_app.close()


if __name__ == "__main__":
    main()
