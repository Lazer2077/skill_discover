"""Evaluate a saved V2 online action set with its state-skill discriminator."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_targets(raw: str) -> List[List[float]]:
    targets: List[List[float]] = []
    for item in raw.split(";"):
        if item.strip():
            x, y = item.split(",")
            targets.append([float(x), float(y)])
    return targets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate saved V2 online skills.")
    parser.add_argument("--task", type=str, default="Isaac-Ant-v0")
    parser.add_argument("--online_action_set", type=str, required=True)
    parser.add_argument("--discriminator", type=str, required=True)
    parser.add_argument("--targets", type=str, default="0.5,0;0.5,0.5;-0.5,0")
    parser.add_argument("--num_trials", type=int, default=3)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--target_threshold", type=float, default=0.3)
    parser.add_argument("--max_high_level_steps", type=int, default=6)
    parser.add_argument("--lambda_energy", type=float, default=0.05)
    parser.add_argument("--lambda_discriminator", type=float, default=1.5)
    parser.add_argument("--lambda_reliability", type=float, default=0.25)
    parser.add_argument("--lambda_no_progress", type=float, default=1.0)
    parser.add_argument("--min_predicted_progress", type=float, default=0.02)
    parser.add_argument("--strict_applicability", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--discriminator_device", type=str, default="cpu")
    parser.add_argument("--output", type=str, default="outputs/online_v2_eval.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from skill_discovery.envs.isaac_env_wrapper import IsaacEnvWrapper, launch_app

    sim_app = launch_app(headless=args.headless)
    try:
        import numpy as np

        from skill_discovery.control.discriminator_skill_composer import DiscriminatorGuidedSkillComposer
        from skill_discovery.learning.state_skill_discriminator import StateSkillDiscriminator
        from skill_discovery.online.online_action_set import OnlineActionSet

        action_set = OnlineActionSet.load(args.online_action_set)
        discriminator = StateSkillDiscriminator.load(args.discriminator, device=args.discriminator_device)
        env = IsaacEnvWrapper.create(args.task, num_envs=args.num_envs, device=args.device)
        composer = DiscriminatorGuidedSkillComposer(
            action_set=action_set,
            discriminator=discriminator,
            target_threshold=args.target_threshold,
            max_high_level_steps=args.max_high_level_steps,
            lambda_energy=args.lambda_energy,
            lambda_discriminator=args.lambda_discriminator,
            lambda_reliability=args.lambda_reliability,
            lambda_no_progress=args.lambda_no_progress,
            min_predicted_progress=args.min_predicted_progress,
            strict_applicability=args.strict_applicability,
        )

        summary = {}
        for target in parse_targets(args.targets):
            target_xy = np.asarray(target, dtype=np.float64)
            key = f"{target[0]:g},{target[1]:g}"
            records = []
            for trial in range(args.num_trials):
                res = composer.rollout(env, target_xy)
                records.append(
                    {
                        "trial": trial,
                        "success": bool(res.success),
                        "final_distance": float(res.final_distance),
                        "energy_proxy": float(res.energy_proxy),
                        "num_skills_used": int(res.num_skills_used),
                        "skill_sequence": [int(s) for s in res.skill_sequence],
                        "applicability": [
                            {
                                "skill_id": int(d.skill_id),
                                "prob": float(d.applicability),
                                "predicted_distance": float(d.predicted_distance),
                                "cost": float(d.cost),
                            }
                            for d in composer.last_decisions
                        ],
                    }
                )
            summary[key] = {
                "success_rate": float(np.mean([r["success"] for r in records])),
                "average_final_distance": float(np.mean([r["final_distance"] for r in records])),
                "average_energy_proxy": float(np.mean([r["energy_proxy"] for r in records])),
                "average_num_skills_used": float(np.mean([r["num_skills_used"] for r in records])),
                "records": records,
            }

        output = {
            "task": args.task,
            "online_action_set": args.online_action_set,
            "discriminator": args.discriminator,
            "num_skills": len(action_set.skills),
            "summary": summary,
        }
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(output, indent=2))
        print(json.dumps(output, indent=2))
        env.close()
    finally:
        sim_app.close()


if __name__ == "__main__":
    main()
