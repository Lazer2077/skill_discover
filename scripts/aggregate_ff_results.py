"""Aggregate the VGFC systematic experiment results into paper-ready numbers."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

GO2 = Path("outputs/isaac_go2_rough_rsl_rl_skills_long")
H1 = Path("outputs/isaac_h1_rough_rsl_rl_skills")
METHODS = "model_feedforward_command,guarded_skill_command,direct_target_command"

FAMILIES = {
    "harder": sorted(GO2.glob("ff_v3_harder_10trials_seed*_positions.json")),
    "moderate": sorted(GO2.glob("ff_v3_moderate_10trials_seed*_positions.json")),
    "holdout": sorted(GO2.glob("ff_v3_holdout_5trials_seed*_positions.json")),
    "h1": sorted(H1.glob("ff_v3_h1_5trials_seed*_positions.json")),
}


def summarize(name: str, inputs: list[Path]) -> dict:
    out = (GO2 if name != "h1" else H1) / f"ff_v3_{name}_summary.json"
    cmd = [
        sys.executable,
        "scripts/summarize_skill_eval_results.py",
        "--inputs",
        *[str(p) for p in inputs],
        "--methods",
        METHODS,
        "--output",
        str(out),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return json.loads(out.read_text())


def fmt(v: dict, nd: int = 4) -> str:
    return f"{v['mean']:.{nd}f} [{v['ci_low']:.{nd}f},{v['ci_high']:.{nd}f}]"


def main() -> None:
    for name, inputs in FAMILIES.items():
        if not inputs:
            print(f"== {name}: NO FILES")
            continue
        d = summarize(name, inputs)
        ew = d["episode_weighted"]
        print(f"\n== {name} ({len(inputs)} files, n={next(iter(ew.values()))['success']['n']}/method)")
        for m in ["model_feedforward_command", "guarded_skill_command", "direct_target_command"]:
            if m not in ew:
                continue
            v = ew[m]
            print(
                f"  {m[:28]:28s} succ {fmt(v['success'],3)}  dist {fmt(v['final_distance'])}  "
                f"energy {fmt(v['energy_proxy'])}  height {fmt(v['final_height'])}  "
                f"term {v['terminated_early']['mean']:.3f}  cmds {v['num_commands_used']['mean']:.1f}"
            )
        if "model_feedforward_command" in ew and "direct_target_command" in ew:
            e_ff = ew["model_feedforward_command"]["energy_proxy"]["mean"]
            e_d = ew["direct_target_command"]["energy_proxy"]["mean"]
            print(f"  -> VGFC energy delta vs direct: {100*(e_ff/e_d-1):+.1f}%")
        if name == "harder":
            print("  per-target (VGFC vs direct):")
            pt = d["per_target"]
            for tgt, mv in pt.items():
                if "model_feedforward_command" not in mv:
                    continue
                f_, d_ = mv["model_feedforward_command"], mv["direct_target_command"]
                print(
                    f"    {tgt:8s} succ {f_['success']['mean']:.3f}/{d_['success']['mean']:.3f}  "
                    f"energy {f_['energy_proxy']['mean']:.3f}/{d_['energy_proxy']['mean']:.3f}  "
                    f"height {f_['final_height']['mean']:.3f}/{d_['final_height']['mean']:.3f}"
                )


if __name__ == "__main__":
    main()
