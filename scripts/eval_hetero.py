"""Register the heterogeneous-terrain Go2 tasks, then run a target script
(the evaluator or the excitation collector) unchanged:

    python scripts/eval_hetero.py <target_script.py> [target args...]

Registration stores only entry-point strings, so nothing from isaaclab is
imported before the target script launches the simulation app; the cfg module
(hetero_env_cfg.py) is resolved lazily by parse_env_cfg afterwards.
"""
import os
import sys
import runpy

import gymnasium as gym

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

_AGENT = (
    "isaaclab_tasks.manager_based.locomotion.velocity.config.go2.agents."
    "rsl_rl_ppo_cfg:UnitreeGo2RoughPPORunnerCfg"
)
for _id, _cls in [
    ("Isaac-Velocity-HeteroFlat-Unitree-Go2-v0", "hetero_env_cfg:HeteroFlatGo2EnvCfg"),
    ("Isaac-Velocity-HeteroRough-Unitree-Go2-v0", "hetero_env_cfg:HeteroRoughGo2EnvCfg"),
    ("Isaac-Velocity-HeteroMid-Unitree-Go2-v0", "hetero_env_cfg:HeteroMidGo2EnvCfg"),
]:
    gym.register(
        id=_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={"env_cfg_entry_point": _cls, "rsl_rl_cfg_entry_point": _AGENT},
    )

_target = sys.argv[1]
sys.argv = sys.argv[1:]
runpy.run_path(_target, run_name="__main__")
