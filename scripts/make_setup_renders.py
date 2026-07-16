#!/usr/bin/env python3
"""Headless RTX renders of the three embodiments on Isaac Lab rough terrain, for the
paper's setup/teaser figure. No GUI: uses AppLauncher(enable_cameras=True) and
env.render() with render_mode='rgb_array'.

Run:  env_isaaclab/bin/python scripts/make_setup_renders.py [robots...]
Saves PNGs to paper/figures/renders/<robot>.png
"""
import os
import sys
import numpy as np

ROBOTS = {
    "go2":    ("Isaac-Velocity-Rough-Unitree-Go2-v0", ".pretrained_checkpoints/rsl_rl/Isaac-Velocity-Rough-Unitree-Go2-v0/checkpoint.pt", 1.6),
    "anymal": ("Isaac-Velocity-Rough-Anymal-D-v0",    ".pretrained_checkpoints/rsl_rl/Isaac-Velocity-Rough-Anymal-D-v0/checkpoint.pt", 1.9),
    "h1":     ("Isaac-Velocity-Rough-H1-v0",          ".pretrained_checkpoints/rsl_rl/Isaac-Velocity-Rough-H1-v0/checkpoint.pt", 2.4),
}
WANT = [r for r in sys.argv[1:] if r in ROBOTS] or list(ROBOTS)
OUT = "paper/figures/renders"
os.makedirs(OUT, exist_ok=True)

from isaaclab.app import AppLauncher
app = AppLauncher(headless=True, enable_cameras=True).app

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
import isaaclab_tasks  # noqa: F401  (registers the Isaac-Velocity-* envs)

try:
    from isaacsim.core.utils.viewports import set_camera_view
except Exception:
    from omni.isaac.core.utils.viewports import set_camera_view

from PIL import Image


def save(img, path):
    a = np.asarray(img)
    if a.dtype != np.uint8:
        a = (255 * np.clip(a, 0, 1)).astype(np.uint8)
    if a.shape[-1] == 4:
        a = a[..., :3]
    Image.fromarray(a).save(path)
    print("saved", path, a.shape, flush=True)


for name in WANT:
    task, ckpt, cam_h = ROBOTS[name]
    print(f"=== {name} : {task} ===", flush=True)
    env_cfg = parse_env_cfg(task, device="cuda:0", num_envs=1)
    env_cfg.seed = 7
    # freeze curriculum so we land on visibly rough terrain, not the flat easy level
    if hasattr(env_cfg, "curriculum") and hasattr(env_cfg.curriculum, "terrain_levels"):
        env_cfg.curriculum.terrain_levels = None
    # hide the velocity-command debug arrow for a clean teaser
    try:
        env_cfg.commands.base_velocity.debug_vis = False
    except Exception as e:
        print("debug_vis warn:", e, flush=True)
    raw_env = gym.make(task, cfg=env_cfg, render_mode="rgb_array")

    # brighten the scene: add a bright grey dome (sky/ambient) + a key distant light
    try:
        import omni.usd
        from pxr import UsdLux, Gf, UsdGeom
        stage = omni.usd.get_context().get_stage()
        for prim in stage.Traverse():
            if prim.IsA(UsdLux.DistantLight):
                UsdLux.LightAPI(prim).GetIntensityAttr().Set(3000.0)
        dome = UsdLux.DomeLight.Define(stage, "/World/PaperDome")
        dome.CreateIntensityAttr(1400.0)
        dome.CreateColorAttr(Gf.Vec3f(0.75, 0.82, 0.95))
        key = UsdLux.DistantLight.Define(stage, "/World/PaperKeyLight")
        key.CreateIntensityAttr(3200.0)
        key.CreateAngleAttr(1.5)
        key.CreateColorAttr(Gf.Vec3f(1.0, 0.97, 0.9))
        UsdGeom.Xformable(key.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(-40.0, 25.0, 0.0))
    except Exception as e:
        print("light warn:", e, flush=True)
    agent_cfg = load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")
    agent_cfg.device = "cuda:0"
    rl_env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(rl_env, agent_cfg.to_dict(), log_dir=None, device="cuda:0")
    runner.load(ckpt)
    policy = runner.get_inference_policy(device=rl_env.unwrapped.device)

    obs, _ = rl_env.reset()
    for i in range(90):  # let the robot settle into a natural walking pose
        with torch.no_grad():
            act = policy(obs)
        obs, _, _, _ = rl_env.step(act)
        # keep the camera trailing the robot
        if i % 5 == 0:
            try:
                p = raw_env.unwrapped.scene["robot"].data.root_pos_w[0].cpu().numpy()
                set_camera_view(eye=(p[0] - 2.2, p[1] - 2.2, p[2] + cam_h),
                                target=(p[0], p[1], p[2] + 0.1))
            except Exception as e:
                if i == 0:
                    print("cam warn:", e, flush=True)
    # a couple of render passes so RTX has a full frame
    img = None
    for _ in range(4):
        img = raw_env.render()
    if img is not None:
        save(img, os.path.join(OUT, f"{name}.png"))
    else:
        print(f"!! {name}: render() returned None", flush=True)
    raw_env.close()

print("ALL RENDERS DONE", flush=True)
os._exit(0)  # Isaac Sim can hang on app.close() after headless render
