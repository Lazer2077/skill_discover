"""Collect a command-response dataset for feedforward compensation.

Runs the pretrained policy in many parallel envs while overriding the velocity
command with uniformly sampled excitation commands (system-identification
style). Records per-step observation, action, joint velocity, base pose, and
episode boundaries, then slices constant-command windows into
(input observation, short-horizon response) pairs:

    X: observation at window start (command slice holds the applied command)
    Y: [delta_x, delta_y, delta_yaw, energy, min_height, mean_height]

No utility filtering: sloppy, unstable, and low-height behavior is kept on
purpose so the model learns where the policy performs poorly.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect command-response dataset.")
    parser.add_argument("--task", type=str, default="Isaac-Velocity-Rough-Unitree-Go2-v0")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--steps_per_env", type=int, default=4000)
    parser.add_argument("--command_slice", type=str, default="9:12")
    parser.add_argument("--command_max", type=float, default=1.0)
    parser.add_argument("--resample_interval", type=int, default=25)
    parser.add_argument("--slow_command_prob", type=float, default=0.3)
    parser.add_argument("--slow_command_scale", type=float, default=0.3)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--height_scan_slice", type=str, default="48:235",
                        help="Obs slice holding the height scan; heights are computed relative to terrain.")
    parser.add_argument("--height_scan_offset", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--output", type=str, required=True)
    return parser.parse_args()


def quat_to_yaw_np(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def main() -> None:
    args = parse_args()

    from skill_discovery.envs.isaac_env_wrapper import IsaacEnvWrapper, launch_app

    sim_app = launch_app(headless=args.headless)
    try:
        import gymnasium as gym
        import torch
        from rsl_rl.runners import DistillationRunner, OnPolicyRunner

        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from isaaclab_tasks.utils import parse_env_cfg
        from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

        import isaaclab_tasks  # noqa: F401

        cmd_lo, cmd_hi = (int(v) for v in args.command_slice.split(":"))
        cmd_dim = cmd_hi - cmd_lo
        rng = np.random.default_rng(args.seed)

        env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
        agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
        agent_cfg.seed = args.seed
        agent_cfg.device = args.device
        env_cfg.seed = args.seed
        env_cfg.sim.device = args.device

        raw_env = gym.make(args.task, cfg=env_cfg)
        state_reader = IsaacEnvWrapper(raw_env, device=args.device)
        rl_env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)

        if agent_cfg.class_name == "OnPolicyRunner":
            runner = OnPolicyRunner(rl_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        else:
            runner = DistillationRunner(rl_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(args.checkpoint)
        policy = runner.get_inference_policy(device=rl_env.unwrapped.device)

        obs, _ = rl_env.reset()

        N = args.num_envs
        T = args.steps_per_env

        def sample_commands(n: int) -> np.ndarray:
            c = rng.uniform(-args.command_max, args.command_max, size=(n, cmd_dim)).astype(np.float32)
            slow = rng.uniform(size=n) < args.slow_command_prob
            c[slow] *= args.slow_command_scale
            return c

        commands = sample_commands(N)
        steps_since_resample = np.zeros(N, dtype=np.int64)
        episode_ids = np.arange(N, dtype=np.int64)
        next_episode_id = N

        obs_dim = None
        obs_buf = None
        act_buf = None
        jvel_buf = None
        pos_buf = None
        quat_buf = None
        epid_buf = np.zeros((T, N), dtype=np.int64)

        for t in range(T):
            resample = steps_since_resample >= args.resample_interval
            if resample.any():
                commands[resample] = sample_commands(int(resample.sum()))
                steps_since_resample[resample] = 0
            steps_since_resample += 1

            key = "policy" if hasattr(obs, "keys") and "policy" in obs.keys() else None
            policy_obs_t = obs[key] if key else obs
            policy_obs_t = policy_obs_t.clone()
            cmd_t = torch.as_tensor(commands, device=policy_obs_t.device)
            policy_obs_t[:, cmd_lo:cmd_hi] = cmd_t
            if obs_dim is None:
                obs_dim = policy_obs_t.shape[-1]
                obs_buf = np.zeros((T, N, obs_dim), dtype=np.float32)
                state0 = state_reader.get_robot_state()
                act_dim = None
                jv_dim = state0["joint_vel"].shape[-1]
                jvel_buf = np.zeros((T, N, jv_dim), dtype=np.float32)
                pos_buf = np.zeros((T, N, 3), dtype=np.float32)
                quat_buf = np.zeros((T, N, 4), dtype=np.float32)

            state = state_reader.get_robot_state()
            obs_buf[t] = policy_obs_t.detach().cpu().numpy()
            pos_buf[t] = np.asarray(state["base_pos"], dtype=np.float32)
            quat_buf[t] = np.asarray(state["base_quat"], dtype=np.float32)
            jvel_buf[t] = np.asarray(state["joint_vel"], dtype=np.float32)
            epid_buf[t] = episode_ids

            with torch.inference_mode():
                obs_in = obs
                if key:
                    obs_in = obs.clone()
                    obs_in[key] = policy_obs_t
                else:
                    obs_in = policy_obs_t
                actions = policy(obs_in)
                if agent_cfg.clip_actions is not None:
                    actions = torch.clamp(actions, -agent_cfg.clip_actions, agent_cfg.clip_actions)
            if act_buf is None:
                act_buf = np.zeros((T, N, actions.shape[-1]), dtype=np.float32)
            act_buf[t] = actions.detach().cpu().numpy()

            obs, _rew, dones, _extras = rl_env.step(actions)
            dones_np = np.asarray(dones.detach().cpu().numpy()).reshape(-1).astype(bool)
            if dones_np.any():
                n_done = int(dones_np.sum())
                episode_ids[dones_np] = np.arange(next_episode_id, next_episode_id + n_done)
                next_episode_id += n_done
                commands[dones_np] = sample_commands(n_done)
                steps_since_resample[dones_np] = 0

            if t % 500 == 0:
                print(f"step {t}/{T}, episodes so far {next_episode_id}")

        print("collection done; slicing windows...")
        H, S = args.horizon, args.stride
        X_list, Y_list, G_list = [], [], []
        for e in range(N):
            for t0 in range(0, T - H, S):
                ep = epid_buf[t0, e]
                if not np.all(epid_buf[t0 : t0 + H + 1 if t0 + H < T else T, e] == ep):
                    continue
                w_cmd = obs_buf[t0 : t0 + H, e, cmd_lo:cmd_hi]
                if float(np.max(np.abs(w_cmd - w_cmd[0]))) > 1e-5:
                    continue
                yaw0 = float(quat_to_yaw_np(quat_buf[t0, e]))
                yaw1 = float(quat_to_yaw_np(quat_buf[t0 + H, e]))
                d_world = pos_buf[t0 + H, e, :2] - pos_buf[t0, e, :2]
                c, s = np.cos(-yaw0), np.sin(-yaw0)
                dx = c * d_world[0] - s * d_world[1]
                dy = s * d_world[0] + c * d_world[1]
                dyaw = float(np.arctan2(np.sin(yaw1 - yaw0), np.cos(yaw1 - yaw0)))
                a = act_buf[t0 : t0 + H, e]
                jv = jvel_buf[t0 : t0 + H, e]
                m = min(a.shape[1], jv.shape[1])
                energy = float(np.mean(np.abs(a[:, :m] * jv[:, :m])))
                sc_lo, sc_hi = (int(v) for v in args.height_scan_slice.split(":"))
                heights = (
                    obs_buf[t0 : t0 + H, e, sc_lo:sc_hi].mean(axis=1) + args.height_scan_offset
                )
                X_list.append(obs_buf[t0, e])
                Y_list.append(
                    np.asarray(
                        [dx, dy, dyaw, energy, float(np.min(heights)), float(np.mean(heights))], dtype=np.float32
                    )
                )
                G_list.append(int(ep))

        X = np.stack(X_list)
        Y = np.stack(Y_list)
        G = np.asarray(G_list, dtype=np.int64)
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        np.savez_compressed(args.output, X=X, Y=Y, G=G, horizon=H, command_slice=[cmd_lo, cmd_hi])
        print(f"saved {X.shape[0]} samples ({len(np.unique(G))} episodes) to {args.output}", flush=True)
        # Isaac Sim app close can hang after headless multi-env runs; the dataset is
        # already on disk, so exit hard instead of risking a stuck process.
        os._exit(0)
    finally:
        sim_app.close()


if __name__ == "__main__":
    main()
