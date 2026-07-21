#!/usr/bin/env python3
"""Create stub H1 macro-transition and task-value models for exact-replay collection.

These models are dimensionally compatible with Isaac-Velocity-Rough-H1-v0
observations but are not trained. They exist only so completion_mpc_command can
drive episodes while constant-scale DAgger prefixes are exact-replayed. The H1
adapter audit uses the observation-only feature set, which ignores predicted
prefix features from these stubs.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def main() -> None:
    out_dir = Path("outputs/response_model")
    out_dir.mkdir(parents=True, exist_ok=True)

    obs_dim = 256
    command_slice = [9, 12]
    response_names = [
        "delta_x",
        "delta_y",
        "delta_yaw",
        "mechanical_power",
        "min_height",
        "max_tilt",
        "max_ang_speed",
    ]
    hidden = 256
    members = 5
    output_dim = obs_dim + len(response_names)

    state_dicts = []
    for seed in range(members):
        torch.manual_seed(20260718 + seed)
        model = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, output_dim),
        )
        # Near-zero residual dynamics so stub rollouts stay numerically calm.
        with torch.no_grad():
            for layer in model:
                if isinstance(layer, torch.nn.Linear):
                    layer.weight.mul_(0.01)
                    layer.bias.zero_()
        state_dicts.append(model.state_dict())

    macro = {
        "model_type": "macro_transition_ensemble",
        "state_dicts": state_dicts,
        "input_dim": obs_dim,
        "hidden_dim": hidden,
        "output_dim": output_dim,
        "response_names": response_names,
        "x_mean": np.zeros(obs_dim, dtype=np.float32),
        "x_std": np.ones(obs_dim, dtype=np.float32),
        "y_mean": np.zeros(output_dim, dtype=np.float32),
        "y_std": np.ones(output_dim, dtype=np.float32),
        "horizon": 4,
        "control_dt": 0.02,
        "command_slice": command_slice,
        "stub": True,
        "note": "Untrained H1 stub for exact-replay constant-scale collection only.",
    }
    macro_path = out_dir / "h1_macro_transition_h4_stub5.pt"
    torch.save(macro, macro_path)

    tv_input = obs_dim + 3  # obs + local target (x, y, unused pad to match Go2 recipe)
    # Go2 uses input_dim = obs_dim + 3 = 238; keep the same convention.
    torch.manual_seed(20260718)
    tv = torch.nn.Sequential(
        torch.nn.Linear(tv_input, hidden),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden, hidden),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden, 3),
    )
    with torch.no_grad():
        for layer in tv:
            if isinstance(layer, torch.nn.Linear):
                layer.weight.mul_(0.01)
                layer.bias.zero_()
    tv_payload = {
        "state_dict": tv.state_dict(),
        "input_dim": tv_input,
        "obs_dim": obs_dim,
        "hidden_dim": hidden,
        "command_slice": command_slice,
        "x_mean": np.zeros(tv_input, dtype=np.float32),
        "x_std": np.ones(tv_input, dtype=np.float32),
        "y_mean": np.zeros(2, dtype=np.float32),
        "y_std": np.ones(2, dtype=np.float32),
        "stub": True,
        "note": "Untrained H1 stub for exact-replay collection only.",
    }
    tv_path = out_dir / "h1_task_value_stub.pt"
    torch.save(tv_payload, tv_path)
    print(f"wrote {macro_path}")
    print(f"wrote {tv_path}")


if __name__ == "__main__":
    main()
