#!/usr/bin/env bash
# Heterogeneous-terrain identification: excite on flat + harsh-rough Go2 tasks
# (same recipe as the paper: uniform commands, 64 envs, defaults), merge the two
# datasets, and train the response model with the unchanged training recipe.
set -uo pipefail
cd /home/thing1/skill_discover
PY=/home/thing1/miniconda3/envs/env_isaaclab/bin/python
CKPT=.pretrained_checkpoints/rsl_rl/Isaac-Velocity-Rough-Unitree-Go2-v0/checkpoint.pt
OUT=/home/thing1/storage/skill_discover/outputs/hetero
export CUDA_VISIBLE_DEVICES=1
export XDG_CACHE_HOME=/home/thing1/storage/skill_discover/cache
export TMPDIR=/home/thing1/storage/skill_discover/tmp

for T in Flat Rough; do
  low=$(echo "$T" | tr 'A-Z' 'a-z')
  echo "=== [$(date +%H:%M:%S)] collect $T ==="
  $PY scripts/eval_hetero.py scripts/collect_response_dataset.py \
    --task Isaac-Velocity-Hetero${T}-Unitree-Go2-v0 \
    --checkpoint "$CKPT" \
    --num_envs 64 --steps_per_env 2000 --seed 7 --headless \
    --output "$OUT/resp_${low}.npz" || echo "!!! collect $T FAILED"
done

echo "=== [$(date +%H:%M:%S)] merge ==="
$PY - <<'EOF'
import numpy as np
OUT = "/home/thing1/storage/skill_discover/outputs/hetero"
a = dict(np.load(f"{OUT}/resp_flat.npz", allow_pickle=True))
b = dict(np.load(f"{OUT}/resp_rough.npz", allow_pickle=True))
assert list(a["output_names"]) == list(b["output_names"])
assert list(a["command_slice"]) == list(b["command_slice"])
off = int(a["G"].max()) + 1
merged = {
    "X": np.concatenate([a["X"], b["X"]]),
    "Y": np.concatenate([a["Y"], b["Y"]]),
    "G": np.concatenate([a["G"], b["G"] + off]),
    "step_index": np.concatenate([a["step_index"], b["step_index"]]),
    "env_index": np.concatenate([a["env_index"], b["env_index"]]),
    "horizon": a["horizon"],
    "command_slice": a["command_slice"],
    "output_names": a["output_names"],
    "control_dt": a["control_dt"],
}
np.savez_compressed(f"{OUT}/resp_hetero_merged.npz", **merged)
print(f"flat {a['X'].shape[0]} + rough {b['X'].shape[0]} -> merged {merged['X'].shape[0]} windows,",
      f"{len(np.unique(merged['G']))} episode groups")
EOF

echo "=== [$(date +%H:%M:%S)] train ==="
$PY scripts/train_command_response_model.py \
  --dataset "$OUT/resp_hetero_merged.npz" \
  --output "$OUT/hetero_go2_response_model.pt"
echo "=== [$(date +%H:%M:%S)] DONE exit=$? ==="
