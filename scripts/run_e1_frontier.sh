#!/usr/bin/env bash
# E1 iso-time frontier (paired): all methods from identical initial states.
# Methods: direct, fixed-0.75, fixed-0.90, reactive governor, proxy-VGCC, power-VGCC.
# VGCC config replicated from the locked power audit (power_vgfc_locked_*).
set -uo pipefail
cd /home/thing1/skill_discover
PY=/home/thing1/miniconda3/envs/env_isaaclab/bin/python
CKPT=.pretrained_checkpoints/rsl_rl/Isaac-Velocity-Rough-Unitree-Go2-v0/checkpoint.pt
ARCH=outputs/isaac_go2_rough_rsl_rl_skills_long/online_action_set.pkl
RM=outputs/response_model/go2_command_response_power_ensemble5.pt
TARGETS="2.0,0;1.5,1.0;0,1.25;-0.75,0;1.0,-1.0;0,-1.25;-1.0,0.75;1.75,-0.75"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
mkdir -p outputs/e1_isotime

SEEDS="$*"
for SEED in $SEEDS; do
  echo "=== [$(date +%H:%M:%S)] launching seed $SEED ==="
  $PY scripts/evaluate_rsl_skill_command_control.py \
    --task Isaac-Velocity-Rough-Unitree-Go2-v0 \
    --checkpoint "$CKPT" --online_action_set "$ARCH" --response_model "$RM" \
    --targets "$TARGETS" --num_trials 2 --seed "$SEED" \
    --max_high_level_steps 80 --execution_horizon 4 \
    --command_gain 1.0 --command_max 1.0 --yaw_command_gain 1.0 \
    --ff_height_scan_slice 48:235 \
    --lambda_state 2.0 --lambda_energy 0.05 --lambda_stability 0.5 \
    --lambda_no_progress 1.5 --lambda_utility 0.1 --lambda_progress 0.25 \
    --k_nearest 16 --skill_target_command_blend 1.0 \
    --relative_targets --store_positions --headless --paired_method_resets \
    --methods direct_target_command,scaled_target_command_75,scaled_target_command_90,reactive_governor_command,model_feedforward_command,mechanical_feedforward_command \
    --output outputs/e1_isotime/frontier_seed${SEED}.json > outputs/e1_isotime/frontier_seed${SEED}.log 2>&1
  echo "=== [$(date +%H:%M:%S)] seed $SEED exit=$? ==="
done
echo "=== E1 FRONTIER DONE for seeds: $SEEDS ==="
