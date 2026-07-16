#!/usr/bin/env bash
# E6: same-candidate-budget sampling MPC baseline (completion_mpc_command) vs the frontier.
# Recursive macro-transition ensemble planning over {0.75,0.90,1.00}^4, no viability gate.
set -uo pipefail
cd /home/thing1/skill_discover
PY=/home/thing1/miniconda3/envs/env_isaaclab/bin/python
CKPT=.pretrained_checkpoints/rsl_rl/Isaac-Velocity-Rough-Unitree-Go2-v0/checkpoint.pt
ARCH=outputs/isaac_go2_rough_rsl_rl_skills_long/online_action_set.pkl
RM=outputs/response_model/go2_command_response_power_ensemble5.pt
MACRO=outputs/response_model/go2_macro_transition_h4_ensemble5.pt
TV=outputs/response_model/go2_task_value_model.pt
TARGETS="2.0,0;1.5,1.0;0,1.25;-0.75,0;1.0,-1.0;0,-1.25;-1.0,0.75;1.75,-0.75"
export CUDA_VISIBLE_DEVICES=1
export TMPDIR="$HOME/storage/skill_discover/tmp"; mkdir -p "$TMPDIR"
mkdir -p outputs/e6_mpc

for SEED in "$@"; do
  echo "=== [$(date +%H:%M:%S)] E6 MPC seed $SEED ==="
  $PY scripts/evaluate_rsl_skill_command_control.py \
    --task Isaac-Velocity-Rough-Unitree-Go2-v0 \
    --checkpoint "$CKPT" --online_action_set "$ARCH" --response_model "$RM" \
    --macro_transition_model "$MACRO" --task_value_model "$TV" \
    --mpc_scales "0.75,0.90,1.00" --mpc_planning_steps 4 \
    --targets "$TARGETS" --num_trials 2 --seed "$SEED" \
    --max_high_level_steps 80 --execution_horizon 4 \
    --command_gain 1.0 --command_max 1.0 --yaw_command_gain 1.0 \
    --ff_height_scan_slice 48:235 \
    --lambda_state 2.0 --lambda_energy 0.05 --lambda_stability 0.5 \
    --lambda_no_progress 1.5 --lambda_utility 0.1 --lambda_progress 0.25 \
    --k_nearest 16 --skill_target_command_blend 1.0 \
    --relative_targets --headless --paired_method_resets \
    --methods direct_target_command,scaled_target_command_90,model_feedforward_command,completion_mpc_command \
    --output outputs/e6_mpc/e6_seed${SEED}.json > outputs/e6_mpc/e6_seed${SEED}.log 2>&1
  echo "=== [$(date +%H:%M:%S)] E6 seed $SEED exit=$? ==="
done
echo "=== E6 DONE: $* ==="
