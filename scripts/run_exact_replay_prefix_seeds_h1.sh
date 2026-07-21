#!/usr/bin/env bash
# Collect H1 exact-replay constant-scale prefixes for Stages 2–3 of the adapter audit.
# Uses untrained dimension-matched stubs for MPC plumbing; the audit analyzes
# observation-only features from exact same-state outcomes, not stub predictions.
set -uo pipefail
cd /home/thing1/skill_discover
PY=/home/thing1/miniconda3/envs/env_isaaclab/bin/python
CKPT=.pretrained_checkpoints/rsl_rl/Isaac-Velocity-Rough-H1-v0/checkpoint.pt
ARCH=outputs/isaac_h1_rough_rsl_rl_skills/online_action_set.pkl
MACRO=outputs/response_model/h1_macro_transition_h4_stub5.pt
TV=outputs/response_model/h1_task_value_stub.pt
TARGETS="2.0,0;1.5,1.0;0,1.25;-0.75,0;1.0,-1.0;0,-1.25;-1.0,0.75;1.75,-0.75"
OUT_DIR="$HOME/storage/skill_discover/outputs/isaac_h1_rough_exact_replay"
LINK_DIR=outputs/isaac_h1_rough_exact_replay
LOG_DIR="$HOME/storage/skill_discover/outputs/exact_replay_prefix_logs_h1"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export TMPDIR="$HOME/storage/skill_discover/tmp"
mkdir -p "$TMPDIR" "$OUT_DIR" "$LOG_DIR" "$LINK_DIR"

if [[ ! -f "$MACRO" || ! -f "$TV" ]]; then
  $PY scripts/make_h1_exact_replay_stubs.py
fi

if [[ $# -lt 1 ]]; then
  echo "usage: $0 SEED [SEED ...]" >&2
  exit 2
fi

for SEED in "$@"; do
  OUT="$OUT_DIR/completion_mpc_prefix_train_seed${SEED}.json"
  LINK="$LINK_DIR/completion_mpc_prefix_train_seed${SEED}.json"
  LOG="$LOG_DIR/seed${SEED}.log"
  if [[ -f "$OUT" && -s "$OUT" ]]; then
    echo "=== [$(date +%F_%H:%M:%S)] skip existing seed $SEED ==="
    ln -sfn "$OUT" "$LINK"
    continue
  fi
  echo "=== [$(date +%F_%H:%M:%S)] collecting H1 exact-replay prefix seed $SEED ===" | tee -a "$LOG_DIR/driver.log"
  $PY scripts/evaluate_rsl_skill_command_control.py \
    --task Isaac-Velocity-Rough-H1-v0 \
    --checkpoint "$CKPT" \
    --online_action_set "$ARCH" \
    --macro_transition_model "$MACRO" \
    --task_value_model "$TV" \
    --methods completion_mpc_command \
    --mpc_scales "0.75,0.90,1.00" \
    --mpc_planning_steps 4 \
    --mpc_uncertainty_k 1.0 \
    --mpc_prefix_uncertainty_k 1.0 \
    --mpc_work_margin_frac 0.02 \
    --collect_mpc_dagger \
    --dagger_queries_per_episode 3 \
    --dagger_candidate_sequences "0.75,0.75,0.75,0.75;0.90,0.90,0.90,0.90;1.00,1.00,1.00,1.00" \
    --targets "$TARGETS" \
    --num_trials 2 \
    --seed "$SEED" \
    --max_high_level_steps 80 \
    --execution_horizon 4 \
    --command_gain 1.0 \
    --command_max 1.0 \
    --yaw_command_gain 1.0 \
    --ff_height_scan_slice 69:256 \
    --relative_targets \
    --freeze_terrain_curriculum \
    --paired_method_resets \
    --headless \
    --output "$OUT" > "$LOG" 2>&1
  status=$?
  echo "=== [$(date +%F_%H:%M:%S)] H1 seed $SEED exit=$status ===" | tee -a "$LOG_DIR/driver.log"
  if [[ $status -eq 0 && -s "$OUT" ]]; then
    ln -sfn "$OUT" "$LINK"
  else
    echo "H1 seed $SEED failed; see $LOG" >&2
    tail -40 "$LOG" >&2 || true
    exit "$status"
  fi
done
echo "=== [$(date +%F_%H:%M:%S)] H1 ALL DONE: $* ===" | tee -a "$LOG_DIR/driver.log"
