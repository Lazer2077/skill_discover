#!/usr/bin/env bash
# After new exact-replay prefix seeds finish, rebuild the 10-seed adapter audit.
set -euo pipefail
cd /home/thing1/skill_discover
OUT=outputs/response_model
PREFIX=outputs/isaac_go2_rough_rsl_rl_skills_long
INPUTS=(
  "$PREFIX/completion_mpc_prefix_train_seed854.json"
  "$PREFIX/completion_mpc_prefix_train_seed855.json"
  "$PREFIX/completion_mpc_prefix_validation_seed856.json"
  "$PREFIX/completion_mpc_prefix_train_seed858.json"
  "$PREFIX/completion_mpc_prefix_train_seed859.json"
  "$PREFIX/completion_mpc_prefix_train_seed860.json"
  "$PREFIX/completion_mpc_prefix_train_seed862.json"
  "$PREFIX/completion_mpc_prefix_train_seed863.json"
  "$PREFIX/completion_mpc_prefix_train_seed864.json"
  "$PREFIX/completion_mpc_prefix_train_seed865.json"
)

for path in "${INPUTS[@]}"; do
  if [[ ! -s "$path" ]]; then
    echo "missing: $path" >&2
    exit 1
  fi
done

python scripts/analyze_exact_replay_oracle.py \
  --inputs "${INPUTS[@]}" \
  --output "$PREFIX/exact_replay_prefix_oracle_10seed.json"

python scripts/train_exact_replay_treatment_effect.py \
  --inputs "${INPUTS[@]}" \
  --feature-set observation_macro \
  --output "$OUT/exact_replay_adapter_audit_observation_macro_10seed.json"

# Keep observation-only as the control-calibrated representation.
python scripts/train_exact_replay_treatment_effect.py \
  --inputs "${INPUTS[@]}" \
  --feature-set observation \
  --output "$OUT/exact_replay_adapter_audit_observation_10seed.json"

python scripts/summarize_adapter_value_audit.py \
  --inputs \
    "$OUT/exact_replay_adapter_audit_observation_macro_10seed.json" \
    "$OUT/exact_replay_adapter_audit_observation_10seed.json" \
  --output "$OUT/adapter_value_audit_10seed_summary.json"

# Semi-synthetic calibration on the expanded seed set.
for strength in 0.02 0.05 0.09 0.10; do
  label=${strength/0./}
  python scripts/train_exact_replay_treatment_effect.py \
    --inputs "${INPUTS[@]}" \
    --feature-set observation \
    --control-mode observable \
    --control-strength "$strength" \
    --output "$OUT/adapter_audit_control_observable_${label}pct_10seed.json"
done

for seed in $(seq 20260720 20260729); do
  python scripts/train_exact_replay_treatment_effect.py \
    --inputs "${INPUTS[@]}" \
    --feature-set observation \
    --control-mode hidden \
    --control-strength 0.05 \
    --control-seed "$seed" \
    --output "$OUT/adapter_audit_control_hidden_05pct_seed${seed}_10seed.json"
done

python scripts/summarize_adapter_value_audit.py \
  --inputs "$OUT"/adapter_audit_control_observable_*pct_10seed.json \
  --output "$OUT/adapter_value_audit_observable_curve_10seed_summary.json"

python scripts/summarize_adapter_value_audit.py \
  --inputs "$OUT"/adapter_audit_control_hidden_05pct_seed*_10seed.json \
  --output "$OUT/adapter_value_audit_hidden_05pct_repeats_10seed_summary.json"

echo "10-seed audit artifacts written under $OUT"
