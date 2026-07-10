# Viability-Gated Feedforward Compensation (VGFC)

Compensating pretrained continuous-control policies with their own identified
command-response models — no retraining, no action-space intervention.

The idea follows classical feedforward compensation: the "plant" is the closed
loop of a frozen velocity-conditioned policy and its robot; a small network is
identified from the policy's own excitation rollouts to predict, per candidate
command, the short-horizon response (body-frame displacement, energy,
terrain-relative posture height); at deployment the model is inverted over a
candidate command set behind viability gates and applied as a bounded, annealed
correction to the task command. Paper: `paper/main.pdf`.

## Core pipeline

| Stage | Script |
| --- | --- |
| 1. Excitation collection (system identification data) | `scripts/collect_response_dataset.py` |
| 2. Response-model training | `scripts/train_command_response_model.py` |
| 3. Evaluation: VGFC (`model_feedforward_command`), retrieval variant VGSR (`guarded_skill_command`), baseline (`direct_target_command`) | `scripts/evaluate_rsl_skill_command_control.py` |
| 4. Aggregation (bootstrap CIs) and plots | `scripts/summarize_skill_eval_results.py`, `scripts/plot_skill_eval_summary.py`, `scripts/aggregate_ff_results.py` |

Experiment chains: `scripts/run_ff_systematic_experiments.sh`,
`scripts/run_paper_extension_experiments.sh`.

## Retrieval variant (VGSR)

The paper's ablation baseline reuses the same self-generated experience as a
quality-diversity-style archive of behavior chunks instead of a model:

- `skill_discovery/online/online_action_set.py` — online archive (utility and
  novelty filtering, outcome-weighted clustering)
- `skill_discovery/descriptors/locomotion_descriptors.py` — body-frame outcome descriptors
- `skill_discovery/control/archive_chunk_composer.py` — chunk selection at deployment
- `scripts/collect_rsl_rl_skills.py` — archive construction from converged-policy rollouts
- `scripts/train_skill_viability_model.py` — learned decision-level gate (paper §4.6)

## Requirements

Isaac Lab (`env_isaaclab` conda env) with public pretrained RSL-RL checkpoints
under `.pretrained_checkpoints/rsl_rl/<task>/checkpoint.pt`. Evaluated tasks:
`Isaac-Velocity-Rough-{Unitree-Go2,Anymal-D,H1}-v0`.

## Reproducing the paper

`paper/README.md` maps every table and figure to its result files and the
commands that produced them. Response models and result JSONs live under
`outputs/`.
