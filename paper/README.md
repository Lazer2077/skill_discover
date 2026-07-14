# VGCC Paper

arXiv-style preprint: **Viability-Gated Command Compensation for Frozen Locomotion Policies**.

VGCC is a training-free controller for a frozen, command-conditioned locomotion
policy. It identifies the policy--robot closed-loop command response offline, then
online chooses between a nominally efficient scaled command and exact direct
control behind a learned progress/posture viability gate. In the code, the unified
controller is `viability_gated_scaling_command` (execution-aligned scoring,
torque-derived power / cost-per-progress objective, exact fallback); the
large-sample benchmark was run with the bounded-correction proxy configuration
(`model_feedforward_command`).

## Layout

- `main.tex` — paper source (arXiv preprint style)
- `appendix.tex` — per-target results, retrieval alternative (VGSR), component ablation
- `arxiv.sty` — style file from [kourgeorge/arxiv-style](https://github.com/kourgeorge/arxiv-style)
- `references.bib` — bibliography
- `figures/` — experiment figures copied from `../outputs/`
- `main.pdf` — compiled output

## Build

Uses [Tectonic](https://tectonic-typesetting.github.io/) (single-binary LaTeX engine, downloads packages on demand; installed at `~/.local/bin/tectonic`):

```bash
~/.local/bin/tectonic main.tex
```

Any standard TeX Live installation also works: `latexmk -pdf main.tex`.

## Data provenance

Every number retained in the paper was verified against raw result files (paths relative to repo root).

| Paper element | Source |
| --- | --- |
| Table 1 (Go2 response model) | `outputs/response_model/go2_command_response_model_summary.json`; world-frame row from the earlier world-z model run (reproducible via `collect_response_dataset.py` without the relative-height patch); power-channel ensemble R² from `outputs/power_objective_audit_summary.json` |
| §4.3 running-cost aggregate + Fig. 1 (paired differences) | `outputs/isaac_go2_rough_rsl_rl_skills_long/ff_v4_harder8_summary.json` (inputs: `ff_v4_harder8_10trials_seed70{1..5}_positions.json`); paired figure via session script `make_paired_figure.py` |
| §4.3 Table 2 (torque-derived mechanical work) | `a_mechanical_energy_2trials_seed78{1,2,3}.json` (3 seeds, 48 episodes/method; applied torque, mechanical work, power, J/m, COT) |
| §4.4 Table 3 (strong baselines: fixed scaling, reactive governor) | `outputs/power_objective_audit_summary.json` (locked seeds 797/798/799, 48 eps/method); power dataset/model `go2_response_dataset_power.npz`, `go2_command_response_power_ensemble5.pt` |
| §4.5 Coverage: Go2 moderate/holdout | `ff_v4_moderate_summary.json`, `ff_v4_holdout_summary.json` (Go2 dir) |
| §4.5 Coverage: ANYmal-D (8 targets) | `outputs/isaac_anymal_rough_ff/ff_v4_anymal_summary.json`; response model `outputs/response_model/anymal_command_response_model_summary.json` |
| §4.5 Coverage: H1 (8 targets, −4.7%) | `outputs/isaac_h1_rough_rsl_rl_skills/ff_v7_h1solved_summary.json` (inputs: `ff_v7_h1solved_5trials_seed90{1..3}_positions.json`; posture floors tightened for humanoid: `--ff_min_height_fraction 0.93 --ff_rescue_height_fraction 0.90 --ff_current_height_fraction 0.96 --ff_max_height_drop 0.008`); response model `outputs/response_model/h1_command_response_model_v2_summary.json` |
| §4.5 H1 failure-mode (loose quadruped gate → posture collapse) | `ff_v6_h1harder8_m10_5trials_seed90{1..3}` (cost margin 0.10, quadruped posture floors) |
| §4.6 / App. C VGCC component ablation | `vgfc4_ablation_{full,noanneal,nogate,nofloor,subst,data25}_seed75{1,2}_positions.json` (Go2 dir) |
| §4.2 data efficiency (25%/50%) | `outputs/response_model/go2_command_response_model_data{25,5}_summary.json` |
| App. B substitution-at-scale, absolute-margin on H1 | `ff_v2_harder_10trials_seed20{1..5}_positions.json`; `ff_v2_h1_5trials_seed23{1..3}_positions.json` |
| App. A per-target harder table + trajectories/control example | `ff_v4_harder8_summary.json`; figures from `ff_v4_harder8_10trials_seed701/704_positions.json` (example: seed 704, trial 9, target (2,0)) |
| App. A residual-failure-mode paragraph (symmetric recovery, n=400/method) | `ff_v5_harder8_rec_10trials_seed80{1..5}_positions.json` + `ff_v5_harder8_summary.json` (Go2 dir) |
| App. retrieval (VGSR): multi-seed aggregate, gate/archive ablations | `go2_harder_strengthened_40trial_summary.json`, `go2_harder_gate_ablation_5trial_summary.json`, `rsl_archive_ablation_arch{500,2000,full}_harder_5trials_seed106_positions.json` |
| Paired per-seed stats, inference latency | computed from `ff_v4_*` records / offline benchmark (session log) |

Pipelines:
- Response dataset: `scripts/collect_response_dataset.py` (uniform command excitation, terrain-relative height + torque-derived power labels)
- Model training: `scripts/train_command_response_model.py`; power ensemble: `scripts/train_command_response_ensemble.py`
- Evaluation: `scripts/evaluate_rsl_skill_command_control.py` (`viability_gated_scaling_command` = unified VGCC, `model_feedforward_command` = bounded-correction proxy VGCC used for the large benchmark, `guarded_skill_command` = VGSR retrieval variant)
- Systematic chain: `scripts/run_ff_systematic_experiments.sh`; aggregation: `scripts/aggregate_ff_results.py`
- VGCC tuned only on seed 301 (harder set); all reported results use disjoint seeds (201–205, 701–705, 231–233, 601–603, 901–903).

> The exact-replay / completion-MPC / task-value / treatment-effect controllers and
> the uncertainty and multi-signal safety filters were development-time audits used
> to stress-test and shape the final algorithm. They are no longer presented in the
> paper; their scripts (`train_task_value_model.py`, `train_macro_transition_ensemble.py`,
> `train_exact_replay_treatment_effect.py`, `train_mpc_prefix_calibrator.py`,
> `train_policy_value_model.py`, `train_paired_scale_selector.py`,
> `analyze_exact_replay_oracle.py`, `analyze_oracle_separability.py`) and result
> files remain in the repository for reference.
