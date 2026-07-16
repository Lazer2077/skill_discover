# VGCC Paper

arXiv-style preprint: **Closed-Loop Command-Response Identification of Frozen Locomotion Policies**.

The paper identifies the closed-loop command response of a frozen,
command-conditioned locomotion policy from excitation rollouts, and uses the
identified model in two modes: as a *controller* (VGCC, a training-free
viability-gated compensator) and as a *measurement* (bounding the efficiency
available at the command interface). Central finding: the command-level efficiency
frontier is narrow (a tuned fixed scale is within 2% of an omniscient per-state
oracle). In the code, the unified controller is `viability_gated_scaling_command`;
the large-sample benchmark uses the bounded-correction proxy config
(`model_feedforward_command`).

## Abstract — version history

Record the abstract verbatim on each substantive change, newest first, with a one-line note on what changed and why. (Titles evolved: v1–v2 "Viability-Gated Command Compensation…"; v3+ "Closed-Loop Command-Response Identification…".)

### v3 (2026-07-15, current) — identification-headline "two modes" framing; shorter; no em-dashes
*Change from v2:* recentered on identification (title + thesis); compensation demoted to one of two modes (controller / measurement); the narrow-frontier / oracle-<2% result made the central finding; trimmed ~235→~195 prose words; removed all `---` em-dashes.

> Learned locomotion policies are increasingly deployed as frozen, command-conditioned building blocks, driven to tasks through a velocity-command interface. We ask what can be learned about such a policy from that interface alone, without its weights, reward, or retraining, and what that knowledge is worth. We identify the frozen closed loop of policy and robot from excitation rollouts: a data-efficient model predicts, for any command in the current situation, the motion it produces, its mechanical cost, and the terrain-relative posture it leaves the robot in (displacement R²≥0.92, posture R²≈0.9, cost R²≈0.75 to 0.8, from about a minute of simulation; terrain-relative posture labels are essential). We use the model in two modes. As a controller, a training-free viability-gated compensator lowers the deployed policy's running cost and mechanical work across three embodiments (Go2, ANYmal-D, H1). As a measurement, it lets us bound the efficiency available at the command interface: from identical initial states, an omniscient per-state oracle beats the best single fixed command scale by under 2%, because the interface exposes almost no state-dependent efficiency structure. No learned per-state controller, ours or a heavier sampling MPC, clears that ceiling. Command-level efficiency compensation of a frozen locomotion policy therefore has a low and measurable ceiling; the contributions are the identification method that exposes it and the evaluation that measures it.

### v2 (2026-07-14) — efficiency-headline, honest-boundary (superseded)
*One-line:* VGCC as an efficiency controller; abstract led with "lowers optimized running cost 4–10%" then conceded the fixed-scaling frontier. Superseded when E3 (H1 safety falsified) + C(ii) probe (oracle <2%) forced the identification pivot. Full text not re-transcribed here; see git history of `main.tex`.

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
| §4.4 Table 3 + Fig. 2 (paired iso-time frontier: direct, fixed 0.75/0.90, governor, VGCC proxy/power) | `outputs/e1_isotime/frontier_seed79{4,5,6,7,8,9}.json` (6 locked seeds, 8 harder targets × 2 trials = 96 eps/method, `--paired_method_resets`); seeds 797-799 via `scripts/run_e1_frontier.sh`, 794-796 via `scripts/run_e6e4_now.sh` (both write to `~/storage/skill_discover/outputs/`, symlinked into `outputs/`). Aggregate `scripts/aggregate_e1_frontier.py`, figure `scripts/make_frontier_figure.py`, adaptivity-headroom probe (§4.6) `scripts/probe_adaptivity_headroom.py`. VGCC beats direct on work/power/COT 6/6 seeds but sits just above the fixed-scaling frontier. Power model `go2_command_response_power_ensemble5.pt` |
| §4.4 sampling-MPC baseline (completion\_mpc\_command, dominated) | `outputs/e6_mpc/e6_seed79{7,8,9}.json` (n=48, paired); run `scripts/run_e6_mpc.sh`; needs `go2_macro_transition_h4_ensemble5.pt` + `go2_task_value_model.pt` |
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
