# VGFC Paper

arXiv-style preprint: **Viability-Gated Feedforward Compensation of Learned Continuous-Control Policies**.

## Layout

- `main.tex` — paper source (arXiv preprint style)
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

Every number in the paper was verified against raw result files (paths relative to repo root).

| Paper element | Source |
| --- | --- |
| Table 1 (Go2 response model) | `outputs/response_model/go2_command_response_model_summary.json`; world-frame row from the earlier world-z model run (recorded in the session logs; reproducible via `collect_response_dataset.py` without the relative-height patch) |
| §4.3 aggregate + per-target table (8 targets, n=400) | `outputs/isaac_go2_rough_rsl_rl_skills_long/ff_v4_harder8_summary.json` (inputs: `ff_v4_harder8_10trials_seed70{1..5}_positions.json`) |
| Fig. 1 (paired per-seed differences) | computed from `ff_v4_*` records across all families (session script `make_paired_figure.py`) |
| Figs. 2-3 (trajectories, control example) | `ff_v4_harder8_10trials_seed701/704_positions.json` (example: seed 704, trial 9, target (2,0)) |
| Coverage table (moderate/holdout/ANYmal/H1) | `ff_v4_moderate_summary.json`, `ff_v4_holdout_summary.json` (Go2 dir); `outputs/isaac_anymal_rough_ff/ff_v4_anymal_summary.json`; `outputs/isaac_h1_rough_rsl_rl_skills/ff_v4_h1_summary.json` |
| §4.6 VGSR multi-seed aggregate | `outputs/isaac_go2_rough_rsl_rl_skills_long/go2_harder_strengthened_40trial_summary.json` |
| §4.6 gate ablation | `go2_harder_gate_ablation_5trial_summary.json` |
| §4.6 archive-size ablation | `rsl_archive_ablation_arch{500,2000,full}_harder_5trials_seed106_positions.json` |
| §4.6 learned decision gate | `outputs/skill_viability/deployable_decision_viability_summary.json`, `go2_harder_learned_gate_5trial_summary.json`, `rsl_learned_guard_holdout_intersect_thr0999_5trials_positions.json` |
| VGFC component ablation table | `vgfc4_ablation_{full,noanneal,nogate,nofloor,subst,data25}_seed75{1,2}_positions.json` (Go2 dir) |
| §4.5 substitution-at-scale | `ff_v2_harder_10trials_seed20{1..5}_positions.json` (250 eps/method) |
| §4.5 absolute margin on H1 | `ff_v2_h1_5trials_seed23{1..3}_positions.json` |
| §4.2 data efficiency (25%/50%) | `outputs/response_model/go2_command_response_model_data{25,5}_summary.json` |
| Paired per-seed stats, inference latency | computed from `ff_v4_*` records / offline benchmark (session log) |
| §4.3 residual-failure-mode paragraph (symmetric recovery, n=400/method) | `ff_v5_harder8_rec_10trials_seed80{1..5}_positions.json` + `ff_v5_harder8_summary.json` (Go2 dir) |
| ANYmal-D extension (coverage table row) | `outputs/isaac_anymal_rough_ff/ff_v3_anymal_summary.json` (inputs: `ff_v3_anymal_5trials_seed60{1..3}_positions.json`) |
| Trajectories / control-example figures | generated from `ff_v3_harder_10trials_seed201_positions.json` (trial 9 on target (2,0)) via session script `make_control_figures.py` |
| H1 response model | `outputs/response_model/h1_command_response_model_v2_summary.json` |
| ANYmal response model | `outputs/response_model/anymal_command_response_model_summary.json` |

Pipelines:
- Response dataset: `scripts/collect_response_dataset.py` (uniform command excitation, terrain-relative height labels)
- Model training: `scripts/train_command_response_model.py`
- Evaluation: `scripts/evaluate_rsl_skill_command_control.py` (`model_feedforward_command` = VGFC, `guarded_skill_command` = VGSR)
- Systematic chain: `scripts/run_ff_systematic_experiments.sh`; aggregation: `scripts/aggregate_ff_results.py`
- VGFC tuned only on seed 301 (harder set); all reported results use disjoint seeds (201–205, 211–213, 221–223, 231–233).
