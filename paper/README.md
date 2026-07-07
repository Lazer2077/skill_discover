# VGSR Paper

arXiv-style preprint: **Viability-Gated Skill Residuals for Test-Time Locomotion Control**.

## Layout

- `main.tex` — paper source (arXiv preprint style)
- `arxiv.sty` — style file from [kourgeorge/arxiv-style](https://github.com/kourgeorge/arxiv-style)
- `references.bib` — bibliography
- `figures/` — experiment figures copied from `../outputs/isaac_go2_rough_rsl_rl_skills_long/`
- `main.pdf` — compiled output

## Build

Uses [Tectonic](https://tectonic-typesetting.github.io/) (single-binary LaTeX engine, downloads packages on demand; installed at `~/.local/bin/tectonic`):

```bash
~/.local/bin/tectonic main.tex
```

Any standard TeX Live installation also works:

```bash
latexmk -pdf main.tex
```

## Data provenance

Every number in the paper's tables was verified against the raw result files:

| Paper table | Source file |
| --- | --- |
| Table 1 (moderate) | `outputs/isaac_go2_rough_rsl_rl_skills_long/rsl_guarded_skill_command_eval_v1_10trials_positions.json` |
| Table 2 (harder, 20-trial + CIs) | `outputs/isaac_go2_rough_rsl_rl_skills_long/go2_harder_strengthened_20trial_summary.json` (episode_weighted) |
| Table 3 (gate ablation) | `outputs/isaac_go2_rough_rsl_rl_skills_long/go2_harder_gate_ablation_5trial_summary.json` |
| Table 4 (holdout) | `outputs/isaac_go2_rough_rsl_rl_skills_long/rsl_learned_guard_holdout_intersect_thr0999_5trials_positions.json` |
| Table 5 (learned gate, harder) | `outputs/isaac_go2_rough_rsl_rl_skills_long/go2_harder_learned_gate_5trial_summary.json` |
| Table 6 (H1) | `outputs/isaac_h1_rough_rsl_rl_skills/rsl_guarded_skill_command_eval_v1_5trials_positions.json` |
| Classifier acc/AUC (§5.4) | `outputs/skill_viability/deployable_decision_viability_summary.json` |
