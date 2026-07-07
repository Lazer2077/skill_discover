# Research Positioning: Viability-Gated Skill Residuals

## Current Claim

The strongest version of the project is not "we discover skills" in isolation. That space is already crowded by options, unsupervised skill discovery, skill priors, and quality-diversity repertoires.

The defensible claim is:

> A pretrained locomotion policy can be improved at test time by an online skill/action archive when the archive is used as a viability-gated high-level command residual, not as raw open-loop action replay.

This positioning matches the current evidence:

- Raw joint-action chunk replay is brittle on complex locomotion.
- Pure target-command control is strong; VGSR can reduce energy and provide small success gains on some Go2 rough harder targets.
- After strengthened trials, guarded skill residuals should be framed as energy-efficient and behavior-prior improving, not as a clean across-metric win.
- On H1, the method is stable and matches direct control, but does not yet clearly outperform it.

## Differentiation From Prior Work

Related areas overlap with parts of the method:

- Options / option-critic: temporal abstraction and learned option policies.
- DIAYN / DADS: unsupervised skill discovery from diversity or predictable dynamics.
- SPiRL / skill priors: learning priors over skills from offline experience.
- MAP-Elites / quality-diversity repertoires: archives of diverse behaviors.
- Residual RL: adding residual signals on top of a base controller.

The local novelty should be framed as the combination:

1. Online archive built from a converged locomotion policy rather than from random raw exploration.
2. Archive chunks are converted into command-space residual priors over a strong low-level RSL-RL controller.
3. Selection is gated by goal alignment, predicted progress, archive stability, energy z-score, and optional height viability.
4. The same evaluation directly compares against the pure pretrained learning controller, not only against random or zero baselines.

The novelty is moderate, not maximal. It can stand as a robotics/RL workshop or systems-style contribution if the ablations are clean and the failure modes are reported honestly.

## Current Evidence

### Go2 Rough Moderate Targets

File: `outputs/isaac_go2_rough_rsl_rl_skills_long/rsl_guarded_skill_command_eval_v1_10trials_positions.json`

| Method | Avg Success | Avg Final Distance | Avg Energy | Avg Final Height |
| --- | ---: | ---: | ---: | ---: |
| guarded skill residual | 1.0000 | 0.2892 | 1.8924 | 0.3588 |
| direct target command | 0.9750 | 0.2893 | 1.8147 | 0.3604 |

Interpretation: guarded residual improves success slightly and preserves distance, with a small energy cost.

### Go2 Rough Harder Targets

File: `outputs/isaac_go2_rough_rsl_rl_skills_long/rsl_guarded_skill_command_eval_v1_recovery_harder_5trials_positions.json`

| Method | Avg Success | Avg Final Distance | Avg Energy | Avg Final Height |
| --- | ---: | ---: | ---: | ---: |
| guarded skill residual + recovery | 0.8000 | 0.2822 | 2.0948 | 0.2953 |
| direct target command | 0.7600 | 0.2884 | 2.3916 | 0.3034 |

Interpretation: this initial 5-trial result motivated the current VGSR design. It improves success, distance, and energy, but should not be used alone as the final empirical claim.

Strengthened aggregate:

File: `outputs/isaac_go2_rough_rsl_rl_skills_long/go2_harder_strengthened_20trial_summary.json`

| Method | Avg Success | Avg Final Distance | Avg Energy | Avg Final Height |
| --- | ---: | ---: | ---: | ---: |
| guarded skill residual + recovery | 0.7900 | 0.2979 | 1.9913 | 0.3071 |
| direct target command | 0.7800 | 0.2888 | 2.1954 | 0.3174 |

Interpretation: the larger aggregate preserves only a marginal success improvement and a clearer energy improvement. Final distance and final height are worse on average, mostly because the 2m forward target remains height-limited. The research claim should emphasize energy-efficient residual behavior priors and report recovery/height as an open limitation.

Gate ablation:

File: `outputs/isaac_go2_rough_rsl_rl_skills_long/go2_harder_gate_ablation_5trial_summary.json`

| Method | Avg Success | Avg Final Distance | Avg Energy | Avg Final Height |
| --- | ---: | ---: | ---: | ---: |
| skill command, no gate | 0.8000 | 0.2739 | 3.1806 | 0.3205 |
| guarded skill residual | 0.8400 | 0.3570 | 2.0802 | 0.3278 |
| direct target command | 0.9200 | 0.2889 | 2.0988 | 0.3372 |

Interpretation: the no-gate residual reaches close to the target but uses much higher energy. The heuristic gate suppresses this overuse, but it can still lose to direct control on some seeds. This supports viability gating as necessary but not yet sufficient.

### H1 Rough Targets

File: `outputs/isaac_h1_rough_rsl_rl_skills/rsl_guarded_skill_command_eval_v1_5trials_positions.json`

| Method | Avg Success | Avg Final Distance | Avg Energy | Avg Final Height |
| --- | ---: | ---: | ---: | ---: |
| guarded skill residual | 1.0000 | 0.2905 | 0.5394 | 1.0169 |
| direct target command | 1.0000 | 0.2911 | 0.4971 | 1.0091 |

Interpretation: the method is not harmful on H1, but H1 is not yet a positive result because direct control is already saturated.

## Negative Result To Keep

2m Go2 rough straight walking remains a hard failure mode. The robot reaches near the XY target but fails the final-height criterion.

Low-height viability gates were tested:

- aggressive: preserves height but stops useful progress;
- balanced: preserves energy but still fails height and success.

This should be reported as a limitation: once the low-level policy enters a crouched/low-height regime, post-hoc command gating cannot reliably recover. The next version needs either a learned viability model that predicts height drop before it occurs, or a low-level policy trained with stronger recovery/upright constraints.

### Viability-Gate Follow-Up

Two low-height viability gates were added to the evaluator:

- `guard_low_height_fraction=0.9`, `guard_low_height_command_scale=0.2`
- `guard_low_height_fraction=0.8`, `guard_low_height_command_scale=0.5`

Both are negative/diagnostic rather than positive results. The aggressive version preserves height but blocks progress; the balanced version reaches the target neighborhood but still fails final height. This suggests a reactive height threshold is too late or too blunt.

The more promising route is a learned viability discriminator. A first offline classifier was trained from existing guarded-skill decisions:

File: `outputs/skill_viability/decision_viability_summary.json`

| Split | Samples | Episode Groups | Test Accuracy | Test AUC |
| --- | ---: | ---: | ---: | ---: |
| episode-level | 2879 | 90 | 0.9136 | 0.8986 |

This is not yet a deployed result, but it indicates that the current decision features contain enough signal to replace hand thresholds with a learned gate.

The deployable variant drops hand-written guard outputs (`guard_blend`, `guard_accepted`, `guard_low_height`) and keeps only candidate/geometry/outcome features:

File: `outputs/skill_viability/deployable_decision_viability_summary.json`

| Split | Samples | Episode Groups | Test Accuracy | Test AUC |
| --- | ---: | ---: | ---: | ---: |
| episode-level | 2879 | 90 | 0.9132 | 0.9728 |

This supports the claim that viability can be learned from decision-level features without directly copying the heuristic accept/reject variable.

## Learned-Gate Holdout

Holdout targets not used in the original Go2 evaluation:

`1.25,0.5; 0.5,1.0; -0.5,0.5; 1.25,-0.5`

File: `outputs/isaac_go2_rough_rsl_rl_skills_long/rsl_learned_guard_holdout_intersect_thr0999_5trials_positions.json`

| Method | Avg Success | Avg Final Distance | Avg Energy | Avg Final Height |
| --- | ---: | ---: | ---: | ---: |
| learned VGSR, intersect gate | 1.0000 | 0.2911 | 1.9096 | 0.3622 |
| heuristic VGSR | 1.0000 | 0.2877 | 1.9346 | 0.3577 |
| direct target command | 1.0000 | 0.2890 | 1.9063 | 0.3585 |

Interpretation: the learned gate is stable on held-out target directions and reduces energy compared with heuristic VGSR, but it is not clearly better than direct control on this target set. The paper should present learned VGSR as a promising ablation, not as the main empirical win yet.

Important negative ablations:

- learned gate in `replace` mode with threshold 0.5 over-accepts residuals and underperforms direct control;
- learned gate in `replace` mode with threshold 0.999 improves distance but remains too energetic;
- lowering residual blend to 0.25 caused an unstable right-diagonal target in one seed.

The current best learned usage is therefore **intersect mode**: the learned model filters candidates that already pass the conservative physical guard.

Harder-target learned-gate check:

File: `outputs/isaac_go2_rough_rsl_rl_skills_long/go2_harder_learned_gate_5trial_summary.json`

| Method | Avg Success | Avg Final Distance | Avg Energy | Avg Final Height |
| --- | ---: | ---: | ---: | ---: |
| learned VGSR, intersect gate | 0.7600 | 0.2779 | 1.9232 | 0.3093 |
| heuristic VGSR | 0.8000 | 0.2865 | 2.0267 | 0.3180 |
| direct target command | 0.8000 | 0.2896 | 2.0812 | 0.3333 |

Interpretation: learned-intersect reduces distance and energy on harder targets, but loses success and height, especially on the 2m forward target. This is a useful negative result: the current episode-level viability labels do not sufficiently encode height-safe arrival. The next learned gate should use short-horizon labels for height-safe progress and energy-normalized recovery.

## Required Ablations

To make the research claim credible, report these ablations:

1. Direct target command only.
2. Pure skill command without guard.
3. Guarded skill command without recovery.
4. Guarded skill command with recovery.
5. Conservative guard that nearly reduces to direct control.
6. Different archives: short collection vs long collection.
7. Cross-body validation: Go2 positive result, H1 saturated/neutral result.
8. Learned-gate modes: replace vs intersect, with threshold calibration.

Avoid claiming improvement from random/zero baselines; those are sanity checks only.

## Method Name

Use a precise name:

**Viability-Gated Skill Residuals (VGSR)**

One-sentence method:

> VGSR converts an online archive of locomotion chunks into command-space residual priors over a pretrained policy, and activates them only when a viability gate predicts that the residual is goal-aligned, progressive, stable, and not energetically out-of-distribution.

## Next Optimization

The current guard is still heuristic. The next research-grade improvement should be a learned viability discriminator:

- inputs: current observation/state feature, target in body frame, candidate skill descriptor, candidate command, direct command;
- labels: short-horizon success, height-safe outcome, energy-normalized progress;
- output: probability that applying the skill residual improves direct control without violating height/stability constraints.

This would turn the current hand-tuned guard into a learned applicability model, matching the original V2 motivation and making the contribution substantially stronger.

Implementation scaffold:

- `scripts/train_skill_viability_model.py`
- `outputs/skill_viability/decision_viability_model.pt`
- `outputs/skill_viability/decision_viability_summary.json`
