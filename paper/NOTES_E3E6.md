# E3–E6 experiment log (2026-07-14)

Follow-up experiments to the top-venue review. **Favorable results are folded into
`main.tex`; unfavorable results are recorded HERE and in memory only, NOT written
into the paper** (per user instruction).

Environment: `env_isaaclab`, GPU 1. Drivers: `scripts/run_e3_h1_frontier.sh`,
`scripts/run_e6_mpc.sh`, `scripts/run_e1_frontier.sh`, orchestrated by
`scripts/run_master_e3456.sh`. Aggregator: `scripts/aggregate_e1_frontier.py`.

## E3 — Fixed-scaling safety on the H1 humanoid
Purpose: the paper claims an ungated aggressive fixed scale collapses H1 posture
where VGCC does not. Previously ASSERTED, never measured against the actual
fixed-scaling baseline. Run direct / fixed-0.75 / fixed-0.90 / VGCC paired on H1
harder targets (seeds 901-903), morphology-adapted posture floors for VGCC.
Files: `outputs/e3_h1_frontier/e3_h1_seed90{1,2,3}.json`.

RESULT (seed 901, n=40/method; UNFAVORABLE — do not add to paper):
Fixed scaling does NOT collapse H1 posture. Paired, all from identical states:
- fixed-0.75: succ 1.000, finalH 0.976, minH 0.939, term 0, COT 1.762 (LOWEST)
- fixed-0.90: succ 1.000, finalH 0.965, minH 0.932, term 0, COT 1.847
- VGCC:       succ 1.000, finalH 0.956, minH 0.931, term 0, COT 1.902 (HIGHEST)
- direct:     succ 0.975, finalH 0.951, minH 0.921, term 0, COT 1.899
On the (2,0) long target the paper cites as "falls to 0.47": fixed-0.90 succ 1.00,
minH 0.827 vs direct 0.786 — fixed scaling IMPROVES posture, no collapse.
=> The paper's "ungated fixed scale collapses the humanoid's posture" is an
OVERCLAIM. The collapse the paper measured (ff_v6_h1harder8_m10) was a VGCC variant
(loose quadruped posture floors + aggressive cost margin 0.10), NOT a fixed scale.
Consistent with Go2: VGCC beats direct on H1 proxy (0.692 vs 0.729, -5.1%, matches
paper's -4.7%) but a fixed scale traces a lower COT and is equally safe.

CONFIRMED pooled seeds 901-903 (n=120/method):
  method        succ  finalH  minH   term  COT
  fixed-0.75    1.000 0.982   0.945  0     1.763  (best COT, best posture)
  fixed-0.90    1.000 0.970   0.936  0     1.859
  VGCC          1.000 0.963   0.933  0     1.904
  direct        0.992 0.960   0.928  0     1.946
  (2,0) target pooled: fixed-0.75/0.90 succ 1.00, minH 0.863/0.851 > direct 0.834.
  Zero terminations for ANY method on ANY seed/target. No posture collapse exists
  for fixed scaling on H1.

VERDICT: the "safe where fixed scaling is not" pillar does NOT survive measurement.
The collapse the paper describes is VGCC's OWN aggressive command selection under a
loose (quadruped) posture floor; it is why H1 needs morphology-adapted floors — an
argument that VGCC required extra per-morphology safety tuning that a fixed scale
did not need. Fixed scaling on H1: safe out of the box AND lower COT.

INTEGRITY ACTION (mandatory, not "adding a result"): removed the false
fixed-scale-collapse / "actively unsafe on the humanoid" claims from
abstract / §4.4 / §4.5 / conclusion / frontier caption. Kept the true statement
(VGCC's aggressive margin needs morphology-adapted floors on H1). No E3 table/figure
added to the paper. E5 (fresh seeds 904-906) will further confirm.

## E5 — H1 leakage-free confirmation
Fresh seeds 904-905 (906 lost to disk-full). Locked config, no re-tuning.
Files: `outputs/e3_h1_frontier/e3_h1_seed90{4,5}.json`.

RESULT (E3+E5 pooled seeds 901-905, n=200/method): confirms E3 across fresh seeds.
  fixed-0.75 succ 0.995 minH 0.942 term 0 COT 1.770 (best)
  fixed-0.90 succ 0.995 minH 0.931 term 0 COT 1.876
  VGCC       succ 0.990 minH 0.926 term 0 COT 1.950
  direct     succ 0.985 minH 0.924 term 0 COT 1.962
Fixed scaling is safe on H1 across all 5 seeds; the collapse claim stays falsified.

## E6 — Same-budget sampling-MPC baseline  [DONE 2026-07-15, FAVORABLE→folded in]
RESULT (completion_mpc_command, seeds 797-799 paired, n=48): DOMINATED. COT 0.757 vs
direct 0.765, VGCC 0.736, fixed-0.90 0.721. MPC is second-worst — planning more over
the identified model beats neither the fixed scale nor VGCC. Favorable for the paper
(answers "why not sampling MPC"): re-added MPC to intro Q3 + contribution 3 + §4.4
prose paragraph + §4.4 subsection title. Files `outputs/e6_mpc/e6_seed79{7,8,9}.json`
(on ~/storage). E6's `direct` reproduced E1's `direct` EXACTLY (134.52 J) → paired
protocol confirmed deterministic.

## E4 — 6-seed frontier  [DONE 2026-07-15, FAVORABLE→folded in]
RESULT (frontier seeds 794-799, n=96): story fully robust. VGCC<direct on COT 6/6
seeds; fixed-0.90<VGCC on 5/6. Pooled: fixed-0.75 0.685 < fixed-0.90 0.724 < VGCC
0.737 < governor 0.740 < direct 0.767. Updated Table 3 to n=96, figure to 6 seeds,
prose to −5.1% work / −16% power / −3.9% COT. Files
`outputs/e1_isotime/frontier_seed79{4,5,6}.json` (on ~/storage). 17 pp, clean.

## (superseded) original disk-blocked note
NOT RUN. The machine's root disk hit 100% (1.7 TB used, ~337 MB free; a pre-existing
system condition, not from this session's ~23 MB of outputs). The master driver's H1
seed 906 hung 5.4 h on `OSError: Errno 28 No space left on device`; killed it. E6 and
E4 need disk to write and were skipped. Removed the "same-budget sampling MPC"
promise from the paper's intro/Q3 rather than claim an experiment that did not run.
Assets remain (`go2_macro_transition_h4_ensemble5.pt`, `go2_task_value_model.pt`).

DISK RESOLUTION (user: 用 ~/storage/，数据存这里，等 seed 861 跑完再继续): root
`/` is 100% full but `~/storage` is a separate 7 TB disk with 2.2 TB free. Migrated
this session's outputs (`outputs/e1_isotime`, `outputs/e3_h1_frontier`,
`outputs/e6_mpc`) to `~/storage/skill_discover/outputs/` and symlinked back (paper
provenance paths unchanged). Set up `scripts/watch_861_then_e6e4.sh` (running,
PID logged) which WAITS for the other project's seed-861 run to finish, then runs
E6 (`run_e6_mpc.sh 797-799`) and E4 (frontier seeds 794-796), all with
TMPDIR + output on `~/storage` and NO `--store_positions` (metrics only, tiny files).
Marker `~/storage/skill_discover/outputs/E6E4_DONE` signals completion; an aggregation
waiter then pools E6 + the 6-seed frontier. Once results land: fold FAVORABLE into
paper (E6 MPC dominated => favorable-for-gate, can re-add MPC baseline row; E4 => 6
seeds), record UNFAVORABLE here only.

## C(ii) adaptivity-headroom probe (go/no-go for the top-venue efficiency path)
`scripts/probe_adaptivity_headroom.py` on existing paired data. For each episode,
compare scales {1.0, 0.90, 0.75}; oracle = per-episode best scale (upper bound on
adaptivity), vs best single global scale.
- Go2 (n=37): headroom (global→oracle) = 0.7%; VGCC is -9.2% vs best single scale;
  0.75 wins 78% of episodes.
- H1 (n=119): headroom = 1.7%; VGCC -8.0% vs best single scale; 0.75 wins 73%.
VERDICT = NO-GO. COT is minimized by UNIFORM slowdown (winner is monotonically the
most-aggressive-available scale), so there is no state-dependent headroom for a
per-state gate to exploit; engineered heterogeneous terrain is unlikely to help
because the COT-vs-scale trend is monotonic. This kills the honest top-venue
efficiency path. Recommend pivot to B (identification-method paper, RA-L) or A
(workshop negative result). Do NOT invest sim in engineered-terrain C(ii).

## E4 — Additional Go2 frontier seeds (statistical strengthening)
NOT RUN — same disk-full block as E6. Not promised in the paper text (frontier still
states "three seeds"), so no paper change needed. Run `scripts/run_e1_frontier.sh
794 795 796` once disk is freed.

## Data audit + negative-result presentation fix (2026-07-15)
User asked: check whether figure/table negative results hurt the argument, fix by
algorithm-optimization OR re-typeset, and verify data accuracy.
- VERIFICATION (scripts recompute from raw): ALL match exactly — Table 1 (model R²),
  Table 2 (physical energy 781-783), Table 3 (frontier n=96), Table 4 (coverage,
  incl. CIs), appendix ablation (n=80, all 6 variants), Figure 1 (17/17 seeds
  negative: harder 701-705, moderate 711-713, holdout 721-723, ANYmal 731-733, H1
  901-903; two-sided sign p=2×0.5^17=1.53e-5). No numerical errors.
- ALGORITHM OPTIMIZATION: NOT attempted — C(ii) probe already proved the frontier gap
  is fundamental (oracle <2%), so any "fix" beating fixed scaling would be fabrication.
- RE-TYPESET (the honest fix): the negative-result figure (Fig 2 frontier) visualized
  only "VGCC dominated" while the actual finding (oracle bound) had NO figure. Added
  **Figure 3 (finding.png, `scripts/make_finding_figure.py`)** in §4.6: (a) optimal
  scale is state-independent (one scale wins 74-78%), (b) per-state oracle only 1.0%
  (Go2, n=72) / 1.9% (H1, n=197) below best fixed scale — thin shaded headroom band,
  VGCC/direct above it. This turns the negative into the thesis: VGCC above the fixed
  scale is the EXPECTED consequence of an empty headroom, not a modeling failure.
  Updated §4.6 numbers to 6-seed/5-seed bases (1.0%/1.9%, was 0.7%/1.7% on 3 seeds);
  Fig 2 caption now forward-references Fig 3. NOTE: oracle computed on paired-successful
  subset, so it CANNOT be dropped onto the all-episode time-COT frontier (would falsely
  appear to beat it) — Fig 3 uses a consistent paired basis. 18 pp, compiles clean.

## Repositioning outcome
C(ii) probe NO-GO → pivoted the whole paper to framing **B**: retitled to
"Closed-Loop Command-Response Identification of Frozen Locomotion Policies",
identification is now the headline/main result (§4.2), VGCC is one application, and
the COT-frontier + oracle-headroom boundary (§4.6) is presented as a delimiting
contribution. Efficiency-over-fixed-scaling and safety-over-fixed-scaling claims
removed throughout. 16 pp, compiles clean.

---

# E7 — Heterogeneous-terrain positive control + gate-objective ablation (2026-07-17)

Purpose (user-directed): test the paper's §4.5/§5 conjecture that heterogeneous
terrain widens the per-episode oracle gap, applying the method unchanged (fresh
identification, frozen gate scalars); if negative, attempt a literature-informed
method improvement; update the paper ONLY if positive. **Both parts NEGATIVE —
paper not updated.**

Setup: custom single-type Go2 tasks on the unchanged rough env
(`scripts/hetero_env_cfg.py` + `scripts/eval_hetero.py` registration wrapper):
FLAT (plane) and ROUGH (`random_rough`, noise 2–10 cm). NOTE: Isaac Lab's
`random_uniform_terrain` IGNORES the difficulty parameter, so the intended
"mid difficulty 0.6" task generated terrain identical to the rough task
(confirmed episode-identical records); `frontier_mid_*` files are therefore a
5-method single-invocation REPLICATE on the rough terrain, not a third level.
Hetero response model: merged flat+rough excitation, 25.3k windows, same recipe
(R² disp 0.92 / energy 0.73 / heights 0.89–0.93). Paired protocol as E1
(8 harder targets x 2 trials, seeds 811-813). Cross-invocation pairing verified
bit-exact on flat (16/16) but DIVERGES on rough contacts (8/48 episodes differ
between 4- and 5-method invocations -> only within-invocation comparisons used
on rough). Files: `~/storage/skill_discover/outputs/hetero/`.

## E7a — Does roughness heterogeneity open the oracle gap? NO.
COT falls monotonically with slowdown on EVERY terrain down to the 0.60 grid
bottom (flat: 1.027/0.964/0.897/0.882 for 1.0/0.9/0.75/0.60; rough:
1.646/1.565/1.513/1.433 on the arrived+no-fall basis). The efficiency-optimal
scale is the slowest grid point everywhere; winners do not flip between
terrains. Flat paired basis (n=41-48): winner 0.60 in 65-68%, headroom
3.2-4.4% (grid-dependent, cf. paper's 1.0-1.9% on 3 scales). Rough
(single-invocation basis, arrived+no-fall, n=34): winners LOOK spread
(0.60 44%, 0.75 24%, 0.90 21%, 1.00 12%; headroom 7.8%) BUT the spread shows
no target structure (identical within every target) -> it is contact-chaos
noise inflating the per-episode min (selection bias), not state structure a
controller could predict. VGCC (evaluated config, hetero-identified model) does
not beat the best fixed scale anywhere: flat -6.3%, rough -1.4% (though on
rough it matches best-fixed COT at 28% less time). Gate active 64% (flat) /
98% (rough) of decisions = near-unconditional slowdown.

## E7b — Success-criterion artifact (affects any high-noise terrain eval)
`final_height = world z` (evaluator ~line 2071); success requires z >= 70% of
spawn z. On +/-2-10 cm noise terrain this is dominated by stop-point ground
elevation: "arrived but low final height" accounts for 0.36 (direct) to 0.58
(VGCC) of episodes on rough. Artifact-free decomposition (arrived + no fall):
direct 0.87, fixed-0.75 0.95, VGCC 0.92 — slow/compensated methods arrive MORE
and fall LESS; falls are 2-4% for all methods. Success comparisons on
high-noise terrain are criterion noise, not control differences. The paper's
curriculum-mixed benchmark is milder but the same confound exists (§4.1
discloses it).

## E7c — Gate-objective ablation (the improvement attempt): NEGATIVE
Added `proxy_efficiency_feedforward_command` (= §3.3's described cost-per-
progress objective + execution-aligned scoring; candidate set and all gates
unchanged). On flat: behavioral no-op vs the evaluated raw-energy gate (COT
0.9631 vs 0.9623, active 60% vs 61%) — slowing genuinely lowers predicted
cost-per-progress here, so both objectives rank candidates the same way. On
the PAPER'S uniform-rough benchmark (seeds 797-799 paired vs archived E1
records, 48 eps): pe is WORSE than the evaluated gate — work 132.3 vs 126.9 J,
COT 0.756 vs 0.733, success 0.792 vs 0.854, and it compensates less (time 2.08
vs 2.25 s). The progress floor already does the anti-slowdown work; dividing
by predicted progress double-penalizes and over-suppresses compensation on
irregular terrain. Literature scan (2601.10723 predictive gait selection;
LoComposition 2606.15896; adaptive energy regularization 2403.20001; power
fine-tuning 2502.10956): all retrain or train from scratch -> outside the
frozen-policy constraint. Honest conclusion: within frozen-policy command
scaling on this benchmark family, "more energy saving" == "more slowdown";
the paper's boundary result predicted this and E7 confirms it across
roughness levels.

## E7d — CRITICAL paper-integrity finding (independent of E7a-c)
§3.3 describes `viability_gated_scaling_command` (two-command set {0.75c, c},
execution-aligned scoring, cost-per-progress Eq. 5) but every reported result
(v4 benchmark, Table 3 / E1, coverage table) ran `model_feedforward_command`:
~70-candidate set (scales 0.4-1.2 + yaw deltas + speed x direction x yaw grid),
RAW predicted-energy argmin, post-selection 0.5 blend with distance annealing
(annealing unmentioned in §3.3). vgs only ever ran on dev seed 861. Appendix A
(added 2026-07-16) propagated the error ("executed command = 0.875 c_goal" and
s=0.75 describe vgs, not the evaluated controller). Matching parts: progress
floor beta=0.9, cost margin eps=0.1, posture floors, restore/rescue reflexes,
exact direct fallback, alpha=0.5 bounded correction formula. E7c shows the
described and evaluated controllers are NOT numerically equivalent on the
paper's benchmark, so the repair must rewrite §3.3 (and Appendix A) to
describe the evaluated controller. PENDING USER DECISION.
