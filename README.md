# Unsupervised Skill Discovery and Skill-Space Control for Locomotion

A **V1 research prototype** for Isaac Sim / Isaac Lab: a robot explores its
own locomotion behaviors, the resulting trajectories are segmented into short
behavior chunks, the chunks are clustered into **meta-behaviors (skills)**,
and downstream tasks are solved by **composing** the discovered skills instead
of training an end-to-end task policy.

```
explore  →  segment  →  describe  →  cluster  →  skill library  →  compose
```

## 1. Motivation

Training one end-to-end RL policy per locomotion task is expensive and
opaque. The alternative explored here: let the robot first discover a
reusable vocabulary of behaviors **without any task reward**, then treat
downstream control as *planning in skill space*. This follows a long line of
work on unsupervised skill discovery and behavior repertoires:

- **Options / temporally extended actions** — Sutton, Precup & Singh, 1999.
- **DIAYN**: Diversity is All You Need — Eysenbach et al., 2018 (skills as
  latent-conditioned policies maximizing state-skill mutual information).
- **DADS**: Dynamics-Aware Unsupervised Discovery of Skills — Sharma et al.,
  2019 (skills with predictable dynamics, used for model-based planning —
  the direct inspiration for our skill-level MPC).
- **Quality-Diversity / behavior repertoires** — Cully et al., Nature 2015
  (behavior descriptors as the space in which diversity is organized — the
  direct inspiration for our descriptor design).

V1 replaces the *learned* components of those methods with the simplest
possible instantiations (random exploration, fixed-horizon chunks, k-means,
replayed action sequences) to get a complete, runnable pipeline whose every
stage can later be upgraded independently.

## 2. What V1 is (and is not)

| V1 does | V1 does **not** |
|---|---|
| State-based (proprioceptive) locomotion | Any image/RGB/depth input |
| Random & latent-conditioned random exploration | Train a neural exploration policy |
| Fixed-horizon segmentation | Learned option termination |
| Hand-designed behavior descriptors | Learned representations |
| K-means / GMM / HDBSCAN clustering | Contrastive skill embeddings |
| Skills = representative **action sequences** (open loop) | Learned closed-loop skill policies |
| Discrete skill-level planning (greedy + brute-force MPC) | Solving the full HJB equation |

The skill-level planner is a **skill-space optimal control approximation**:
it searches over discrete skill sequences using each skill's *average*
recorded outcome as a deterministic dynamics model, with cost
`distance_to_goal + λ_energy · energy + λ_yaw · yaw_error`.

## 3. Repository structure

```
├── README.md
├── requirements.txt
├── configs/                  # YAML configs (default / ant / go2)
├── scripts/
│   ├── collect_exploration.py        # 1. run Isaac Lab, collect rollouts
│   ├── gym_online_skill_control.py   # Gymnasium V2 online action-set test
│   ├── extract_descriptors.py        # 2. segment + descriptors (no sim needed)
│   ├── cluster_skills.py             # 3. cluster + build skill library (no sim)
│   ├── evaluate_skill_composition.py # 4. target reaching by skill composition
│   └── visualize_skills.py           # 5. plots + skill_summary.json (no sim)
├── skill_discovery/
│   ├── envs/isaac_env_wrapper.py     # Isaac Lab env + robust state extraction
│   ├── exploration/                  # random & latent policies, rollout collector
│   ├── segmentation/fixed_horizon_segmenter.py
│   ├── descriptors/locomotion_descriptors.py
│   ├── clustering/skill_clusterer.py
│   ├── library/skill_library.py
│   ├── control/skill_composer.py     # greedy composition
│   ├── control/skill_mpc.py          # brute-force skill-level MPC
│   └── utils/                        # buffers, math, logging, plotting
└── outputs/                          # rollouts, libraries, plots (gitignored)
```

## 4. Installation

1. **Install Isaac Sim + Isaac Lab** following the
   [official Isaac Lab instructions](https://isaac-sim.github.io/IsaacLab/).
   This repo is tested against the Isaac Lab python API and supports both the
   `isaaclab.*` (≥ 2.0) and legacy `omni.isaac.lab.*` (1.x) package names.
2. Clone this repo anywhere (e.g. next to your IsaacLab checkout).
3. Install the light dependencies into the Isaac Lab python environment:

```bash
./isaaclab.sh -p -m pip install -r /path/to/this/repo/requirements.txt
```

`numpy`, `torch`, `scikit-learn`, `matplotlib`, `pyyaml`, `tqdm` are the only
dependencies; `hdbscan` is optional (the clusterer falls back to k-means).

**Isaac Lab dependency notes**

- Steps 1 and 4 (collection, evaluation) need Isaac Sim and must be launched
  through `./isaaclab.sh -p …`. Steps 2, 3, 5 are pure NumPy/sklearn and run
  with any python that has the requirements installed.
- Task names change between Isaac Lab versions. List what your install has:

```bash
./isaaclab.sh -p scripts/collect_exploration.py --list_envs
# or Isaac Lab's own: ./isaaclab.sh -p scripts/environments/list_envs.py
```

Preferred tasks, in order: `Isaac-Ant-v0`,
`Isaac-Velocity-Flat-Unitree-Go2-v0`, `Isaac-Velocity-Flat-Anymal-C-v0` —
pick the closest locomotion task your version registers.

## 5. Running the pipeline

```bash
# 1. Collect exploration rollouts (Isaac Sim)
./isaaclab.sh -p scripts/collect_exploration.py \
  --task Isaac-Ant-v0 --num_envs 256 --num_steps 50000 \
  --headless --output outputs/rollouts_ant.pkl --config configs/ant.yaml

# 2. Segment + extract behavior descriptors (no sim)
python scripts/extract_descriptors.py \
  --input outputs/rollouts_ant.pkl \
  --output outputs/segments_descriptors_ant.pkl \
  --segment_horizon 32 --segment_stride 16

# 3. Cluster into skills + build the library (no sim)
python scripts/cluster_skills.py \
  --input outputs/segments_descriptors_ant.pkl \
  --num_skills 8 --output outputs/skill_library_ant.pkl

# 4. Evaluate target reaching by skill composition (Isaac Sim)
./isaaclab.sh -p scripts/evaluate_skill_composition.py \
  --task Isaac-Ant-v0 --skill_library outputs/skill_library_ant.pkl \
  --target_x 3.0 --target_y 2.0 --num_trials 10 --headless

# 5. Plots + research summary (no sim)
python scripts/visualize_skills.py \
  --segments outputs/segments_descriptors_ant.pkl \
  --skill_library outputs/skill_library_ant.pkl
```

### 5.1 V2 online action-set learning

V2 adds an online loop that updates the skill/action set while exploration is
running.  Instead of waiting for one fixed offline clustering pass, every
iteration collects fresh rollouts, segments them, scores candidate action
chunks by reward, descriptor novelty, displacement, stability, smoothness and
energy, stores useful chunks in an archive, then rebuilds the action set with
weighted k-means over behavior descriptors.  The descriptor metric emphasizes
outcome dimensions (`delta_x`, `delta_y`, `delta_yaw`, body-frame velocity)
more than incidental action norm / energy dimensions, which makes the clusters
less singleton-heavy during online updates.

It also trains a learning-based state-skill discriminator: a small MLP over
`(current_state_feature, skill_initial_state_feature)` pairs.  At test time,
`DiscriminatorGuidedSkillComposer` uses a hybrid applicability score: learned
MLP probability, RBF initial-state similarity, and a reliability penalty for
low-sample skills.  It penalizes or filters skills whose recorded initial-state
distribution does not match the current robot state.

```bash
TERM=xterm CONDA_PREFIX=/home/thing1/miniconda3/envs/env_isaaclab \
/home/thing1/IsaacLab/isaaclab.sh -p scripts/online_skill_learning.py \
  --task Isaac-Ant-v0 \
  --num_envs 64 \
  --online_iterations 5 \
  --steps_per_iter 8192 \
  --max_skills 16 \
  --headless \
  --output_dir outputs/online_v2_ant
```

V2 outputs:

```
outputs/online_v2_ant/online_action_set.pkl          # continuously updated action set
outputs/online_v2_ant/skill_library_v2.pkl           # V1-compatible skill library view
outputs/online_v2_ant/state_skill_discriminator.pt   # learned applicability model
outputs/online_v2_ant/online_history.json            # per-iteration update/discriminator stats
outputs/online_v2_ant/online_v2_summary.json         # final evaluation summary
```

### 5.2 Gymnasium control smoke tests

`scripts/gym_online_skill_control.py` ports the V2 online action-set idea to
standard Gymnasium continuous-control environments.  It is useful when Isaac
Lab is unavailable or when you want a fast diagnostic loop:

```bash
python scripts/gym_online_skill_control.py \
  --env Pendulum-v1 \
  --online_iterations 5 \
  --episodes_per_iter 20 \
  --eval_episodes 10 \
  --output outputs/gym_online_skill_control/pendulum_summary.json

python scripts/gym_online_skill_control.py \
  --env MountainCarContinuous-v0 \
  --max_episode_steps 999 \
  --online_iterations 5 \
  --episodes_per_iter 20 \
  --eval_episodes 10 \
  --output outputs/gym_online_skill_control/mountaincar_summary.json
```

The Gym script reports:

- `skill_mpc`: local skill-space MPC with the learned outcome model and
  discriminator/RBF applicability.
- `nearest_chunk`: state-conditioned nearest-neighbor replay from the
  action-set archive.  This is the most direct diagnostic for "does the
  learned action set preserve a converged controller?"
- `elite_replay`: replay of successful full-episode action sequences found by
  online exploration, used to diagnose whether the action set found a
  long-horizon solution even when local skill composition fails.
- `rl_policy`: the converged RL policy baseline, reported when
  `--source_policy rl` is used.
- `random_action` and `zero_action`: simple baselines.

On the current machine, MuJoCo tasks were not available, but built-in
Gymnasium tasks ran without extra dependencies.

For control-quality experiments, prefer the RL-first path: train an RL policy
to a task threshold, then extract skills/action chunks from that converged
policy's rollouts.  By default, the script refuses to compute skills if the RL
policy did not reach the target reward.

```bash
python -m pip install "stable-baselines3==2.3.2"

python scripts/gym_online_skill_control.py \
  --source_policy rl \
  --env Pendulum-v1 \
  --rl_algo SAC \
  --rl_train_steps 50000 \
  --rl_eval_freq 5000 \
  --rl_min_steps_before_stop 15000 \
  --rl_patience_evals 2 \
  --rl_target_reward -250 \
  --rl_collect_episodes 100 \
  --chunk_horizon 1 \
  --max_archive_size 20000 \
  --max_skills 16 \
  --eval_episodes 10 \
  --output outputs/gym_online_skill_control/pendulum_rl_converged_h1_summary.json
```

In the local Pendulum run, SAC reached the convergence gate after 20k steps
with final evaluation mean `-122.24`.  Extracting a single-step action set
from 100 converged-policy rollouts gave `nearest_chunk` mean return `-167.51`
versus the original `rl_policy` mean `-178.41`.  In contrast, using 16-step
open-loop chunks with `skill_mpc` remained poor (`≈ -1043`), which indicates
that high-control tasks need either short action-set elements or learned
closed-loop skills rather than long open-loop chunks.

**Expected outputs**

```
outputs/rollouts_ant.pkl                    # per-episode trajectories
outputs/segments_descriptors_ant.pkl        # segments + descriptor matrix
outputs/skill_library_ant.pkl               # skill library (+ .clusterer.pkl)
outputs/composition_eval.json               # success_rate, avg distance, ...
outputs/skill_summary.json                  # per-skill research summary
outputs/plots/skill_pca.png                 # PCA of descriptors by skill
outputs/plots/skill_histogram.png           # segments per skill
outputs/plots/skill_descriptors.png         # mean descriptors per skill
outputs/plots/skill_displacements.png       # per-skill displacement arrows
outputs/plots/composition_trajectories.png  # eval rollout traces
```

Example `skill_summary.json` entry:

```json
{
  "skill_id": 0,
  "num_segments": 1432,
  "mean_delta_x": 0.31,
  "mean_delta_y": -0.02,
  "mean_delta_yaw": 0.05,
  "mean_energy": 0.24,
  "interpretation": "forward-low-energy"
}
```

## 6. Pipeline details

### 6.1 Exploration
`RandomExplorationPolicy` samples smoothed Gaussian actions
(`a_t = α·a_{t−1} + (1−α)·ε`). `LatentExplorationPolicy` additionally holds a
per-env latent `z` mapped through fixed random projections to per-joint
offsets and gains, resampled every `latent_horizon` steps — a training-free
analogue of skill-conditioned exploration that visibly diversifies gaits.

### 6.2 Robot state extraction
The wrapper reads `root_pos_w / root_quat_w / root_lin_vel_w / root_ang_vel_w
/ joint_pos / joint_vel` from the robot articulation in
`env.unwrapped.scene`, subtracting per-env origins so positions are
comparable across parallel envs. **Fallback:** if no articulation is found,
those fields become zeros with a one-time warning and the pipeline continues
on obs/actions only (displacement descriptors then degrade — documented
limitation). Joint **torques are not recorded**, hence the energy proxy below.

### 6.3 Descriptors (13-D, per segment)
`delta_x, delta_y, delta_yaw` (displacement in the *body frame at segment
start*, making descriptors heading-invariant), `average_forward_velocity`,
`average_lateral_velocity`, `average_yaw_rate`, `mean_body_height`,
`body_height_std`, `mean_action_norm`, `mean_joint_velocity_norm`,
`energy_proxy = mean(|action · joint_vel|)` (torque·velocity if torques become
available in V2), `stability_score` (upright + steady height + survived:
`exp(−5·var(roll,pitch)) · exp(−10·std(height))`, zeroed on early
termination), and `smoothness_score = exp(−mean‖Δa‖)`.

### 6.4 Clustering
Descriptors are z-score normalized, then clustered with **k-means**
(default), GMM, or HDBSCAN (optional dependency, k-means fallback). Cluster
centers are stored in original descriptor units; each skill keeps the 3
segments nearest its center as representatives.

### 6.5 Skill composition
`GreedySkillComposer`: at each high-level step, rotate every skill's mean
body-frame displacement into the world frame at the current yaw, predict the
resulting position, and pick the skill minimizing
`‖target − predicted‖ + λ_energy · energy`; then replay the skill's
representative action sequence open-loop for H steps. `SkillMPC` extends this
to brute-force search over skill *sequences* (default depth 3 → `8³ = 512`
mean-model rollouts) with the documented cost function.

### 6.6 V2 online discriminator-guided control
`OnlineActionSet` maintains a bounded archive of action chunks during
exploration.  After every update it performs a small NumPy weighted k-means
clustering pass over the descriptor archive (NumPy is used instead of sklearn
inside the Isaac process to avoid Kit import/runtime conflicts).  Each skill
stores representative action sequences plus examples of states where those
chunks started.

`StateSkillDiscriminator` learns whether a current local state is dynamically
similar to the initial-state examples for a skill.  The feature includes
policy observation, body height/orientation, base velocity and joint state, but
not global x/y position.  Its score combines the learned classifier with an
RBF state-similarity prior so the controller remains calibrated in the small
online-data regime.  This lets final evaluation prefer skills that are both
goal-useful and likely executable from the current state.

## 7. Known limitations (V1)

1. **Open-loop skills.** Replayed action sequences do not adapt to the
   current state; transitions between skills can destabilize the robot. This
   caps composition success rate, especially on Go2/Anymal (position-offset
   action spaces tolerate replay better than Ant's torque-like actions).
2. **Mean-outcome dynamics.** Planning uses cluster-average displacements;
   variance within a cluster is ignored.
3. **Random exploration coverage.** Purely random exploration rarely produces
   fast, coordinated gaits; discovered skills are biased toward small, noisy
   displacements. The latent policy helps but does not fix this.
4. **Energy is a proxy** (`|action·joint_vel|`), not true mechanical power.
5. **Fixed-horizon chunks** can cut behaviors mid-gait-cycle.
6. Early-terminating segments are kept but flagged (`terminated_early`) and
   penalized through the stability score rather than filtered out.
7. Evaluation drives env 0 only; vectorized evaluation is future work.

## 8. Future work

- [x] Online action-set updates during exploration (`online/online_action_set.py`)
- [x] Learning-based state-skill applicability discriminator (`learning/state_skill_discriminator.py`)
- [x] Discriminator-guided skill application at test time (`control/discriminator_skill_composer.py`)
- [ ] Learned low-level skill policy `π(a|s, z)` distilled from cluster segments
- [ ] Option termination function instead of fixed H (`segmentation/fixed_horizon_segmenter.py`)
- [ ] Skill-level dynamics model `p(s′|s, k)` with uncertainty (`control/skill_mpc.py`)
- [ ] HJB-inspired value function over (x, y, yaw) for infinite-horizon skill selection (`control/skill_mpc.py`)
- [ ] Receding-horizon MPC with replanning from state mismatch (`control/skill_mpc.py`)
- [ ] Diffusion model over action chunks conditioned on desired outcome (`control/skill_mpc.py`)
- [ ] Trained diversity-driven exploration (DIAYN/DADS-style) (`exploration/latent_policy.py`)
- [ ] Terrain height-scan input; vision/depth perception (explicitly out of scope for V1)

## 9. License / citation

Research prototype; no license chosen yet. Key references: Sutton et
al. 1999 (options); Eysenbach et al. 2018, *Diversity is All You Need*;
Sharma et al. 2019, *DADS*; Cully et al. 2015, *Robots that can adapt like
animals* (Nature).
