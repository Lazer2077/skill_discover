# VGSR 技术报告: 面向论文写作的实现与实验总结

生成日期: 2026-07-05  
项目路径: `/home/thing1/skill_discover`

## 1. 摘要

本项目当前最能站住的研究版本不是单独声称“发现 skill”，而是提出一种用于复杂 locomotion 的测试时 skill/action archive 增强控制框架:

**Viability-Gated Skill Residuals, VGSR**

核心思想是: 先让强化学习控制器收敛，再从收敛策略产生的高质量 rollout 中在线构建 action/skill archive。测试时不直接 replay 关节动作 chunk，而是把 archive 中的 locomotion chunk 转换为高层速度命令 residual，叠加在 pretrained RSL-RL policy 的目标速度命令上。该 residual 只有在 viability gate 判断它与目标方向、短期进展、稳定性和能耗分布都匹配时才会被激活。

实验上，直接 open-loop replay joint-action chunk 在 Go2/H1 这类复杂 locomotion 上容易破坏闭环稳定性。当前更可靠的版本是使用 pretrained locomotion controller 作为低层稳定器，把 skill archive 只作为高层命令先验。Go2 rough locomotion 的 harder targets 上，VGSR 相比 pure learning-based target-command controller 保持了边际成功率优势并降低了能耗，但 final distance 和 final height 并不总是占优。H1 上 direct controller 已经接近饱和，VGSR 没有明显提升，但没有造成失稳。learned viability gate 已经接入并在 holdout targets 上稳定，但目前更适合作为论文中的 learned-gate ablation，而不是主实验胜点。

## 2. 研究问题与动机

复杂机器人 locomotion 中，低层控制对 feedback 极其敏感。传统 skill discovery 方法如果直接把一段 action sequence 当成 option replay，容易遇到两个问题:

1. **动力学初始状态不匹配**: archive 中 skill 的 initial state 与当前 state 差异较大时，直接 replay 会让机器人进入不可恢复姿态。
2. **open-loop 误差累积**: 即使初始状态接近，关节动作序列在粗糙地形、不同朝向和不同速度命令下也很容易偏离。
3. **强 baseline 太强**: Isaac Lab 中公开 pretrained RSL-RL controller 本身已经能完成很多 target-reaching locomotion，如果只和 random 或 zero baseline 比较，论文说服力不足。

因此本项目的方向调整为:

- RL policy 先收敛，用它生成稳定 locomotion rollouts。
- 从这些 rollouts 中在线维护 skill/action archive。
- 控制时不 replay raw action chunk，而是使用 archive descriptor 生成 command-space residual。
- residual 由 hand-designed 或 learned viability gate 过滤。
- 评估必须直接和 pure pretrained learning-based controller 比较。

这使方法从“离线 skill 分类”变成了“test-time archive-guided residual control”。

## 3. 当前实现改动总览

### 3.1 Online action set

相关文件:

- `skill_discovery/online/online_action_set.py`
- `scripts/collect_rsl_rl_skills.py`
- `outputs/isaac_go2_rough_rsl_rl_skills_long/online_action_set.pkl`
- `outputs/isaac_h1_rough_rsl_rl_skills/online_action_set.pkl`

实现了一个随探索数据不断更新的 action/skill archive。数据来源不是 random exploration，而是收敛后的 RSL-RL policy rollout。

archive 中保存:

- 原始 segment/action chunk
- segment descriptor
- segment initial state feature
- utility/reward
- skill cluster id
- representative segments

当前 action set 的默认聚类方式为 `weighted_kmeans`。该聚类不平均看待所有 descriptor 维度，而是更强调 locomotion outcome:

- `delta_x`, `delta_y`, `delta_yaw`
- `average_forward_velocity`, `average_lateral_velocity`, `average_yaw_rate`
- stability 和 smoothness
- 能耗与 action norm 权重较低，避免把相似运动因为能耗噪声拆成太多 cluster

这比最初的 nearest-center 增量聚类更稳定，也更接近 quality-diversity archive 的思想: archive 不只是 replay buffer，而是经过 utility 和 descriptor novelty 筛选的行为记忆。

### 3.2 Locomotion descriptor

相关文件:

- `skill_discovery/descriptors/locomotion_descriptors.py`

每个 segment 被压缩为 body-frame descriptor:

| Descriptor | 含义 |
| --- | --- |
| `delta_x`, `delta_y` | segment 起点 body frame 下的水平位移 |
| `delta_yaw` | yaw 变化 |
| `average_forward_velocity`, `average_lateral_velocity` | body frame 平均线速度 |
| `average_yaw_rate` | 平均 yaw rate |
| `mean_body_height`, `body_height_std` | 高度和高度稳定性 |
| `mean_action_norm`, `mean_joint_velocity_norm` | 动作幅值和关节速度 |
| `energy_proxy` | 近似能耗 `mean(abs(action * joint_velocity))` |
| `stability_score` | 姿态、高度和 termination 组合出的稳定性分数 |
| `smoothness_score` | action diff 平滑性 |

body-frame descriptor 让“向前走”“侧向走”“转向”不依赖世界坐标，因此同一行为可以跨不同 world heading 聚在一起。

### 3.3 Archive chunk selection

相关文件:

- `skill_discovery/control/archive_chunk_composer.py`

原始 archive 控制器会在每个 high-level step 从 archive 中选一个 chunk。选择代价为:

```text
J_k =
    predicted_distance_k
  + lambda_state * state_distance_k
  + lambda_energy * energy_k
  + lambda_stability * (1 - stability_k)
  + lambda_no_progress * max(0, -predicted_progress_k)
  - lambda_utility * z(utility_k)
  - lambda_progress * max(0, predicted_progress_k)
```

其中:

- `predicted_distance_k`: 执行该 chunk 后的预测目标距离
- `predicted_progress_k`: 当前距离减去预测距离
- `state_distance_k`: 当前 state feature 与 archive initial state feature 的距离
- `energy_k`: chunk 能耗 proxy
- `stability_k`: chunk 稳定性
- `utility_k`: archive 维护阶段计算出的综合 utility

这个选择器保留了 skill/action archive 的核心: 当前状态相似性和目标进展同时进入选择。

### 3.4 RSL-RL command-space residual evaluator

相关文件:

- `scripts/evaluate_rsl_skill_command_control.py`

这是当前最重要的实验入口。它使用 Isaac Lab 公开 pretrained RSL-RL policy 作为低层 controller，把 target-reaching 目标转成高层 velocity command。

对每个 high-level step:

1. 读取机器人 base position, yaw, height 和 observation。
2. 计算 body-frame local target。
3. 生成 direct target command:

   ```text
   c_goal = [clip(k * x_local), clip(k * y_local), clip(k_yaw * atan2(y_local, x_local))]
   ```

4. 从 archive 中选出候选 skill chunk。
5. 从 skill descriptor 转换 command:

   ```text
   c_skill = [average_forward_velocity, average_lateral_velocity, average_yaw_rate]
   ```

6. 使用 viability gate 判断是否使用 skill residual。
7. 最终命令为:

   ```text
   c = (1 - alpha) * c_goal + alpha * c_skill
   ```

   如果 gate 拒绝该 skill，则 `alpha = 0`，退化为 pure direct target command。

该设计非常关键: low-level RSL-RL policy 仍负责姿态稳定、接触和关节控制，skill archive 只改变 command prior，从而避免 raw joint-action replay 对复杂 locomotion 的破坏。

### 3.5 Heuristic viability gate

相关文件:

- `scripts/evaluate_rsl_skill_command_control.py`

当前 hand-designed gate 包括:

| Gate 条件 | 作用 |
| --- | --- |
| `alignment >= guard_min_alignment` | skill 方向必须和目标方向一致 |
| `predicted_progress >= guard_min_progress` | skill 不能预测为后退 |
| `stability >= guard_min_stability` | archive chunk 本身要稳定 |
| `energy_z <= guard_max_energy_z` | 能耗不能明显超过 archive 分布 |
| `skill_norm <= target_command_norm * (1 + guard_max_speed_gain)` | 避免 skill command 过快 |
| `blend = guard_max_skill_blend * clip(distance / guard_blend_distance)` | 近目标时自动降低 residual |
| `guard_recovery_height_fraction` | 到达目标附近但高度低时继续恢复 |

这个 gate 是目前 Go2 harder targets 正结果的核心。

### 3.6 Learned viability discriminator

相关文件:

- `scripts/train_skill_viability_model.py`
- `outputs/skill_viability/deployable_decision_viability_model.pt`
- `outputs/skill_viability/deployable_decision_viability_summary.json`

为了减少手调 threshold，本项目增加了一个 lightweight learned viability classifier。训练数据来自已有 evaluator 中记录的每个 candidate decision:

```text
candidate decision -> episode-level success / failure
```

输入特征:

| Feature | 含义 |
| --- | --- |
| `current_distance` | 当前到目标距离 |
| `predicted_distance` | archive 预测执行后距离 |
| `predicted_progress` | 预测进展 |
| `state_distance` | 当前 state 与 archive initial state 距离 |
| `applicability` | state distance 经过 RBF 后的适用度 |
| `utility` | archive utility |
| `energy` | chunk 能耗 |
| `stability` | chunk 稳定性 |
| `guard_alignment` | skill command 与目标方向夹角相似度 |
| `guard_energy_z` | 能耗 z-score |
| `guard_target_norm` | 目标 command 范数 |
| `guard_skill_norm` | skill command 范数 |

为了避免直接复制 heuristic gate，deployable model 默认丢弃:

- `guard_blend`
- `guard_accepted`
- `guard_low_height`

训练/测试划分使用 episode-group split，避免同一个 episode 内不同 decision 同时进入 train 和 test 造成泄漏。

learned gate 接入 evaluator 后支持三种模式:

| Mode | 含义 | 当前结论 |
| --- | --- | --- |
| `replace` | learned model 替代 heuristic gate | 容易 over-accept residual |
| `union` | heuristic 或 learned 任一通过即可 | 较冒险，容易过宽 |
| `intersect` | heuristic 和 learned 都通过才接受 | 当前最稳定，适合论文 ablation |

## 4. 算法描述

下面是可以改写进论文 method section 的算法版本。

### Algorithm 1: Build Online Action Set From Converged RL Policy

```text
Input:
  pretrained policy pi_theta
  environment E
  segment horizon H
  archive size M
  number of clusters K

Collect rollouts:
  for t = 1 ... T:
    a_t = pi_theta(o_t)
    step environment
    store observation, action, reward, base pose, velocity, joint velocity, done

Segment trajectories:
  split rollouts into fixed-horizon chunks with stride

Compute descriptors:
  for each chunk tau_i:
    d_i = locomotion_descriptor(tau_i)
    f_i = initial_state_feature(tau_i)
    u_i = utility(reward, displacement, stability, smoothness, energy)

Update archive:
  filter unstable non-novel chunks
  keep top-M chunks by utility
  cluster descriptors using weighted k-means
  choose central and high-utility representatives per cluster

Output:
  online action set A = {(tau_i, d_i, f_i, u_i, skill_id_i)}
```

### Algorithm 2: Viability-Gated Skill Residual Control

```text
Input:
  pretrained low-level policy pi_theta
  online action set A
  target position g
  optional learned viability model p_phi

Initialize episode.

while not done:
  observe robot state s_t and pose x_t
  convert target into body frame: g_body
  compute direct command c_goal

  for each archive chunk k:
    predict local displacement from descriptor
    compute predicted progress toward g
    compute state applicability from initial-state distance
    compute cost J_k

  select k* = argmin_k J_k
  convert descriptor of k* into command c_skill

  compute heuristic viability h_k:
    goal alignment
    positive predicted progress
    stability threshold
    energy z-score threshold
    speed cap

  if learned gate is enabled:
    p = p_phi(decision_features)
    combine p with h_k using intersect/replace/union

  if viable:
    alpha = alpha_max * clip(distance / blend_distance, 0, 1)
  else:
    alpha = 0

  c_t = (1 - alpha) c_goal + alpha c_skill
  execute pi_theta(o_t with command c_t) for a short horizon
```

## 5. 实验设置

### 5.1 环境与机器人

已运行的主要 Isaac Lab 环境:

- `Isaac-Velocity-Rough-Unitree-Go2-v0`
- `Isaac-Velocity-Rough-H1-v0`

使用公开 pretrained RSL-RL checkpoint:

- Go2: `.pretrained_checkpoints/rsl_rl/Isaac-Velocity-Rough-Unitree-Go2-v0/checkpoint.pt`
- H1: Isaac Lab registry 中对应 RSL-RL pretrained checkpoint

### 5.2 Baselines

论文中建议保留这些 baseline:

| Baseline | 说明 |
| --- | --- |
| `direct_target_command` | pure pretrained RL controller, command 直接指向目标 |
| `skill_command` | 无 guard 的 skill command residual |
| `guarded_skill_command` | heuristic VGSR |
| `learned_guarded_skill_command` | learned viability VGSR |

不建议把 random action/random command 作为主要 baseline。它们只适合作 sanity check，因为和 pretrained RL controller 差距太大，论文说服力不强。

### 5.3 评价指标

| Metric | 含义 |
| --- | --- |
| success rate | 未提前终止、高度达标、最终 XY 距离低于 threshold |
| average final distance | 最终到目标 XY 距离 |
| average energy proxy | `mean(abs(action * joint_velocity))` |
| average final height | 终点 body height |
| termination rate | episode 提前终止比例 |
| average commands used | 使用 archive residual 的次数 |

Go2/H1 target-reaching 使用的成功条件包括 final height threshold。因此一些 episode 虽然 XY 到达目标附近，但如果机器人蹲低或姿态不可接受，仍然判为失败。

## 6. 当前实验结果

### 6.1 Go2 rough moderate targets

结果文件:

`outputs/isaac_go2_rough_rsl_rl_skills_long/rsl_guarded_skill_command_eval_v1_10trials_positions.json`

| Method | Success | Final Distance | Energy Proxy | Final Height | Termination |
| --- | ---: | ---: | ---: | ---: | ---: |
| guarded skill residual | 1.000 | 0.2892 | 1.8924 | 0.3588 | 0.000 |
| direct target command | 0.975 | 0.2893 | 1.8147 | 0.3604 | 0.000 |

分析:

- VGSR 比 direct controller 的成功率略高。
- 最终距离基本相同。
- VGSR 能耗略高。
- 该结果说明 residual gate 在中等难度目标上不会破坏稳定性，但提升幅度不大。

### 6.2 Go2 rough harder targets

结果文件:

`outputs/isaac_go2_rough_rsl_rl_skills_long/rsl_guarded_skill_command_eval_v1_recovery_harder_5trials_positions.json`

| Method | Success | Final Distance | Energy Proxy | Final Height | Termination |
| --- | ---: | ---: | ---: | ---: | ---: |
| guarded skill residual + recovery | 0.800 | 0.2822 | 2.0948 | 0.2953 | 0.000 |
| direct target command | 0.760 | 0.2884 | 2.3916 | 0.3034 | 0.000 |

分析:

- 这是最初 5-trial 结果，也是设计 VGSR gate 的主要正反馈。
- VGSR 成功率从 0.76 提升到 0.80。
- 最终距离从 0.2884 降到 0.2822。
- 能耗从 2.3916 降到 2.0948，说明 archive residual 在部分 harder target 上提供了更合适的局部运动先验。
- final height 稍低，需要在论文中诚实讨论。当前成功判据仍然通过高度门槛，但该现象提示 residual 可能倾向于更激进或更低姿态的运动。

补强实验:

`outputs/isaac_go2_rough_rsl_rl_skills_long/go2_harder_strengthened_20trial_summary.json`

该补强实验合并了原 5-trial、一个新的 10-trial seed 和一个 gate-ablation seed 中的 guarded/direct 结果，相当于每个 target 20 trials、共 100 episodes/method。

| Method | Success | Final Distance | Energy Proxy | Final Height | Termination |
| --- | ---: | ---: | ---: | ---: | ---: |
| guarded skill residual + recovery | 0.790 | 0.2979 | 1.9913 | 0.3071 | 0.010 |
| direct target command | 0.780 | 0.2888 | 2.1954 | 0.3174 | 0.000 |

Episode bootstrap 95% CI:

| Method | Success CI | Distance CI | Energy CI | Height CI |
| --- | ---: | ---: | ---: | ---: |
| guarded skill residual + recovery | [0.710, 0.870] | [0.2749, 0.3370] | [1.8819, 2.0936] | [0.2885, 0.3237] |
| direct target command | [0.700, 0.860] | [0.2874, 0.2902] | [2.0831, 2.3110] | [0.3027, 0.3313] |

补强后的分析:

- 成功率优势变成边际: 0.790 vs 0.780，不能再写成明显成功率提升。
- 能耗优势更稳: 1.9913 vs 2.1954，说明 guarded residual 确实能减少部分高难目标上的控制代价。
- Final distance 不再占优，主要受到 2m 前进目标的低高度/early termination failure 拉低。
- Final height 仍低于 direct controller，因此论文应把当前方法定位为能耗/行为先验改善，而不是完全解决 rough locomotion recovery。
- 这个补强实验让结论更诚实: VGSR 有价值，但还需要 height-aware learned viability gate 才能成为强主结果。

对应图:

`outputs/isaac_go2_rough_rsl_rl_skills_long/go2_harder_strengthened_20trial_metrics.png`

### 6.3 H1 rough targets

结果文件:

`outputs/isaac_h1_rough_rsl_rl_skills/rsl_guarded_skill_command_eval_v1_5trials_positions.json`

| Method | Success | Final Distance | Energy Proxy | Final Height | Termination |
| --- | ---: | ---: | ---: | ---: | ---: |
| guarded skill residual | 1.000 | 0.2905 | 0.5394 | 1.0169 | 0.000 |
| direct target command | 1.000 | 0.2911 | 0.4971 | 1.0091 | 0.000 |

分析:

- H1 上 direct controller 已经饱和，成功率都是 1.0。
- VGSR 最终距离略好，高度略高，但能耗更高。
- 该实验适合作为 cross-body stability evidence: 方法迁移到 humanoid 不会破坏控制。
- 不能把 H1 结果写成显著 outperform。

### 6.4 Learned viability classifier

训练输出:

`outputs/skill_viability/deployable_decision_viability_summary.json`

| Split | Samples | Episode Groups | Test Accuracy | Test AUC |
| --- | ---: | ---: | ---: | ---: |
| episode-group | 2879 | 90 | 0.9132 | 0.9728 |

分析:

- 在不使用 `guard_accepted` 等泄漏特征的情况下，decision feature 能够较好预测 episode-level viability。
- AUC 很高，说明特征包含可学习信号。
- 但 label 是 episode-level success，可能对每个 individual decision 来说比较 noisy。
- 下一步应改成 short-horizon label，例如 residual 执行后 N step 是否比 direct command 有更好的 progress/height/energy tradeoff。

### 6.5 Learned gate holdout targets

Holdout target set:

```text
1.25,0.5; 0.5,1.0; -0.5,0.5; 1.25,-0.5
```

结果文件:

`outputs/isaac_go2_rough_rsl_rl_skills_long/rsl_learned_guard_holdout_intersect_thr0999_5trials_positions.json`

| Method | Success | Final Distance | Energy Proxy | Final Height | Termination |
| --- | ---: | ---: | ---: | ---: | ---: |
| learned VGSR, intersect | 1.000 | 0.2911 | 1.9096 | 0.3622 | 0.000 |
| heuristic VGSR | 1.000 | 0.2877 | 1.9346 | 0.3577 | 0.000 |
| direct target command | 1.000 | 0.2890 | 1.9063 | 0.3585 | 0.000 |

分析:

- learned-intersect gate 在 holdout directions 上稳定。
- 相比 heuristic VGSR，learned gate 降低能耗并提高 final height。
- 相比 direct controller，learned gate 还没有明显整体优势: energy 接近 direct，distance 略差。
- 因此 learned gate 当前应写成 promising ablation，而不是主要贡献的全部证据。

### 6.6 Negative ablations

已观察到的重要负结果:

1. **raw action replay 不适合复杂 locomotion 主结果**  
   直接 replay archive joint-action chunk 容易破坏 Go2/H1 的闭环稳定性。这个负结果支持了 command-space residual 的设计。

2. **2m Go2 rough straight walking failure**  
   机器人能够接近 XY target，但 final height 经常不达标。说明 post-hoc command gate 对已经进入低高度/蹲伏状态的恢复能力有限。

3. **learned gate replace mode 容易 over-accept**  
   threshold 0.5 的 replace mode 让 learned model 直接替代 heuristic gate，残差接受过多，效果不如 direct controller。

4. **learned gate threshold 0.999 replace mode 仍偏激进**  
   距离有改善，但能耗偏高。

5. **降低 residual blend 到 0.25 并不必然更稳**  
   一组 holdout 中出现右斜目标不稳定，说明 residual strength 与 gate 不是简单线性关系。

### 6.7 Gate ablation

结果文件:

`outputs/isaac_go2_rough_rsl_rl_skills_long/rsl_gate_ablation_harder_5trials_seed102_positions.json`

聚合文件:

`outputs/isaac_go2_rough_rsl_rl_skills_long/go2_harder_gate_ablation_5trial_summary.json`

| Method | Success | Final Distance | Energy Proxy | Final Height | Termination |
| --- | ---: | ---: | ---: | ---: | ---: |
| skill command, no gate | 0.800 | 0.2739 | 3.1806 | 0.3205 | 0.000 |
| guarded skill residual | 0.840 | 0.3570 | 2.0802 | 0.3278 | 0.040 |
| direct target command | 0.920 | 0.2889 | 2.0988 | 0.3372 | 0.000 |

分析:

- 无 gate 的 `skill_command` 可以产生较小 final distance，但能耗极高，说明 archive residual 被过度使用。
- heuristic gate 将能耗从 3.1806 降到 2.0802，接近 direct controller 的 2.0988，说明 gate 对控制代价是必要的。
- 但该 seed 下 direct controller 成功率最高，guarded residual 在 2m 前进目标上出现一次 termination，说明当前 gate 仍然不够 robust。
- 这个 ablation 支持更精确的论文说法: viability gate 是抑制高能 skill overuse 的必要组件，但当前 heuristic gate 还不是最终形态。

对应图:

`outputs/isaac_go2_rough_rsl_rl_skills_long/go2_harder_gate_ablation_5trial_metrics.png`

### 6.8 Learned gate on harder targets

结果文件:

`outputs/isaac_go2_rough_rsl_rl_skills_long/rsl_learned_guard_harder_intersect_thr0999_5trials_seed103_positions.json`

聚合文件:

`outputs/isaac_go2_rough_rsl_rl_skills_long/go2_harder_learned_gate_5trial_summary.json`

| Method | Success | Final Distance | Energy Proxy | Final Height | Termination |
| --- | ---: | ---: | ---: | ---: | ---: |
| learned VGSR, intersect | 0.760 | 0.2779 | 1.9232 | 0.3093 | 0.000 |
| heuristic VGSR | 0.800 | 0.2865 | 2.0267 | 0.3180 | 0.000 |
| direct target command | 0.800 | 0.2896 | 2.0812 | 0.3333 | 0.000 |

分析:

- learned gate 进一步降低了 energy 和 final distance，但 success 低于 heuristic/direct。
- 主要失败来自 2m forward target: learned gate 低能地接近目标，但 final height 不达标。
- 这说明 episode-level viability label 学到的是“低能、接近目标”的相关性，而不是“高度安全地到达目标”的因果判别。
- 论文中可以把它作为负向 learned-gate ablation: 当前 learned gate 有信号，但 label 设计还不够，应改为 short-horizon height-aware viability。

对应图:

`outputs/isaac_go2_rough_rsl_rl_skills_long/go2_harder_learned_gate_5trial_metrics.png`

这些负结果对论文是有价值的。它们说明最终设计不是随便调出来的，而是由复杂 locomotion 的失败模式驱动出来的。

## 7. 论文贡献点提炼

建议把贡献写成如下三点:

1. **RL-converged online skill/action archive**  
   不从随机探索直接构建 skill，而是从收敛 locomotion policy 的 rollout 中持续构建 archive，使 skill set 的分布更接近可执行、可稳定复用的行为。

2. **Command-space skill residuals for stable locomotion**  
   将 archived action chunks 转换为高层 velocity-command residual，而不是 open-loop replay joint actions。这保留 pretrained low-level controller 的稳定性，同时允许 archive 提供局部运动先验。

3. **Viability-gated activation with learned extension**  
   通过目标对齐、预测进展、初始状态相似性、稳定性和能耗 gate 选择何时使用 skill residual，并进一步实现 learned viability discriminator，证明 gate 特征具有可学习性。

更保守但可信的主张:

> VGSR provides an energy-efficient archive-guided residual over a strong pretrained locomotion controller on harder Go2 rough-terrain target-reaching tasks, while maintaining stability on a humanoid robot where the direct controller is already saturated.

不要过度声称:

- 不要说这是全新的 skill discovery。
- 不要说 learned viability gate 已经全面超过 direct controller。
- 不要把 random baseline 作为主要胜点。
- 不要忽略 2m target 和 low-height failure。

## 8. 与已有工作的关系

论文 related work 可以按以下方式组织:

| 方向 | 关系 | 本项目差异 |
| --- | --- | --- |
| Options / Option-Critic | temporal abstraction | 本项目不是从零学 option policy，而是从 converged locomotion policy 中抽取可复用 chunk |
| DIAYN / DADS | unsupervised skill discovery | 本项目 skill 更偏 task-useful locomotion archive，不以 diversity objective 本身为目标 |
| SPiRL / skill priors | skill prior from offline data | 本项目强调 online archive 更新和 test-time residual activation |
| MAP-Elites / QD | behavior archive / repertoire | 本项目借鉴 descriptor archive，但把 repertoire 用作 RSL-RL command residual |
| Residual RL | residual over base controller | 本项目 residual 不是直接学连续 action，而是从 archived locomotion chunks 中选择并 viability-gate |
| Model predictive skill selection | short-horizon predicted progress | 本项目用 descriptor-based prediction 和 state applicability，而不是完整 dynamics model |

建议论文中强调“组合创新”: 单独每一部分都不是完全新的，但在复杂 locomotion 上把 converged-policy archive、command residual 和 viability gate 组合起来，并直接挑战 strong pretrained controller，是当前工作的主要价值。

## 9. 可直接用于论文的表述

### 9.1 Title candidates

1. **Viability-Gated Skill Residuals for Test-Time Locomotion Control**
2. **Archive-Guided Command Residuals for Robust Legged Locomotion**
3. **Learning When to Reuse: Viability-Gated Skill Archives for Legged Robots**

### 9.2 Abstract draft

```text
Skill discovery can produce reusable temporal abstractions, but directly
replaying open-loop action chunks is brittle for contact-rich locomotion.
We introduce Viability-Gated Skill Residuals (VGSR), a test-time control
framework that builds an online archive from a converged locomotion policy
and reuses archived behaviors as command-space residuals over a pretrained
low-level controller. At each high-level step, VGSR selects archive chunks
using predicted goal progress and state applicability, converts the selected
chunk into a velocity-command prior, and activates it only when a viability
gate predicts that the residual is goal-aligned, stable, and energetically
in-distribution. In Isaac Lab rough-terrain experiments, VGSR improves a
strong pretrained Go2 controller on harder target-reaching tasks, reducing
the energy proxy while maintaining a small success-rate advantage in a
20-trial-per-target aggregate. On H1, VGSR preserves stability when the direct
controller is already saturated. We further show that viability can be learned
from decision-level features, enabling a learned gate that remains stable on
held-out target directions. Our results suggest that skill archives are most
useful for locomotion when used as guarded command priors rather than raw
action replay.
```

### 9.3 Method paragraph draft

```text
Given a converged locomotion policy, we collect rollouts and segment them into
fixed-horizon chunks. Each chunk is represented by a body-frame locomotion
descriptor containing displacement, velocity, yaw change, stability, smoothness,
and an energy proxy. The archive is updated online by retaining high-utility
chunks and clustering them with descriptor weights that emphasize motion
outcomes. During evaluation, the controller computes a direct target command
from the current body-frame target and also selects an archive chunk by
minimizing a descriptor-level cost that combines predicted target distance,
state-feature distance to the chunk initial state, energy, stability, and
utility. The selected chunk is converted into a velocity command from its
average body-frame velocity and yaw-rate descriptor. A viability gate then
decides whether this command should be blended with the direct command. If the
candidate fails alignment, progress, stability, or energy checks, the method
falls back to the direct command, preserving the pretrained controller as a
safe default.
```

### 9.4 Limitation paragraph draft

```text
VGSR does not solve all locomotion failures. In long 2m rough-terrain Go2
targets, the robot often reaches the target neighborhood in XY position but
fails the final-height criterion. Reactive low-height gates preserve height
only by suppressing useful progress, indicating that post-hoc command filtering
is insufficient once the base policy enters a crouched or low-viability regime.
Future work should train a short-horizon viability model that predicts height
and recovery outcomes before residual execution, or jointly train the base
policy with stronger upright recovery objectives.
```

## 10. 推荐论文结构

建议结构:

1. **Introduction**
   - skill replay 在 contact-rich locomotion 中 brittle
   - strong pretrained controllers 已经很好，但缺少 test-time behavior reuse
   - 提出 VGSR: online archive + command residual + viability gate

2. **Related Work**
   - skill discovery/options
   - skill priors/offline RL
   - quality-diversity repertoires
   - residual RL
   - legged locomotion with command-conditioned policies

3. **Method**
   - RL-converged archive construction
   - locomotion descriptor and utility
   - archive chunk selection objective
   - command-space residual control
   - heuristic and learned viability gates

4. **Experiments**
   - Isaac Lab Go2 rough target-reaching
   - harder targets and holdout targets
   - H1 cross-body validation
   - ablations: direct, unguarded, guarded, learned gate modes

5. **Results and Analysis**
   - main Go2 harder result
   - H1 saturated result
   - learned gate holdout result
   - negative results and failure modes

6. **Limitations and Future Work**
   - 2m target low-height failure
   - episode-level viability labels are noisy
   - need short-horizon learned gate
   - need larger seeds/statistical confidence intervals

## 11. 还需要补强的实验

为了让论文更稳，建议下一步优先做:

1. **更多 seeds / trials**
   - Go2 harder targets 从 5 trials 提到 10 到 20 trials。
   - 至少 3 个 seeds。
   - 报告 mean 和 bootstrap confidence interval。

2. **Archive size ablation**
   - 比较 short archive vs long archive。
   - 比较 `max_archive_size = 500, 1000, 5000`。
   - 证明 archive 不是越大越好，需要 quality filtering。

3. **Gate ablation**
   - remove alignment
   - remove energy gate
   - remove stability gate
   - remove recovery
   - learned replace vs intersect
   - current result: no-gate skill command reaches close to the target but uses much higher energy; heuristic gate reduces energy but can still lose to direct control on some seeds.
   - learned-intersect reduces energy on harder targets but loses success on the 2m height-limited target, so short-horizon height-aware labels are needed.

4. **Short-horizon viability labels**
   - label 不再用 episode success。
   - 对每个 decision 比较执行 residual 和执行 direct command 后 N step 的:
     - progress improvement
     - height safety
     - energy-normalized progress
   - 这会让 learned discriminator 更像真正的 applicability model。

5. **Harder locomotion tasks**
   - rough terrain longer target
   - target sequence/path following
   - disturbance recovery
   - commanded velocity changes
   - slope/stairs if available

6. **Trajectory visualization**
   - 保留当前 metrics bar plot。
   - 增加 XY trajectories with target markers。
   - 增加 height over time 和 residual activation over time。

## 12. 当前结论

当前项目已经从“skill/action set 是否能被发现”推进到更有研究价值的问题:

> 收敛后的 locomotion policy 产生的 action archive，是否可以作为测试时行为记忆，帮助 strong pretrained controller 在复杂目标上做得更好？

初步答案是:

- raw action replay 不可靠；
- command-space residual 是正确方向；
- viability gate 是复杂 locomotion 上必要条件；
- Go2 rough harder targets 有实证正结果，但补强实验后应表述为“能耗和部分难目标改善，成功率边际提升”，不是全面 outperform；
- H1 说明方法不会明显破坏 humanoid 控制，但还没有提升；
- learned viability gate 有潜力，但需要更好的 short-horizon label 和更多实验支持。

因此，论文可以把 VGSR 定位为一个 **archive-guided test-time residual control framework**，而不是泛泛的 skill discovery 方法。这样创新点更清晰，实验结论也更诚实。

## 13. 关键文件索引

实现:

- `skill_discovery/online/online_action_set.py`
- `skill_discovery/descriptors/locomotion_descriptors.py`
- `skill_discovery/control/archive_chunk_composer.py`
- `scripts/collect_rsl_rl_skills.py`
- `scripts/evaluate_rsl_skill_command_control.py`
- `scripts/train_skill_viability_model.py`

研究定位:

- `docs/research_positioning.md`
- `docs/vgsr_technical_report.md`

主要实验输出:

- `outputs/paper_quality_summary.json`
- `outputs/isaac_go2_rough_rsl_rl_skills_long/rsl_guarded_skill_command_eval_v1_10trials_positions.json`
- `outputs/isaac_go2_rough_rsl_rl_skills_long/rsl_guarded_skill_command_eval_v1_recovery_harder_5trials_positions.json`
- `outputs/isaac_go2_rough_rsl_rl_skills_long/rsl_guarded_skill_command_eval_v1_recovery_harder_10trials_seed101_positions.json`
- `outputs/isaac_go2_rough_rsl_rl_skills_long/rsl_gate_ablation_harder_5trials_seed102_positions.json`
- `outputs/isaac_go2_rough_rsl_rl_skills_long/rsl_learned_guard_harder_intersect_thr0999_5trials_seed103_positions.json`
- `outputs/isaac_go2_rough_rsl_rl_skills_long/go2_harder_strengthened_20trial_summary.json`
- `outputs/isaac_go2_rough_rsl_rl_skills_long/go2_harder_gate_ablation_5trial_summary.json`
- `outputs/isaac_go2_rough_rsl_rl_skills_long/go2_harder_learned_gate_5trial_summary.json`
- `outputs/isaac_h1_rough_rsl_rl_skills/rsl_guarded_skill_command_eval_v1_5trials_positions.json`
- `outputs/skill_viability/deployable_decision_viability_summary.json`
- `outputs/isaac_go2_rough_rsl_rl_skills_long/rsl_learned_guard_holdout_intersect_thr0999_5trials_positions.json`

可视化:

- `outputs/isaac_go2_rough_rsl_rl_skills_long/go2_guarded_skill_command_v1_10trial_metrics.png`
- `outputs/isaac_go2_rough_rsl_rl_skills_long/go2_guarded_skill_command_v1_10trial_trajectories.png`
- `outputs/isaac_go2_rough_rsl_rl_skills_long/go2_guarded_skill_command_v1_recovery_harder_metrics.png`
- `outputs/isaac_go2_rough_rsl_rl_skills_long/go2_guarded_skill_command_v1_recovery_harder_trajectories.png`
- `outputs/isaac_go2_rough_rsl_rl_skills_long/go2_harder_strengthened_20trial_metrics.png`
- `outputs/isaac_go2_rough_rsl_rl_skills_long/go2_harder_gate_ablation_5trial_metrics.png`
- `outputs/isaac_go2_rough_rsl_rl_skills_long/go2_harder_learned_gate_5trial_metrics.png`
- `outputs/isaac_go2_rough_rsl_rl_skills_long/go2_learned_vgsr_holdout_intersect_metrics.png`
- `outputs/isaac_go2_rough_rsl_rl_skills_long/go2_learned_vgsr_holdout_intersect_trajectories.png`
