# 论文审查与修改记录

## 结构与论述审查（顶会视角，2026-07-17；续修 2026-07-18）

按 CoRL / RSS / NeurIPS robotics track 审稿标准，此前版本的核心问题不是数字错误，而是**论文身份摇摆**与**论述层级混乱**。

### 拒稿级结构问题（已修）

1. **三重 thesis 并存**：标题与贡献强调 adapter necessity audit，Related Work 与 Figure 1 却以 VGCC 为中心，Introduction 又先卖 identification。审稿人会问：“这是 identification 论文、控制论文，还是方法论审计论文？”→ 已统一为：**audit 是贡献主体，identification 是可选特征源，VGCC 是 case study**。
2. **Related Work 与主贡献错位**：原稿用约两页为 VGCC 辩护（IMC / MBRL / MPC / governor / options），真正支撑 audit 的 treatment-choice / policy-value 文献被埋在末尾。→ 已压缩为四段，并把 treatment choice 提到核心位置。
3. **实验叙事与贡献声明不一致**：原稿写 Q1=core method (identification)，Q3 才是 central question。→ 已改为 Q3 明确标为 central；Q2 降为 case-study frontier placement。
4. **重复论述过载**：同一组数字（1.0–1.9%、5.2%、0.55%、Abstain）在 abstract / intro / results / limitations / conclusion 以近乎相同句式重复五次。→ Conclusion 与 Limitations 大幅压缩；Intro 贡献列表去重。
5. **过度对冲导致 claim 不可读**：每段“we do not claim X; instead …”使审稿人抓不住可检验主张。→ 改为先陈述可检验主张，再一句话限定范围。

### 2026-07-18 续修

6. **实验小节顺序与 Q2→Q3 声明不一致**：Coverage / Ablations 原在 Adapter Necessity Audit 之后，读起来像审计完又回头卖 VGCC。→ 已重排为 Baselines → Coverage → Ablations → **Audit（高潮）**；Ablations 末增加桥接句，明确消融只解释 *how*，不证明 allocation value。
7. **VGCC 视觉权重过高**：Algorithm 1 占正文栏。→ 已下沉至 Appendix（仍保留 `alg:vgcc` 交叉引用）；方法图移入 VGCC 小节，不再出现在 Method 开篇。
8. **Keywords / Model 开篇仍偏 identification-controller**：→ keywords 改为 audit / counterfactual evaluation；Model 开篇改为同时服务 adapters *and* audits。
9. **Coverage / H1 段落过长且像方法推销**：→ Coverage 改为 “before auditing recoverability, check artifact”；H1 压缩为 morphology-adapted 限定。

### 语言 / 论述问题（已部分修）

- 长句过多（40+ 词嵌套从句）；已缩短 Scope、VGCC eligibility、Baselines、Conclusion、H1。
- “deliberately / exactly / rather than” 模板化对比过多；已删减。
- Abstract 信息密度过高且中心句靠后；已改为先问 adapter 是否值得建，再给 audit 定义与诊断结果。
- Method 开篇仍按 model→audit→VGCC 的实现顺序叙述，与“audit 优先”的贡献层级冲突；已改 framing（audit primary），实现顺序保留以便读者跟实验管线。

### 仍可能被顶会抓住的内容问题（未声称已解决）

- 全部仿真、无硬件功耗；proxy ≠ energy（Limitations 已诚实披露）。
- H1 为 morphology-adapted，非 zero-shot；seed 数对 family-level 推断仍偏少。
- Audit 在单一 domain 返回 Abstain，缺少第二 domain 的已知 recoverable-headroom 正对照。
- 若投严格双栏 page limit，可考虑再把 Coverage 表或 hetero 地形表下沉附录。

\subsection 2026-07-18 续修（画图 + 内容缺口）

10. **Audit pipeline 主文图**：新增 `figures/audit_pipeline.png`（`scripts/make_audit_pipeline_figure.py`），置于 Method §audit 开篇，视觉主线改为 existence ≠ recoverability ≠ allocation。
11. **校准图填补决策规则正/负对照缺口**：新增 `figures/audit_calibration.png`。纠正旧文“γ=5% gain channel clears δ”的不严谨表述：mean 过 δ 在 5%，**gain-only GO（$L_H>\delta$）在 9%**；joint rule 全程 Abstain。Hidden 映射在 δ=1% 零 false GO。诚实声明：这是决策规则的 known-signal domain，**不是**第二机器人 domain。
12. **第二 embodiment 缺口的部分填补**：Stage 1 明确标为 Go2+H1 双 embodiment screen；Stages 2–3 exact-replay 仍仅 Go2，Limitations 写明确认路径。
13. **Hetero 表下沉附录**（`app:hetero`），为主文腾出校准图空间。
14. **仍未填**：硬件功耗；H1/ANYmal exact-replay Stages 2–3（需新仿真采集）。

15. **H1 Stages 2–3 exact-replay（最大缺口）**：完成管线移植（stub macro/TV 仅驱动 episode；observation-only 审计；`--height-scan-slice 69:256`）。4 seeds（870–873，192 queries，replay L2=0）：oracle −4.18%，selector vs mixture −0.33%（CI [0.04,0.66]），δ=1% → **No-Go**。已写入主文；875+ 继续采集中。诚实披露：stub 不参与决策特征。

## 总体判断

当前工作具备一个可辨识的方法主线：对冻结策略的“策略—机器人闭环”建立命令响应模型，再以进度约束、姿态门控和有界修正进行在线搜索。这个组合有一定科研价值，也比单纯经验库检索更完整。但按机器人学习主会/主刊标准，现有证据仍不足以支撑“真实节能”“安全”“统计等效”或“广泛通用”的强结论。更合理的定位是有潜力的 simulation study / workshop 或预印本；若要冲击较强 venue，需要补真实能耗指标、强基线、按 seed 聚类的统计推断和硬件或更广任务验证。结构修订后，论文至少具备清晰、可辩护的主 claim：**existence ≠ recoverability ≠ allocation**。

## 1. Figure 审查

- 方法总图的信息流合理，offline identification 与 online deployment 区分清楚。原图注过长、重复正文，已缩短为只解释读图所需的信息。
- paired-difference 图比柱状图更合适，能够显示 seed 间变异。原图中的 `p < 10^{-5}` 不准确；17/17 同方向的双侧精确符号检验为 `p = 1.53e-5`，已修正，并说明该检验混合了不同任务族，只能说明方向一致性。
- trajectory 图使用单一 seed，适合作为定性控制行为展示，但不能证明总体路径等价。正文仍把它限定为 control example；建议终稿进一步加入跨 seed 的终点误差分布或轨迹距离统计。
- 单 episode 图存在选择性展示风险。已在图注和正文中明确其为刻意选择的机制示例，并把总体证据指回表格。
- 现有图均为高分辨率，PDF 中没有明显裁切或重叠；但 8 子图轨迹图在单栏缩放后的字号偏小。正式投稿建议改成双栏宽图、只保留 4 个代表目标，其余放附录。

## 2. 公式与算法审查

- 直接命令公式原先的 `clip` 没有上下界，无法复现。已改为分别标明纵向、横向速度和角速度界限。
- 原文将 `mean(|a * qdot|)` 称为 mechanical power / energy，但实现中的 `a` 是策略动作而不是关节力矩，因此量纲和物理解释不成立。已统一限定为无量纲 actuation-effort proxy，并在方法、实验和局限性中明确不能解释成机械能、电能或 COT。
- 原算法未定义可行候选集为空时的行为。已补充回退到直接命令的规则。
- 参数计数原文写“7 个”，但包含 `delta` 后实际为 8 个，已修正。
- H1 并非只修改一个 posture floor：实验记录显示 eligibility、rescue、restore 等多个姿态阈值发生调整。已改为 morphology-specific posture thresholds，并明确这是看到 H1 失败后进行的适配，不是 zero-shot transfer。
- 有界修正公式本身清楚；建议后续在附录完整列出候选网格、所有命令上下界和三种机器人的具体姿态阈值，避免只靠代码复现。

## 3. 科研价值与 publishability

### 有价值的部分

- 对冻结 RL 控制器进行 command-level closed-loop identification，避免 action-level dynamics 的高维建模，问题设定清楚。
- 将 progress 作为约束、effort 作为目标、posture 作为 viability channel 的结构具有可迁移的设计逻辑。
- 同一 quadruped 配置从 Go2 转到 ANYmal-D，以及 gate / progress floor / bounded correction 的消融，为方法机制提供了一定证据。

### 目前最可能导致拒稿的部分

- 核心优化指标不是真实能耗。仅凭 action 与 joint velocity 的乘积无法宣称 energy efficiency。
- baseline 太弱。只有 proportional direct controller 和作者自己的 retrieval variant，缺少 tuned command scaling、reference governor、同预算 sampling MPC、以及使用真实 simulator state/torque 的 oracle 或上界。
- 统计单位处理不充分。episode bootstrap 忽略 episode 嵌套于少量 seeds 的相关性；CI 重叠不能证明 success 等效。已将相关表述降为 descriptive，并建议预注册 non-inferiority margin 后做 cluster/hierarchical bootstrap。
- H1 存在适配后的测试泄漏风险；必须在新的 humanoid task、seed 或第二种 humanoid 上确认。
- 全部结果来自模拟 target reaching。尚无硬件、扰动、路径跟踪、速度计划或 manipulation 结果，不能宣称适用于任意 command-conditioned policy。
- 方法与 learned reference governor / one-step sampling MPC 接近。论文需要更清楚地证明“predictive posture gate + bounded inversion”相对这些直接基线带来的独立增益。

## 已直接完成的修改

- 纠正双侧 sign-test 数值。
- 删除“CI 重叠即 statistically matched/equivalent”的表述。
- 将 energy 主张改为 actuation-effort proxy，并增加 validity limitation。
- 删除“最坏只会低效、不可能失衡”的错误安全暗示。
- 更正 H1 调参事实和参数数量。
- 补全公式边界和空可行集回退。
- 缩短方法图图注，标明单 episode 图的选择性与定性用途。
- 增加 statistical validity、energy validity、baseline/tuning 三类局限。
- 重新组织主文与附录：正文 15 页，Appendix 3 页，参考文献 2 页；没有公式越界、图表裁切或未解析引用。

## 投稿前建议的最低补实验集合

1. 用 simulator joint torque × joint velocity、总机械功、单位距离能耗/COT 重新评估，并报告 episode duration 与距离，防止“走得慢所以每步 proxy 低”的混淆。
2. 增加 tuned scalar command scaling、measured-height reference governor、无 gate response-model search、同候选预算 MPC 四类基线。
3. 以 seed 为 cluster，预先定义 success non-inferiority margin，报告 paired cluster bootstrap CI；每个 family 至少增加到约 10 个独立 seeds。
4. 在未用于任何阈值调整的新 H1 targets/seeds 上做一次锁定配置的确认实验。
5. 至少增加一种 task interface（路径跟踪或扰动恢复）；若目标 venue 强，加入硬件功率测量。

## 后续实验更新（C → D → A）

- C：完成 5-member ensemble 与 episode-group conformal calibration。离线 minimum-height 下界达到 95.64% test coverage，但两 seed 在线实验没有改善 minimum height 或 success，rescue 触发反而增加约 68%。论文将其作为负消融，不声称 uncertainty-aware gate 提高安全性。
- D：新增 maximum tilt 与 maximum angular speed 标签，预测 $R^2$ 分别为 0.797 和 0.584。nominal benchmark 上变化很小；在 static/dynamic friction 固定为 0.35/0.25 的单 seed stress test 中，多维门控保持相同 success，并将最大倾角从 0.321 降至 0.287 rad、最大角速度从 3.04 降至 2.71 rad/s。该结果有希望，但样本不足以形成一般安全结论。
- A：完成 3 个新 seeds、48 episodes/method 的 applied-torque audit。VGFC 的绝对机械功下降 8.2%，平均机械功率下降 16.1%，单位路径能耗/COT 下降 6.3%；同时任务耗时增加 10.9%，success 为 0.771 vs 0.792。该结果支持模拟机械效率，但仍不等于硬件电能。

## Objective alignment 与强基线复核

- 新增 applied-torque mechanical-power 响应标签，5-member ensemble 的 held-out $R^2=0.797$、MAE 13.0 W。
- 直接最小化 mechanical power、以及 mechanical power / predicted progress，都没有稳定降低 episode mechanical work；降低瞬时功率会被更长完成时间抵消。
- 锁定三 seed 验证中，power-VGFC 和 proxy-VGFC 的 COT 分别比 direct 高约 4.5% 和 3.6%；fixed 0.75 scaling 的 COT 低 6.6%，但耗时增加约 40%。
- 使用原论文 response model 在另外两个 seeds 上复核，VGFC 的总机械功变化仅 −0.05%，而 0.75 scaling 为 −11.9%，同样伴随约 38% 时间增加。
- 单 seed scaling sweep 中，0.90 scaling 与 VGFC 耗时相近（2.14 vs 2.10 s），但 COT 更低（0.684 vs 0.734）。因此当前 VGFC 没有证明位于 efficiency–time Pareto 前沿。
- 这推翻了“proxy reduction 足以证明机械效率”的强解释。论文核心已改为 objective-misalignment 与 falsification audit；当前版本不再将 VGFC 描述为已验证的最终节能算法。

## Execution-Aligned VGFC 实例改进

- 原算法预测完整候选命令，却执行候选与 direct command 的 bounded blend，导致模型约束和 cost 不是针对真正执行的命令。
- 新版本先构造实际可执行命令，再直接预测其 progress、cost 和 stability，并测试 proxy 与 mechanical-power/progress 两种目标。
- 第一个开发 seed 上，execution alignment 将同模型 COT 从 0.932 改善到 0.905，但仍被 0.90 scaling 的 0.890 支配。
- 第二个开发 seed 上，aligned proxy/physical 目标 COT 为 0.752/0.737，仍高于 direct 的 0.705 与 scaling 的 0.716。
- 因此该修改是理论上正确、实验上局部有效的实现修复，但没有形成稳定的效率优势。未继续做结果导向调参，也未写成多 seed 正结论。

## Task-Level VGFC 实例与在线否证

- 新增任务级三头模型，输入当前观测、局部目标与候选命令，预测从当前时刻到回合结束的剩余机械功、剩余时间和成功概率。数据来自 3 个采集 seeds、144 个混合控制器回合、4,298 个高层样本。
- episode-grouped held-out 指标为：剩余功 MAE 15.7 J、$R^2=0.846$；剩余时间 MAE 0.362 s、$R^2=0.654$；成功 AUC 0.819、Brier 0.141。
- 在线规则在原有短期进度/姿态门控之外，要求候选的成功概率不比 direct 低 0.02 以上、预测时间不超过 direct 的 1.10 倍，并最小化预测剩余机械功。
- 三个后续开发 seeds（每方法 48 回合）中，task-level VGFC 为 success 0.729、work 140.64 J、COT 0.802；固定 0.90 scaling 为 0.771、130.38 J、0.755，仍在主要指标上支配学习式反演。
- 为排除约 70 个候选导致的模型外推，又实现了只允许训练数据中 $\{0.60,0.75,0.90,1.00\}$ 四档的 adaptive-scaling VGFC。新 seed 上它仍被固定 0.90 scaling 支配（success 0.750 vs 0.875；work 121.92 vs 111.07 J；COT 0.711 vs 0.649）。
- 该实验给出比“短期目标错位”更深的结论：behavior distribution 上平均预测准确，不保证干预状态上的候选排序正确；混合 continuation controller 产生的 value label，也不是新 switching controller 的 policy-consistent value。论文据此要求后续工作使用策略一致的数据或 off-policy correction、决策状态 ranking/calibration、保守不确定性约束，并预注册对 fixed-scaling Pareto frontier 的比较。

## Seed-Matched Episode Ranking 审计

- 新增重播种的固定增益 episode block 协议，并发现 Isaac Lab 启动后的第一个 rollout 仍有历史缓冲伪差；后续精确回放审计又证明 terrain curriculum 未冻结时该协议并非严格同状态配对。因此论文把它降级为 seed matched diagnostic，同时报告剔除启动伪差块的结果和包含它的敏感性结果。
- 五个采集 seeds 的 40 对 $0.90$ vs direct 回合显示，在 success 不低于 direct、time 不超过 direct 的 1.10 倍约束下，paired oracle 有 16/40 次应选 $0.90$，可将 work 从 132.28 J 降到 128.15 J，success 保持 0.800，time 增加 3.4%。
- policy-conditioned 五模型 ensemble 的 leave-one-seed-out work $R^2$ 平均为 0.821，但 40/40 都选择 direct。模型错误地预测 $0.90$ 的 work/direct 比为 1.034，而真实聚合比为 0.940；绝对 return 的时间变化信号掩盖了档位间的小 treatment effect。
- 改为直接学习配对标签后，leave-one-seed-out ranker 选择 21/40 次 $0.90$，success 0.800、work 128.28 J，接近 paired oracle。随后锁定模型，在全新 seed 825 做 5 trials × 8 targets。
- 剔除唯一的 simulator-startup 伪差块后（$n=39$/method），ranker 为 success 0.872、work 120.88 J、time 1.936 s、COT 0.696；direct 为 0.872、123.39 J、1.848 s、0.711；fixed 0.90 为 0.872、115.22 J、2.066 s、0.667。
- 关键否证是 matched-mixture baseline：按 ranker 相同的 15/39 比例随机混合 $0.90/1.00$，期望 work 120.25 J、time 1.932 s、COT 0.694，均略优于状态条件 ranker。因此 ranker 只形成速度—能耗折中，没有证明利用了状态信息。
- 后续若继续，需要更多严格配对的 intervention states、直接 treatment-effect/ranking 损失、nested-seed validation 与 uncertainty-aware abstention，并必须同时比较 fixed-scaling frontier 和 matched randomized mixture。

## 递归 Completion MPC 与精确回放 DAgger

- 新增 5-member macro-transition ensemble，以 4 个低层步（0.08 s）为一个宏步，递归枚举 $\{0.75,0.90,1.00\}^4$ 的 81 条计划；规划跨度 0.32 s，每步重规划。模型输出完整 observation residual、位移/偏航、mechanical power、minimum height、maximum tilt 和 angular speed。
- 四宏步递归误差为 displacement MAE 3.3 cm、累计 work MAE 5.17 J、minimum-height MAE 8.8 mm；在线 81 计划推理约 2.72--2.73 ms/高层决策。
- Isaac Lab 的 `reset_to` 只恢复 scene state，不能完整恢复 observation/action history。采用“同 seed 从 episode 起点重放 MPC gain 前缀”的严格反事实协议；同时发现 rough-terrain curriculum 会使同 seed 的重复 reset 改变地形等级。冻结 curriculum 后，全部 215 个分支的 query-observation 重放 L2 误差精确为 0；未冻结时冒烟实验均值为 2.92。
- 两轮共收集 215 个候选前缀 + direct 收尾分支，对应 1,075 个 ensemble terminal inputs。macro-predicted terminal input 与真实输入的 L2 误差中位数 6.73、90 分位 11.56，直接证明原 MPC 存在 model-induced terminal distribution shift。
- 第一轮纠偏模型在下一轮全新 seed 的 113 个反事实分支上只将 work MAE 从 17.08 降到 14.85 J；time MAE 从 0.253 变差到 0.311 s，success Brier 从 0.211 变差到 0.217。训练/随机 episode split 上的校准改善不能替代跨 seed 决策校准。
- 两次独立闭环开发验证（各 $n=16$/method）结论一致：MPC 相对 direct 仅少 0.65/0.43 J，却慢 0.12/0.20 s，success 均低 6.25 个百分点；相对 fixed 0.90 多 8.55/8.01 J。精确回放校正修复了真实的终端分布偏移，但没有形成 Pareto 改进。
- 因此当前仍不能把该工作写成“已实现真实节能的 completion-aware 算法”。可以站住脚的新核心是：用严格反事实重放将 query-state mismatch 与 recursive model shift 分离，并实证说明 terminal DAgger 单独不足，剩余瓶颈包括跨 seed terminal-value 泛化、机械功率 head 的中等精度和姿态误差累积。

## 精确回放前缀结果校准与候选收缩

- 在 3 个新 seeds 上收集 156/157/162 个严格前缀分支，共 475 个；冻结 terrain curriculum 后 query observation 的最大重放误差继续为 0。
- 五模型 residual calibrator 在前两个 seed 训练、第三个 seed 严格留出。prefix progress MAE 从 0.0237 m 降到 0.0127 m，$R^2$ 从 0.874 升到 0.962；prefix work bias 从 +1.59 J 降到 −0.07 J。
- 关键排序指标没有改善：best-sequence accuracy 均为 0.854，pairwise work-ranking accuracy 从原模型 0.884 变为 0.879。因此论文将其表述为 progress/bias calibration，而不是更强的通用排序器。
- 完全未参与训练或模型选择的 seed 857 上，prefix-calibrated MPC 与 direct/fixed-0.9 的 success 均为 0.688，但 work 为 149.89 J（direct 148.10 J、fixed-0.9 135.37 J），time 为 2.010 s（1.935/2.150 s），没有 Pareto 改进。
- 事后移除已反复被支配的 0.75 候选后，prefix-calibrated MPC work 改善到 145.38 J、time 2.050 s，但仍比 fixed-0.9 多 10.01 J；在相同平均时间下，fixed-0.9/direct 端点线性插值约为 141.29 J，仍优于学习式切换。
- 新增严格机制检验：三个精确回放 source seed、144 个同状态 query 中，oracle 在每个候选前缀后统一执行 direct continuation。在不增加剩余时间的约束下，pooled work 下降 3.1%；允许剩余时间至多增加 10% 时下降 5.1%，且三个 seed 均为正向。这不是可部署控制器结果，也不比较 fixed scaling；它证明 benchmark 存在局部状态自适应空间，并将瓶颈定位为 treatment-effect ranking。
- 对 oracle 标签做固定的 leave-one-source-seed-out 线性可分性 probe（无 hyperparameter sweep）：仅用状态/目标特征的准确率为 0.472，仅略高于 all-direct 多数类的 0.424，实际 work 仅变化 −0.3%，且 13.2% query 违反 success/time 约束；加入宏模型预测均值与方差后准确率降至 0.438、work 反增 0.5%、违约率升至 18.1%。这不能排除更强的非线性 treatment-effect learner，但排除了“简单状态线索或现有宏预测足以恢复 oracle”的解释。
- 该实验进一步强化论文的核心诊断：修正单个预测通道、甚至显著降低其留出误差，不等于获得干预候选的 treatment-effect ranking，也不等于越过简单 scaling frontier。
- 进一步使用五个冻结 curriculum source、240 个严格回放 query，直接拟合相对 direct 的 work/time/eligibility treatment effect。留一 source 验证的候选 work MAE 为 12.2\%、time MAE 为 7.9\%、eligibility accuracy 为 71.9\%；保守选择仅在 12.9\% query 激活、实际 success/time 违例率为 1.7\%，但实际 work 仅下降 0.54\%。这说明额外数据和相对标签改善了校准与约束，却还不能恢复 5.1\% oracle 空间；下一瓶颈是可部署的 state dependent treatment effect，而不是阈值调参。
