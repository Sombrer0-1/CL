# effectiveness_v3 / Thor `thor_r2`：C01–C08 人工裁决

更新：2026-09-19。宿主 Jetson AGX Thor；allocator 配额 ≠ cgroup / 板级硬限额。
协议：`frozen_hash=9289a239c1f10ff7d32d9b6422b8dc1a467ded2516658a3b3ea668abea798e6c`，
`source_hash=264eb322e88c490030523262362cc66e6a7548acb183003363b79f892a6c49e1`。
正式矩阵 129/129 终态（completed 75，cuda_oom 54，无 implementation_error）。C 组 15 格未执行，H 组 9 格 unavailable。

本文是 PLAN §9.2 要求的人工裁决，**不是** `report.py` 自动填写的 `RESULTS.md`。不要重跑 `report` 来覆盖本文。判定口径：P/S 各下降不超过 1 个百分点；online 下降至少 5%；同范围峰值内存下降至少 10%。n=3 不作强显著性宣称。不得用测试通过、URGE 次数或完成格数替代本表。

**总括：本协议范围内没有「Orion 有效」结论。** 主比较 O11 在宽档建成场景且发生真实控制动作，但相对同预算 S0/S*/R11 塑性更差、墙时更长、峰值 reserved 更高。

数字来源：`means.csv`、`paired.csv`、`coverage.csv`、`control_events.csv`、`plugin_activity.csv`、`phase_resources.csv`、`supply.csv`、`search_cost.csv`。

---

## 0. 三层验收与场景

| 层 | 本 revision |
|---|---|
| 实现 | G1 修补与冻结前 pytest 保留；正式无 implementation_error。实现正确 ≠ 方法有效。 |
| 场景建设 | 见下表。开发盖章 `scenario_coverage.json` 把 S01 标在 q160 OOM 上，那是紧档失败，不能当作宽松 S01 已建。 |
| 方法有效 | 见 C01–C08。宽档跑完 ≠ 优于静态。 |

| 场景 | 建设 | 有效性（相对静态） | 说明 |
|---|---|---|---|
| S01 宽松、质量不足 | **宽档建成** | 不支持 | B 组 q320：O11 9/9 完成；U≈0.23>Thr0=0.05；GEM/EWC 打开，replay 207→271，batch 16→18。G2 盖章用的是 q160 OOM，与本行分开。 |
| S02 已足够好、不必扩张 | **部分** | 不支持实用收益 | CIFAR 紧档 O00/O01：U≈0.0007<0.05，插件保持关，batch/replay 微缩。P 与 S0 同量级，online 未降 5%。NC 与校准 O11 未走此分支。 |
| S03 内存近上限需收缩 | **部分可辨识** | 不支持（完成率） | 紧档 S0/S*/R11 可完成；打开 GEM 的 O10/O11/F11（及 NC 上全部 O*）在训练阶段 OOM。控制器先扩张，未见成功收缩后完成。 |
| S04 可用内存变化 | **未建**（A29） | 未验证 | C 组 `scenario_not_realized`，0 条执行，未填 0。 |
| S05 工作量增长与衰减 | **部分** | 未验证（衰减） | 宽档 O11 replay/插件状态随 experience 增长。正式 G 组无完成的 delta 半衰期配对。 |
| S06 用户偏好 | **未建** | 未验证 | D 组 9/9 紧档 OOM，无完成的偏好分化。 |
| S07 供给瓶颈 | **建成** | 见 C07 | 开发 wait_ratio≈0.52；正式 E 组 static_off 等待均值 0.515，static_on 0.002。 |
| S08 组合与开销 | **部分** | 见 C08 | 有阶段计时与插件驻留；未把预取/保护层记成 URGE 收益。 |

平台：GPU allocator 配额（紧 160 / 中 240 / 宽 320 MiB）。统一内存上的 reserved 峰值不是整机 RAM，也不是 cgroup。

---

## 1. C01 参数 / 插件权衡 — **部分支持**

**命题：** batch、replay、可选插件改变质量—时间—内存权衡。

**场景区间：** G2 静态搜索（两数据集×紧/宽各 6 格）+ 正式 A/B 的 S0/S*/R11/F11。原式与平台适配都出现在对照里；本条不依赖 O11 是否优于静态。

**证据：**

- batch：紧档 b64/b256 全部 cuda_oom；宽档 b256 全部 cuda_oom。可行区主要是 b16（宽档另有 b64 可完成）。
- replay：S* 两数据集、两档均为 b16/r2000。正式均值相对 S0（b16/r2000 vs b16/r200）：A/NC P 0.508 vs 0.600、S 0.757 vs 0.475；A/CIFAR P 0.427 vs 0.463、S 0.700 vs 0.547。方向与 light24 一致：r2000 抬 S、降 P；(P+S)/2 按规则选中 S*。
- 插件：R11（安装 GEM/EWC 但始终关）在紧档 NC/CIFAR 与宽档均完成，P 接近 S0（B/NC 0.605 vs 0.607）。F11（始终开）宽档完成但 P 更低（B/NC 0.145，B/CIFAR 0.161），online 更长。紧档 F11 与打开插件的 O10/O11 训练 OOM。
- 搜索成本单列，见 `search_cost.csv`；不把 6 格搜索墙时算进单次训练加速。

**失败：** 紧档插件格 训练阶段 OOM（不是评价阶段，与 light24 的 128 MiB 评价 OOM 不同）。

**局限：** 6 候选网格，不是连续最优；插件只有 GEM+EWC；n=3。部分支持「这些旋钮有可测代价」，不支持「Orion 选得更好」。

代表 run_id（B/NC S0/S*/R11/F11 seed0）：`effectiveness_v3_20260918T141508Z_c5bb9561`、`…T141959Z_1d7322eb`、`…T144127Z_d92f4f45`、`…T144620Z_be1c54a2`。

---

## 2. C02 自适应 URGE 价值 — **不支持（本协议范围）**

**命题：** 校准后的 O11 相对同预算可信静态改善完成率或质量—时间—内存，且发生真实控制动作。

**主证据：** 同预算 O11 vs S*，并列 S0/R11。PLAN：场景成立、动作发生、静态稳定占优 → 不支持（本协议范围）。

### 2.1 宽档（q320）：场景建成，静态占优

B 组 30 格全部 completed。配对 `both_completed=True`。O11−S0 逐 seed ΔP 全为负：

| 数据集 | seed | ΔP (O11−S0) | ΔS | Δonline_s |
|---|---:|---:|---:|---:|
| NC | 0 | −0.356 | +0.255 | +701 |
| NC | 1 | −0.321 | +0.229 | +638 |
| NC | 2 | −0.340 | +0.218 | +674 |
| CIFAR | 0 | −0.279 | +0.258 | +288 |
| CIFAR | 1 | −0.291 | +0.267 | +285 |
| CIFAR | 2 | −0.245 | +0.209 | +273 |

O11−S* 的 ΔP 同样全为负（NC 约 −0.20～−0.26，CIFAR 约 −0.22～−0.25）。相对 R11（同控制器、插件强制关）ΔP 同样为大负，说明损失主要来自打开 GEM/EWC，而不是「没搜过的 S0」。

均值（B）：

| 方法 | NC P | NC online_s | CIFAR P | CIFAR online_s |
|---|---:|---:|---:|---:|
| S0 | 0.607 | 281 | 0.465 | 89 |
| S* | 0.501 | 284 | 0.434 | 89 |
| R11 | 0.605 | 278 | 0.466 | 89 |
| O11 | 0.268 | 952 | 0.193 | 371 |
| F11 | 0.145 | 1161 | 0.161 | 429 |

质量：O11 相对 S0/S* 的 P 下降远超 1pp。时间：O11 更慢，不是下降 5%。内存：B/NC O11 reserved 峰值约 204–206 MiB，S0/S*/R11 约 152 MiB，RSS 约 2260 vs 2038 MiB，不是下降 10%。

**控制动作（真实，不是模式字符串）：** B/NC O11 seed0 `effectiveness_v3_20260918T142454Z_e358987c`：U 0.23–0.25 > 0.05；applied batch 16→18、replay 207→271、optimizer `advanced`、`gem/ewc: true`。插件 activity：experience 0 关闭，后期 GEM `before_backward` 与 EWC 状态字节上升（GEM 状态约 4.9 MiB，EWC 约 16.6 MiB）。控制器相位本身 <0.03 s/运行。

O11 相对 F11：多数 seed ΔP 为正（自适应开关略好于始终开），但两者都远差于 R11/S0。这最多说明「有时关插件比永远开好」，不能翻转主比较。

### 2.2 紧档（q160）：完成率更差，失败可归因

A 组 O11 vs S0/S*：六对 `pairing_ok`，全部 O11 `cuda_oom`、静态 `completed`，无 P/S 差值。NC：O00/O01/O10/O11/F11 共 15 格 OOM；S0/S*/R11 9 格完成。CIFAR：O10/O11/F11 9 格 OOM；O00/O01/S0/S*/R11 完成。

归因：NC 上 O* 在首步就把 GEM/EWC 打开（U≈0.15–0.23）；R11 同期 U 相近但插件强制关，跑完。CIFAR 上 O00/O01 因 L 参照仍为 30 s，U≈0.0007，插件保持关，跑完且 P≈S0；O10/O11 用 L_cal≈5.84 s 后 U≈0.13，打开 GEM，exp1 OOM。失败在训练阶段，trained=2。不是无法归因的崩溃。

完成率：O11 低于同预算 S*。不满足「完成率更高且质量在容差内」。

### 2.3 原式 O00 vs 校准 O11

CIFAR 紧档 O00 几乎不扩张，指标接近 S0，无实用时间收益。校准 O11 扩张后 OOM。这是「平台 L 校准驱动扩张」的证据，**不能**写成原论文方法在本机有效。NC 紧档 O00 同样打开插件并 OOM。

G@240 `quota_mid` 三 seed 完成，P 均值 0.217（0.167/0.191/0.292），与 320 同类「能跑完、塑性低」，不是 160 上的行为。紧档敏感性（thr/alpha/beta/lr/初始 batch/replay）21 格仍 OOM。

**覆盖缺口：** 无 host/cgroup；无 S04；无完成的衰减配对。不把本条推广为「URGE 在任意设备无效」。在本冻结协议、本两数据集、本配额网格上，主比较不支持自适应优于可信静态。

---

## 3. C03 跨场景方法权衡 — **部分支持（方向） / 不支持（Orion 胜出）**

**命题：** 权衡方向在不同数据流上可复现。

**证据：** 两数据集符号一致。紧档：打开 GEM 则 OOM，插件关的静态/R11 可完成。宽档：O11 相对 S0/S*/R11 的 ΔP 全负、Δonline 全正。S* 相对 S0 都是 P↓ S↑。

**不支持：** 不能把「方向一致」说成 Orion 在两场景都更好。紧档与宽档建设不同（160 完成率失败 vs 320 完成但质量失败），不能合成一个胜负分。不外推五基准全集。无 S04 动态内存场景。

---

## 4. C04 多算法 — **部分支持（负结果，仅 AGEM 宽档）**

**命题：** 控制器价值不限于 ER。本阶段第二算法仅为 AGEM + 可选 EWC（A10），不是五算法普适。

**场景：** F 组 NC 宽档 6/6 completed。

| 方法 | P_mean | S_mean | online_s |
|---|---:|---:|---:|
| agem_static | 0.649 | 0.368 | 422 |
| agem_adaptive_ewc | 0.369 | 0.597 | 826 |

自适应+EWC 塑性更低、墙时约 2×。与 ER 上 O11 vs R11 同号。

**未验证：** GEM 作为主算法、GSS、其他组合。历史 light24/小型 read 不得并入本表当正式证据。

代表 run_id：`effectiveness_v3_20260918T200116Z_50105d6d`（static seed0）、`…T200827Z_d913a753`（adaptive seed0）。

---

## 5. C05 用户偏好 — **未验证**

**命题：** 不同权重产生对应权衡。

D 组 9 格（latency / plasticity_stability / memory × 3 seeds）全部紧档 `cuda_oom`，与 A/NC O11 同类，无完成轨迹可比。S06 未建成。不沿用 light24「权重动了但行为不变」的本范围不支持——那是上一阶段、无插件的负载。本阶段是场景未建成。

---

## 6. C06 内存预算 — **部分支持（配额可区分） / 不支持（受压适应保住完成） / 未验证（host）**

**命题：** 预算改变完成与峰值；受压时适应有价值。

**部分支持：** allocator 配额真实起作用。q160：带 GEM 的格训练 OOM，reserved 顶到 160。q240/q320：同族 O11 完成（G mid reserved≈204–206 MiB，B O11 同）。S0 在 160 即可完成（reserved≈152 MiB，落入 80–95% 选档）。swap 采样为 0。

**不支持：** 紧档上适应没有通过收缩保住完成率；U>Thr 时先开插件。240/320 能跑完不能回写为 160 上「其实能适应」。

**未验证：** H 组 host/cgroup `unavailable`。不得把 allocator 写成 Jetson 共享内存硬限制。预算执行机制是进程内 device allocator 限额，峰值来自 `phase_resources.csv` 分阶段 reserved/allocated/RSS。

---

## 7. C07 统一预取 — **部分支持（观测） / 不支持（实用收益）**

**命题：** 同供给后端上，预取降低端到端时延且输入等价。

**场景：** S07 建成。正式 E 组 12/12 completed，仅 NC。

| 方法 | P_mean | S_mean | online_s | wait_ratio 均值 |
|---|---:|---:|---:|---:|
| static_off | 0.600 | 0.472 | 316 | 0.515 |
| static_on | 0.600 | 0.472 | 303 | 0.002 |
| orion_off | 0.234 | 0.750 | 976 | 0.179 |
| orion_on | 0.261 | 0.719 | 922 | 0.005 |

静态 on/off 的 P/S 均值相同（观测一致性）。等待从约 0.52 降到约 0.002。online 降幅 (316−303)/316 ≈ **4.0%**，未达预先 5%。RSS/reserved 无 10% 下降（reserved 均为 248 MiB 量级）。

主证据用静态轨迹，不用 Orion on/off：后者配置不等价（P 0.261 vs 0.234），预取与插件训练缠在一起。Orion 两条都远慢于静态。不把等待下降写成净加速，也不把 producer 时间与 consumer 等待相加。

代表 run_id：static_off `effectiveness_v3_20260918T175340Z_168b5058`，static_on `…T175908Z_5a9860d1`。

---

## 8. C08 系统开销 — **支持（限定）**

**命题：** 分项开销可报告，且不能把插件/评价/预取记成控制器收益。

B/NC 阶段墙时（三 seed）：

| 方法 | controller_s | reconfiguration_s | training_s | evaluation_s |
|---|---:|---:|---:|---:|
| S0 | 0 | 0 | 176–182 | 98–101 |
| O11 | ≤0.001 | 0.018–0.025 | 821–881 | 100–101 |

控制器+重配置相对训练可忽略。O11 相对 S0 多出的约 640–700 s online 几乎全在 GEM/EWC 训练（插件 extra forward，不是 URGE 四因子计算）。评价约 100 s，两方法相近。进程 RSS 约 2.0–2.3 GiB。无 swap。E 组预取未抬高 reserved。

限定：这是 Thor + allocator 160/320 + ResNet-20 + 本重建插件实现。不是论文原设备能耗，也不是机器人闭环。保护层（OOM 停止、配额）与 URGE 分开记账。

---

## 9. 与 light24 / pressure_v2 的边界

- 上表 **不覆盖、不改写** `docs/claims_status.md` 中 light24_v1 的 C01–C08。那是 RTX 5060 Ti、96 格、插件未真实切换的另一阶段。
- pressure_v2 未验收，不并入。
- thor_r1 混 source_hash，只作过程证据，不写入本裁决数字。
- 本裁决适用：Jetson AGX Thor、effectiveness_v3、`thor_r2`、CORe50-NC 与 SplitCIFAR100、allocator 配额 160/240/320 MiB、seeds 0/1/2。

---

## 10. 覆盖缺口与若继续时不该做的事

已缺：S04/C 组、H 组 host、D 组完成偏好、G 组除 `quota_mid` 外的敏感性完成轨迹、衰减配对、五算法、论文原 SKU。

不要：为让 O11 跑完而改 Q_tight；在 240 加一套 A/B；执行 C/H 并填 0；改哈希源码后仍用本 `frozen_hash`；把 129/75 或 URGE>Thr 写成有效。

若另开 revision：那是新协议，必须重校准，不能把本文件数字混进新冻结。
