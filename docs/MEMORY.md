# Thor 长程实验记忆（agent 必读 / 必更新）

每次会话开始先读本文和 [STATUS.md](../STATUS.md)。做出会改变冻结、revision、场景覆盖或「下一步命令」的决定后，**立刻改这一节**，不要等收尾。上下文压缩后以本文为准，不以聊天摘要里的权宜数字为准。

更新时：改「当前指针」和「不可违背」；把过时状态移到「已发生事实」；不要删决定原文。

---

## 当前指针（2026-09-19）

| 项 | 值 |
|---|---|
| 宿主 | Jetson AGX Thor，统一内存，单卡 `cuda:0`，allocator 配额 ≠ cgroup |
| 环境 | `/home/zhuzetong/miniconda3/envs/orion/bin/python` |
| 研究阶段 | **第三阶段已结项**（effectiveness_v3 / `thor_r2`）。后续选项见 [docs/plans/after_effectiveness_v3.md](plans/after_effectiveness_v3.md)；未选定轨道前无训练命令 |
| 过程证据 | `thor_r1` 95 条，混 source_hash，不写入冻结数字 |
| 冻结/正式 revision | **`thor_r2`** |
| source_hash | `264eb322e88c490030523262362cc66e6a7548acb183003363b79f892a6c49e1`（与 frozen 一致） |
| frozen_hash | `9289a239c1f10ff7d32d9b6422b8dc1a467ded2516658a3b3ea668abea798e6c` |
| 矩阵 | 设计 153；可执行 129 全部终态；排除 24（C=15 未执行，H=9 unavailable） |
| 正式进度 | **129/129**（completed 75，cuda_oom 54）。`formal_run_exit=0`。无 implementation_error。C 组 0 条执行。日志 `reports/effectiveness_v3/thor_r2/formal_run.log` |
| S04 / C 组 | 放弃建设；`scenario_not_realized`；未执行、未填 0 |
| 240 | 只做 G 组 `quota_mid`：O11 三 seed 均 completed 9/9 |
| Orion 有效性 | **不支持（本协议主比较）**。裁决见 `reports/effectiveness_v3/thor_r2/CLAIMS.md`。宽档 O11 对 S0/S*/R11 的 ΔP 全负；紧档 O11 训练 OOM。不是「未跑完所以未验证」 |

正式分组（3 seeds）：

| 块 | n | completed | cuda_oom |
|---|---:|---:|---:|
| A/NC 紧档 160 | 24 | 9 | 15 |
| A/CIFAR 紧档 160 | 24 | 15 | 9 |
| B/NC 宽档 320 | 15 | 15 | 0 |
| B/CIFAR 宽档 320 | 15 | 15 | 0 |
| D 紧档偏好 | 9 | 0 | 9 |
| E IO | 12 | 12 | 0 |
| F AGEM 宽档 | 6 | 6 | 0 |
| G 敏感性（含 mid=240） | 24 | 3 | 21 |

B/NC 三 seed 均值（means.csv）：S0 P≈0.607，R11≈0.605，S*≈0.501，**O11≈0.268**，**F11≈0.145**。配对 O11−S0 的 ΔP 约 −0.32～−0.36。跑完宽档 ≠ 优于静态。

G@240 的 O11 三 seed 均跑完（P≈0.167 / 0.191 / 0.292），与 320 同类「能跑完但塑性低」，不是紧档 160 上的行为。

裁决：`reports/effectiveness_v3/thor_r2/CLAIMS.md`。自动表 `RESULTS.md` 仍是占位。light24 的 C01–C08 表不得覆盖。第三阶段已结项。

冻结数字（`thor_r2` 单一 source_hash，不是 thor_r1）：

| 数据集 | eval_batch | Q_tight | Q_loose | Q_mid | L_cal（秒） | 紧/宽 S* |
|---|---:|---:|---:|---:|---:|---|
| CORe50-NC | 128 | 160 | 320 | 240 | 15.872 | 均为 b16/r2000 |
| SplitCIFAR100 | 128 | 160 | 320 | 240 | 5.842 | 均为 b16/r2000 |

紧档静态仅 b16 可行（2/6）；宽档 b16+b64 可行（4/6），b256 OOM。S01/S04 盖章均在 q160、trained=2 后 `cuda_oom`。S07 wait_ratio≈0.52 已建成（不是预取净加速）。

下一条：阶段已结项。后续先读 `docs/plans/after_effectiveness_v3.md`。默认 T0 只读澄清；T1–T3 另开阶段。不要重跑 129、不要改哈希源码、不要执行 C/H。

---

## 不可违背（违反即行为偏离）

1. **用户明文优先于 PLAN / SDD / 本文旧段。** 三项冻结决定见 [A29](decisions/A29.md)。
2. **身份：** 写入 frozen 的 eval_batch / Q / L_cal / S* 不得混用不同 `source_hash`。混用 + 脚注 ≠ 闭包。
3. **停改哈希源码：** 冻结后仍禁止改 `src/orion_repro`、`tests`、`PLAN.md`、`AGENTS.md`、`docs/SDD_effectiveness_v3.md`、非生成 `configs/`、`experiments/`（revisions 除外）、锁文件。改了就必须新 revision 重校准。
4. **thor_r1 只作过程证据。** 95 条全部保留。
5. **不要为了让 Orion 跑完而改 Q_tight。** 160 上自适应 OOM 是结果。
6. **不要为了建成 S04 把预留换到 240/320。** 宽档内存摆动必须另开 revision、另起名字。
7. **不要在 240 加一套 A/B。** B=320 是宽松完整比较；G@240 只问配额上调后 O11 行为是否同类。
8. **C 组不执行、不填 0、不说动态预留已验证。** H 组 unavailable。
9. **时间估算不是杀训练的理由。**
10. **解释器、wandb、压力入口：** `ORION_PY=.../envs/orion/bin/python`；`WANDB_MODE=disabled`；禁止 `pressure_*`；禁止覆盖历史报告。
11. **未要求不 commit / 不 push。**
12. **三层验收分开：** 实现正确 ≠ 场景建立 ≠ 方法有效。主比较已裁决为不支持（本协议范围）；仍禁止把完成格数或 URGE 触发说成有效。

---

## 已发生事实

### thor_r2 身份校准（已冻结）

循环 18:52–20:36 自然收束（波 17 后 n_probes=0）。61 行探针，单一 source_hash，无 implementation_error。S01/S04 身份盖章 OOM 留证。已写 g2_review、freeze、emit。

### thor_r1（过程证据，勿混入冻结）

95 条，混 source_hash。只读数字与 thor_r2 冻结值接近（Q 同为 160，L_cal 约 15.915/5.924），但不得当作冻结来源。

light24_v1 已结项；pressure_v2 中断未验收；5090 校准已放弃（A26）。

---

## 更新日志

- 2026-09-19：用户确认第三阶段结项。写入 `docs/plans/after_effectiveness_v3.md`。准备入库 push。未改 `thor_r2` 冻结数字。
- 2026-09-19：写入 `reports/effectiveness_v3/thor_r2/CLAIMS.md`。C02 不支持（本范围）；C01 部分支持；C05/S04 未验证。未改哈希源码，未重跑 129。
- 2026-09-19 06:21：正式 129/129 终态，`formal_run_exit=0`。已 `report --revision thor_r2`。completed 75 / cuda_oom 54。C 未执行。当时 C01–C08 尚未裁决。
- 2026-09-18 23:58：正式 58/129。B/NC 宽档 seed0+seed1 十格全跑完；F11 seed1 p_diag≈0.057。当前 O11/seed2。未宣称 Orion 有效。
- 2026-09-18 23:12：正式 54/129。B/NC 宽档 seed0 五格全跑完；O11/F11 p_diag 低于 S0/R11。当前 O11/seed1（已过紧档 OOM 点）。未宣称 Orion 有效。
- 2026-09-18 22:41：正式 51/129。A 组 48 格齐。CIFAR 紧档 3 seeds：O00/O01 跑完，O10/O11/F11 OOM。B/NC 宽档 S0、S*、O11 seed0 均 completed；O11 在 320 跑完 9/9（160 上 OOM）。当前 R11。未宣称 Orion 有效。
- 2026-09-18 21:46：正式 24/129。A/NC 紧档 24 格 3 seeds 齐（S0/S*/R11 完成；O 四格与 F11 全部 cuda_oom）。当前 CIFAR 紧档 S0/seed0。未宣称 Orion 有效。
- 2026-09-18 21:02：正式 8/129。A 组 NC 紧档 seed0 八格齐：S0/S*/R11 完成；O 四格与 F11 cuda_oom。当前 S*/seed1。未宣称 Orion 有效。
- 2026-09-18 20:56：正式 6/129。A 组 NC 紧档 seed0：S0/S* 完成；O00/O10/O01/O11 均 cuda_oom（trained=2，160 MiB）。当前 R11。未宣称 Orion 有效。
- 2026-09-18 20:53：正式 3/129。S0 与 S* seed0 跑完 9/9；O00 seed0 在 exp2 `cuda_oom`（配额 160 MiB）。当前 O10。未宣称 Orion 有效。
- 2026-09-18 20:40：身份循环结束；calibrate/g2_review/freeze/emit 完成。frozen_hash=`9289a239…`；153=129+24。启动正式 `run --revision thor_r2`。未宣称 Orion 有效。
- 2026-09-18 19:05：波1–4 完成；波5 配额扫描跳过 plugin_cost。
- 2026-09-18 18:55：波1–2 eval_batch 完成。
