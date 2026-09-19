# 结论状态：第一阶段判定与第二阶段边界

更新：2026-09-14，light24_v1 全部 96 格执行完毕（90 completed + 6 个 128MiB 首个experience训练后的评价阶段OOM 资源失败）。逐项证据见 reports/light24/RESULTS.md；下表为判定摘要。历史 71 次运行仍只作背景，不并入本表判定。

| 命题 | 当前判定 | 本轮证据与边界 |
|---|---|---|
| C01 参数/插件权衡 | 部分支持 | batch/replay 方向证据强（b256 摧毁 P_diag、r2000 稳定抬 S）；插件维度未激活，无证据；6 格/场景，r2000 为边界值 |
| C02 自适应机制有效 | 不支持（本范围） | URGE 峰值 0.006 < 最低受试阈值 0.025；optimizer/插件从未切换；实际batch16→15、replay收缩；未证明与静态ER严格等价。非证明机制在受限设备无效：当前负载/阈值未覆盖扩张或高级分支 |
| C03 跨场景方法权衡 | 部分支持 | 两场景方向一致（同资源 Orion≈静态；搜索静态占优 (P+S)/2）；CORe50 有小幅 P↑S↓ 衰减效应；不外推五基准全集 |
| C04 多算法普适性 | 未验证（范围外） | 本轮不扩展；历史 ER/GEM/AGEM/GSS 证据及 GSS 崩塌保留为限定背景 |
| C05 用户偏好 | 不支持（本范围） | 权重实现改变 URGE 数值但无任何行为/指标分化；偏好表达依赖控制器触发，本负载下不发生 |
| C06 内存预算 | 部分支持 | allocator配额生效；128MiB六例均在评价阶段失败，256/512MiB完成；未确定训练可行边界，未检验受压训练适应能力；allocator ≠ host/全 GPU 硬限制 |
| C07 统一预取 | 部分支持（观测一致性）/不支持（收益） | 逐seed完整准确率矩阵及访问计数相同；不能据此证明每步训练严格等价；开启后+31–44%时间，具体开销归因待验证；仅静态轨迹 |
| C08 系统开销 | 支持（限定） | 控制器+重配置 <0.01s/运行；评价占 online 16%/39%（CIFAR100/CORe50）；RSS ~2GB、swap 60–93MiB；分项口径见 overhead.csv |

判定用支持、部分支持、不支持、未验证；每条结论的适用范围与局限见 ../reports/light24/RESULTS.md §5。Endless 为开发划分辅助证据，其 IC 时间差异因同配置重跑也有明显计时波动，不足以归因于Orion。

**第一阶段已结束。** 本表是限定范围的阶段性判定；候选补充验证不作为本阶段未完成项，下一阶段另行讨论。

### 2026-09-15 实现复核补充

第一阶段完整Orion配置的optional_plugins=none，没有安装可切换GEM/EWC；旧运行接口仍允许记录advanced。核心实际轨迹均为default，因此原指标保留，但插件启用能力未验证，旧小型readiness的模式字符串证据不能视为真实插件执行证明。当前源码已拒绝空advanced切换；第二阶段必须显式安装插件并验证hook及状态。第一阶段仍按原范围结项，不自动重跑；原始结项版本见light24-v1-complete。

### 2026-09-15 pressure_v2 收尾

第二阶段正式 54 格在原 RTX 5060 Ti 上记录 45 格后中断并收尾，未验收，不更新上表 C01–C08。部分正式格的完成与 cuda_oom 保留为该阶段原始记录，见 reports/pressure_v2/CLOSEOUT.md。不得跨机续跑剩余格并混入同一矩阵。当前执行宿主是 Jetson AGX Thor，见 [A27](decisions/A27.md)，不更新上表 C01–C08。

### 2026-09-16 effectiveness_v3 设计与换机

[PLAN](../PLAN.md)与[SDD](SDD_effectiveness_v3.md)是第三阶段契约。RTX 5090 上的开发校准与正式跑数已按 [A26](decisions/A26.md) 放弃。本机按 [A27](decisions/A27.md) 重建环境后重新校准，不得把 5090 数字写入新冻结协议。源码修补见[实现审计](../reports/effectiveness_v3/IMPLEMENTATION_AUDIT.md)。

### 2026-09-19 effectiveness_v3 / Thor `thor_r2`

上表 **仍是 light24_v1 判定，不要用本小节覆盖。** 本机正式证据与逐项 run_id 见 [CLAIMS.md](../reports/effectiveness_v3/thor_r2/CLAIMS.md)。`RESULTS.md` 是自动表，不含本裁决。平台：Jetson AGX Thor，allocator 配额 160/240/320 MiB ≠ cgroup。正式 129 格：75 completed，54 cuda_oom。**没有 Orion 有效结论。**

| 命题 | 当前判定 | 本轮证据与边界 |
|---|---|---|
| C01 参数/插件权衡 | 部分支持 | b256 不可行；S* 均为 b16/r2000，相对 S0 为 P↓ S↑；R11≈S0；F11/打开 GEM 则宽档 P 大降、紧档训练 OOM |
| C02 自适应机制有效 | 不支持（本范围） | 宽档场景建成且有真实开关/扩 replay；O11 对 S0/S*/R11 的 ΔP 六对全负，online 约 3–4×，reserved 更高。紧档 O11 完成率低于 S*（训练 OOM，可归因于开 GEM）。L_cal 驱动 CIFAR 扩张 |
| C03 跨场景方法权衡 | 部分支持（方向）/不支持（胜出） | NC 与 CIFAR 同号；紧/宽建设不同，不合成一个胜负分；不外推五基准 |
| C04 多算法普适性 | 部分支持（负结果，仅 AGEM 宽档） | agem_static P≈0.65 vs adaptive+EWC P≈0.37；不是五算法 |
| C05 用户偏好 | 未验证 | D 组 9/9 紧档 OOM，S06 未建成 |
| C06 内存预算 | 部分支持（配额）/不支持（受压适应）/未验证（host） | 160 上 GEM 训练 OOM，240/320 同族可完成；适应未在 160 保住完成率；H 组未执行 |
| C07 统一预取 | 部分支持（观测）/不支持（收益） | S07 建成；静态 P/S 一致，wait 0.515→0.002；online 约 −4%，未达 5%；不用 Orion on/off 当等配置消融 |
| C08 系统开销 | 支持（限定） | 控制器+重配置 ≪ 训练；O11 额外墙时在 GEM/EWC；RSS ~2 GiB；非原设备能耗 |

判定用支持、部分支持、不支持、未验证。S04/C 组按 [A29](decisions/A29.md) 放弃，未验证动态预留。n=3 不作强显著性宣称。

**第三阶段已结束。** 本小节是 `thor_r2` 限定判定；覆盖缺口与外推见 [after_effectiveness_v3.md](plans/after_effectiveness_v3.md)，不作为本阶段未完成项自动续跑。
