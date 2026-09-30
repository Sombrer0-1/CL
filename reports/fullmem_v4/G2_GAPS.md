# G2 可执行 vs 未建立登记（非正式）

日期：2026-09-24。对应 PLAN §6 的 1320 设计名额。这是覆盖分母，**不是可执行正式队列**，也不是 G3 冻结。

已验收启动档：`mem=64G` / `mem=32G` / `mem=16G` / `mem=8G`。准入 ≠ 冻结。静态压力四档均未建立（8G 最少仍剩 ≈3.06 GiB，`factor_m=1.0`）。见 [G2_MEM8](G2_MEM8.md)。

## 总表

| 块 | 设计名额 | 可执行（开发探针，seed=17） | 未建立 / 不能冒充可执行 |
|---|---:|---|---|
| H1 主比较 | 315 | CIFAR10/100、CORe50 NI/NC/NIC 的 S0/LR/插件栈/GSS/GEM/AGEM 已在至少一档跑过剖析；16G 还补了 MAX-A/MAX-P（NIC MAX-A 未发） | 3 seeds 正式比较未做；不能把三档 `mem=` 写成已冻结的宽/中/紧；Endless 不能填进 H1 五主基准 |
| H2 跨算法 | 108 | NC 上 GSS/GEM/AGEM 与 ER 对照已有 16G/64G 剖析 | 正式 3 容量 × 3 seeds 未做 |
| H3 Endless | 189 | 盘上有数据；A22 grouped 的 S0/LR/GSS/GEM/AGEM/插件/MAX-P 已有 16G 开发探针 | **官方 IC/IL/WC 协议未建立**；分组变体不冒充官方 |
| H4 动态压力 | 48 | CIFAR100/NC S0 与 NC O-recon、NIC S0、BR-adapt、六格静态搜索都有 seed=17 开发结果，见 [G2_H4](G2_H4.md) | 正式 48 格未做；动态 S* 只是开发选择、未冻结；静态 tight 未建立；NIC O-recon 两条是资源失败 |
| H5 插件与物理 | 180 | 开发上有 ER / GEM / EWC / 组合与 replay/batch 探针；H5c pause-keep-state 已测，bytes 不随 disable 下降 | 两档可辨识预算未冻结；release-rebuild 对 Avalanche EWC 不可实现，不并入 O-recon |
| H6 偏好与衰减 | 180 | 开发批次 30/30 完成，见 [G2_H6](G2_H6.md)：`factor_m=1.0`；NIC 上权重/衰减拉开 replay | 两档预算与 3 seeds 正式块未做 |
| H7 预取 | 120 | 供给哈希 off/on 已对齐；确定性路径下参数哈希 10/10 对齐 | **预取净收益未建立**；正式 3 seeds × 两预算未做 |
| H8 敏感性与开销 | 180 | 记录器空载/运行有校准 | 15 个预注册变体 × 两预算未做 |
| 合计 | **1320** | 只作开发证据 | 未建立项留在分母，不删覆盖 |

## 明确缺口（不要用别的结果填）

1. **压力场景**：`unestablished_at_mem64g` / `mem32g` / `mem16g` / `mem8g`。8G 静态最少仍剩 ≈3.06 GiB。H4 共运行对 NC/CIFAR100/NIC S0 可辨识（最少 ≈1.24 GiB）。NIC O-recon + 2.5 GiB 在两条序列的高压段都把 worker systemd oom-kill。不把这写成静态 tight，也不冒充 48 格正式块。不放大模型。不把盘满或 cgroup 写成板级紧内存。
2. **Endless 官方协议**（A22）：现有 grouped 只是开发数据，不是论文机器人闭环或官方划分。
3. **H7**：无确定性标志时参数更新分叉；`deterministic_algorithms` 下参数哈希已对齐。墙时差仍不能记成预取净收益。见 [G2_PREFETCH_HASH_DET](G2_PREFETCH_HASH_DET.md)。
4. **S* / Oracle**：seed=17 选择是 S* b16/r2000、Oracle b16/r10000，**未冻结**；不得改名 MAX-A。
5. **G3 身份**：远端 Git HEAD 仍是 v3 `b59b4c4`；冻结包清点见 [G3_INVENTORY](G3_INVENTORY.md)，**未冻结**。
6. **硬整机不可恢复崩溃**：按设计不主动撞机。NIC O-recon + H4 两条序列都出现过 worker systemd oom-kill（机子与执行器仍活），记为资源失败，不冒充已完成硬整机崩溃验收。
7. **H5c release-rebuild**：朴素清空不是合法 EWC 重建，见 [G2_H5C](G2_H5C.md)。

`kind=train` 在上述冻结包齐备前拒绝。
