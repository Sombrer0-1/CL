# G3 预算身份草稿（非正式，未冻结）

日期：2026-09-25。对照设计契约 §2 / §4.2 与 PLAN §4A。**这不是 G3 冻结，不授权 `kind=train`，不把 8G 写成 tight。**

设计允许：建不成三个可辨识静态档时，保留实际档数与缺口，不拿同容量充三档。本机 ResNet-20 32×32 在已准入的 64/32/16/8GiB 上静态均宽松。共运行压力按 H4 专属后台进程建立，不放大模型、不切更小 `mem=`、不把 `M_max` 改小冒充实占用。

## 若进入冻结包，建议写下的身份

| 项 | 草稿值 | 不是什么 |
|---|---|---|
| 板级启动档 | `mem=8G`（MemTotal ≈ 7.799 GiB，boot 准入已测） | 不是紧档，不是三档里的「紧」 |
| 静态宽/中/紧 | 四档均 `unestablished_at_{capacity}` | 不能把 64/32/16 写成已冻结宽/中 |
| 动态场景 | H4 专属进程，`high_bytes=2684354560`（2.5 GiB），`low_bytes=0`，序列 `low_high_low` / `high_low_high`，切换 `floor(N/3)`、`floor(2N/3)` | 不是静态 tight；空载校准曾到 4 GiB，训练用 2.5 GiB 以免硬耗尽 |
| 内存因子 | 原生 URGE `km=0.25`、MiB、以 live MemTotal 为 `M_max` | 饱和（余量 ≫ 32 MiB）是主方法结果；归一化只能叫 `O-eng-*` |
| S* / Oracle | 仍是 seed=17 开发选择：S* b16/r2000，Oracle b16/r10000 | 未冻；动态 S* 必须在该实际序列上单独静态搜索，不能把开发 S* 改名 |
| H4 方法 | 设计：S0 / 动态开发 S* / O-recon / 仅 batch-replay 自适应 | seed=17 开发格已齐；动态 S* 按序列记下但未冻结；NIC O-recon 是资源失败 |

## 仍挡住冻结

源码 revision、orion 环境锁、S*/Oracle 正式选择、Endless 官方协议、正式 3 seeds。未齐之前执行器继续拒绝 `kind=train`。
