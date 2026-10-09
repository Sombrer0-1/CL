# G2 `mem=5G` 准入与静态压力

日期：2026-09-30。`kind=probe` 批次 `g2-dev-mem5g` 9/9 completed。**不是三档冻结。** 4G 没有切：8G 上 S0 板级峰值约 3.97 GiB，4G 的 MemTotal 会落在这条峰值下面，S0 本身很可能完不成，不能记成 tight。

## 启动

- 新增 LABEL `orion-mem5g`，DEFAULT 指向它。保留 `primary`、64G、32G、16G、8G。
- 8G 回退：`/boot/extlinux/extlinux.conf.bak.orion-fullmem-v4-20260930-mem8g`
- 未改 nvpmodel、内核、驱动。
- boot_id=`984c4ed8-ed8f-4782-bc05-2577fd938c46`
- `mem=5G`，MemTotal=5093776 kB ≈ 4.858 GiB，swap=0，120W/mode 1
- 共享池小探针（最大 2 GiB、块 128 MiB）：CPU/CUDA/pinned/页缓存/子进程/混合均 ok。GPU 占用后 CPU 再降，CPU 释放后余量回来。

## 预注册剖析

| 运行 | 状态 | 最少 MemAvailable |
|---|---|---:|
| CIFAR100 S0 第一次 | completed | 0.945 GiB |
| CIFAR100 S0 第二次 | completed | 0.946 GiB |
| CIFAR100 replay2000 | completed | 0.941 GiB |
| CIFAR100 ER+GEM+EWC | completed | 0.714 GiB |
| CIFAR100 batch256 | completed | **0.181 GiB（194809856 B）** |
| CORe50-NC S0 | completed | 1.067 GiB |

8G 上同一条 batch256 最少仍剩 3.06 GiB。5G 上它剩 0.181 GiB，低于本项目 H4 校准时使用的 1 GiB 安全地板，但没有 oom-kill。S0 两次都完整完成。

按设计，这是 tight 候选：最小的已准入容量上，S0 训练/评估两次完成，且预注册高资源静态出现可测压力。它不是资源失败。`km=0.25` 时大约 32 MiB 余量就让 `factor_m` 饱和，0.181 GiB 仍高于这个膝点，所以内存因子预计仍是 1。这是主方法公式的结果，不改成归一化。

loose 候选仍是 8G。tight 与 loose 的几何中点附近没有另一档已准入容量，mid 未建立，不用 5G 再抄一档。

## 接着跑的正式比较

`g4-static-tight-mem5g`：CIFAR-100 与 CORe50-NC，S0 / 开发静态 S*（b16/r2000，没有在 5G 上重搜）/ O-recon，seed 0/1/2，无 H4 持有。`kind=train`。不是 1320 格。NIC、Endless、H5–H8 还没放进这条队列。

第一条 `g4-cifar100-s0-static-seed0-mem5g` 因 formal 阶段不能用 `development_val_seen` 失败，保留为 `implementation_error`。未启动的 17 条从 inbox 撤到 `queue/not_started_withdrawn_feedback/`，没有执行。补跑 b2 把反馈改成与已完成 H4 正式格相同的 `official_test_seen`，18 条已重新入队。执行器正在跑，不挂在 SSH 上。
