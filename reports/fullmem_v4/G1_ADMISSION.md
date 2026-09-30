# G1 64GiB 启动机制与共享池准入（2026-09-20）

状态：**`mem=64G` 启动机制已验收；64GiB 档可作为后续开发容量。16/32GiB 未启动。硬整机 OOM 未做。不是 G3 冻结，没有正式矩阵。**

控制机：Windows `C:\Users\admin\Desktop\ssh`。Thor：`zhuzetong@10.5.229.37`（重启后 DHCP 地址未变）。WiFi MAC `50:bb:b5:50:db:5c`。

## 启动项

- 只新增 LABEL `orion-mem64g`，DEFAULT 指向它；LABEL `primary` 未改、无 `mem=`。
- 未改内核、驱动、nvpmodel（仍为 120W / mode 1）。
- 备份：`/boot/extlinux/extlinux.conf.bak.orion-fullmem-v4-20260920`
- 方案目录：`reports/fullmem_v4/boot/2026-09-20T052653Z/`
- 回退：将该目录 `extlinux.conf.rollback` 拷回 `/boot/extlinux/extlinux.conf` 后重启（DEFAULT=primary）。控制台 TIMEOUT 30 也可选手动 primary。

## 重启前后对照

| 项 | 重启前 | 重启后 |
|---|---|---|
| boot_id | `f0b2b777-3d70-42e3-9af4-bdabca3b7631` | `7e7ae8b3-a605-4653-8536-5ddedbf44c44` |
| cmdline `mem=` | 无 | `mem=64G` |
| MemTotal | 128766532 kB ≈ 122.801 GiB | 65856188 kB ≈ 62.805 GiB |
| CUDA `total_memory` | （未在本轮重启前重测；envcheck 历史约整机 RAM） | 67436736512 B = 64312 MiB ≈ 62.80 GiB |
| SwapTotal | 0 | 0 |
| nvpmodel | 120W / 1 | 120W / 1 |
| IPv4 | 10.5.229.37 | 10.5.229.37 |
| linger | 本轮已 enable | yes；用户服务随开机起来 |

请求 64G、实际 MemTotal ≈ 62.8 GiB，差额视为 firmware/carveout，不以参数字符串代替实测。身份：`reports/fullmem_v4/postboot_identity_mem64g.json`。envcheck：`reports/fullmem_v4/envcheck_mem64g.json`（`forward_backward_update=ok`）。

## Detached 执行器

- 用户单元 `orion-fullmem-v4-executor.service`，解释器 orion，WorkingDirectory 为仓库根。
- Linger=yes。正式训练不得绑 SSH。
- `g1-detach`：a(8s) 完成后自动启动 b；重复提交幂等；SSH 未保持。
- 重启时旧监督进程收到 SIGTERM，记 `supervisor_signal`。空队列重启后可继续新批次。
- systemd-run `--pipe` 已加上，便于后续日志进 executor log。

## 共享池（受控 ≤8GiB，未耗尽整机）

第一次探针受 CUDA caching 影响，mixed 的 `gpu_drop_kb=0`。第二次在 mixed 前仅作测量隔离 `empty_cache`（不是训练期 URGE 策略）。

第二次 `reports/fullmem_v4/admission/20260920T133630Z/shared_pool.json`：

- CPU 1GiB 持有/释放：MemAvailable 与 RSS 同向变化后回收。
- CUDA 1GiB fill+sync：MemAvailable 下降约 1GiB。
- pinned / 子进程 / 64MiB 页缓存写入：ok。
- mixed：GPU 先占约 1.01 GiB 可用内存，再叠加 CPU 约 1.00 GiB；释放 CPU 后回收约 0.94 GiB，随后 GPU 还能再占约 0.45 GiB。

结论：在 `mem=64G` 下，CPU 与 CUDA 竞争同一 MemAvailable/统一内存池。host cgroup 仍不是总预算（旧 20260920 反例保留）。

**未验证：** 贴近 62.8 GiB 的硬 OOM、系统卡死、记录器被杀。PLAN 禁止未安排窗口耗尽整机。

## 小规模真实数据时序

`configs/fullmem_v4/g1_cifar100_exp1.yaml`：SplitCIFAR100、experience_limit=1、S0 ER、`budget.enforcement=observed_only`（不安装 allocator 配额）。经执行器提交，`required_capacity=mem64g`。

`runs/fullmem_v4_g1_cifar100_exp1/summary.json`：status=completed，train+eval 各 1，p_diag=0.439，s_initial=1.0，run_total_s≈7.0。这只证明 mem=64G 下持续学习链路可跑，**不是 Orion 有效性结论**。

## 明确未完成

- 32GiB / 16GiB 启动项未应用；16GiB 仍只是候选。
- detach 故障注入未全做（注销后存活可用 linger 间接支持，但未 `terminate-user`；盘满、硬整机 OOM、运行中重启未注入）。
- 监督重启采纳与资源失败后续跑已用专属探针验收（2026-09-21）：`g1-adopt-a` SIGTERM 后 `finalize_dead` completed exit 0，`g1-adopt-b` 在 `resume` 后才跑；`g1-resource-fail` status=`resource_failure`、health=`ok`，随后 `g1-resource-after` completed。paused 路径会收尾 recorded worker，但保持 paused，不自动取 inbox。
- G2 对齐/数据全集/物理 buffer 审计/开发选档：进展见 [G2_CALIBRATION](G2_CALIBRATION.md)。
- 禁止把本次 64GiB 写成三档预算或正式主比较。
