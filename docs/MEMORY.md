# 阶段4长程记忆

更新：2026-10-11 23:30（UTC+8）。阶段 4 已结项。指针：[STATUS](../STATUS.md)、[G5_CLAIMS](../reports/fullmem_v4/G5_CLAIMS.md)。

## 结论

- Orion（O-recon）没有可信的实用优势，不借鉴。五个数据流、5G 与 8G、静态与共运行压力、NC 上三种算法，O-recon 都比调好的静态 S* 或同算法基础静态差：ΔP −0.17 到 −0.39，时间 1.3 到 4.5 倍。NIC 的压力场景里 O-recon 是唯一被 oom-kill 的方法。
- 归因：内存因子饱和（`factor_m=1.0`，`km=0.25` 时约 32 MiB 余量才下降），batch/replay 几乎不动；urge 恒高于阈值使 GEM+EWC 常开，EWC（λ=100）拖垮可塑性。只调 batch/replay 的 BR-adapt 与 S0 无差别。
- 这是对本重建实现与本场景的负结果，不是对作者平台的反驳。中间档、Endless 官方协议、Oracle 全搜索、LR/MAX-A/MAX-P、H1–H3、H5–H8 是覆盖缺口，不写成无效。

## 环境

- 2026-10-11 23:10 已还原为原始 `primary` 启动项，boot_id `68b3d186-454f-44a1-adaa-eaee2be1f369`，`MemTotal` ≈ 122.8 GiB，无 `mem=`。实验配置的备份是 `/boot/extlinux/extlinux.conf.bak.orion-fullmem-v4-20261011-pre-restore`。
- 执行器已 stop 并 disable。Linger 保持原状 yes。没有改 nvpmodel、内核、驱动。
- 控制机连接：这台控制机的 `id_ed25519` 已在 2026-10-09 追加进 `authorized_keys`。密钥和密码脚本并存。

## 已知坑

- **H5b 无效。** 配置 `plugin_policy: fixed_default` 令 `TogglePlugin` 全程关闭（`loop.py` 起跑时 `enabled = (initial_mode == "advanced")`）。要固定打开插件须用 `fixed_advanced`。判断插件是否真的在跑，看 `auxiliary_visits`、`plugin_state_bytes`、`optimizer_mode`，不能只看 `plugins.json` 里的 `enabled`。
- `executor.py` 把 `cuda error: out of memory` 也记成 `resource_failure`，热身 CUDA OOM 约 10 秒，换新 ID 补跑即可。
- 失败记录保留，不改写已失败的 done 记录，不用同一 task_id 重跑。
- 不把 5G/8G 写成已冻结三档，不把开发选择 S* 写成静态冻结。
