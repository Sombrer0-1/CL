# 当前状态：正在补两条热身失败，并跑 NC 上的 H2 Orion

更新：2026-10-09 11:40（UTC+8）。`mem=5G`，boot_id=`13f1512f-57ca-4ce5-8054-f415f168afd3`，MemTotal ≈ 4.856 GiB。5G 是 tight 候选，8G 是 loose 候选，mid 未建立。

## 已完成

8G 上 CIFAR-10、CIFAR-100、NI、NC、NIC 的 S0 / S* / O-recon 都完成。5G 上 CIFAR-100、NC、NIC、CIFAR-10 也完成。各流 O-recon 相对 S* 都没有实用优势。见 [G4_STATIC_LOOSE](reports/fullmem_v4/G4_STATIC_LOOSE.md)、[G4_STATIC_TIGHT](reports/fullmem_v4/G4_STATIC_TIGHT.md)。

5G NI 的 O-recon 只有 seed 0/1 有质量结果，ΔP 约 −0.17，ΔS 约 −0.13，online 约 2.6 倍。seed 2 的原 attempt 和 b2 都在热身创建 CUDA context 时 OOM（`cuDevicePrimaryCtxRetain`），服务运行约 11 秒，队列记 `resource_failure`。没有第三条质量结果。

H2 基础静态 `g4-h2-basic-nc-mem5g` 于 2026-10-08 13:36（UTC+8）终态。范围只是 NC 上 GSS、GEM、AGEM 的基础静态，seed 0/1/2。调优静态、算法上的 Orion、固定可选策略和三档都不在这批里。

P = `p_diag`，S = `s_initial`。失败格不填 0。

| 方法 | seed | 状态 | P | S | online 秒 |
|---|---:|---|---:|---:|---:|
| GSS | 0 | completed | 0.435 | 0.664 | 792 |
| GSS | 1 | completed | 0.450 | 0.622 | 791 |
| GSS | 2 | completed | 0.011 | 1.000 | 523 |
| GEM | 0 | completed | 0.596 | 0.462 | 680 |
| GEM | 1 | completed | 0.588 | 0.482 | 681 |
| GEM | 2 | completed | 0.600 | 0.468 | 688 |
| AGEM | 0 | completed | 0.637 | 0.383 | 514 |
| AGEM | 1 | completed | 0.646 | 0.370 | 504 |
| AGEM | 2 | resource_failure | — | — | 热身约 9 秒 |

GSS seed 2 的九个经验都训练并评估完了。P 约 0.011，S 停在 1，和 seed 0/1 不是同一量级。AGEM seed 2 与 seed 0/1 的 batch/replay 相同，失败发生在 CUDA context 创建。

## 正在跑

2026-10-09 11:40（UTC+8）提交 11 条，执行器已从 idle 进入 running。顺序按 seq：

1. `g4-core50-nc-agem-basic-seed2-mem5g-b2`。补跑 H2 基础静态里热身 CUDA OOM 的 AGEM seed 2。原 attempt 保留。
2. `g4-core50-ni-orecon-static-seed2-mem5g-b3`。补跑 5G NI O-recon seed 2。前两次 attempt 保留。
3. `g4-h2-orecon-nc-mem5g`：NC 上 GSS、GEM、AGEM 各 seed 0/1/2，共 9 条。控制器常数抄自已完成的 NC O-recon 静态格。可选插件只有 EWC，因为 GSS/GEM/AGEM 不能再叠 GEM。预取保持关闭，与基础静态相同。这一批不是调优静态，不是固定可选策略，也不是三档。

基础静态成功时 AGEM 大约 9 分钟，GSS 大约 13 分钟，GEM 大约 11 分钟。NC 上 ER 的 O-recon 在线大约 16 分钟。按这个量级，11 条大约 3 到 5 小时。热身再 OOM 的补跑会在十几秒内结束，后面的格子继续。

## 现场

执行器从 2026-10-07 12:35 起一直 active。`FREEZE.json` 已准入 `g4-h2-orecon-nc-mem5g`。此前四批 `g4-h4-mem8g`、`g4-static-tight-mem5g`、`g4-static-loose-mem8g`、`g4-h2-basic-nc-mem5g` 的原格子都有终态。

[G5_CLAIMS](reports/fullmem_v4/G5_CLAIMS.md) 仍只覆盖已准入的 H4。后来的静态比较和这组 H2 基础静态写在各自报告和本文件里，还没有并进那份裁决。

`executor.py` 把 `cuda error: out of memory` 也记成 `resource_failure`。上面两次热身失败按这一行归类。

查询：`ssh zhuzetong@10.5.229.37` 后看 `experiments/fullmem_v4/state/executor.json`。
