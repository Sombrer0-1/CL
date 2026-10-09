# 当前状态：H2 Orion 已终态，正在跑固定 EWC

更新：2026-10-09 18:20（UTC+8）。`mem=5G`，boot_id=`13f1512f-57ca-4ce5-8054-f415f168afd3`，MemTotal ≈ 4.856 GiB。5G 是 tight 候选，8G 是 loose 候选，mid 未建立。

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

## 已完成

上午那 11 条在 14:32 全部 completed。AGEM 基础静态 seed 2 的 b2：P 0.629，S 0.388，online 514 秒。NI O-recon seed 2 的 b3：P 0.053，S 0.962，online 1028 秒，相对 S* 仍然更差更慢。

`g4-h2-orecon-nc-mem5g` 九条都 completed。相对同算法基础静态，GEM 的 ΔP 约 −0.32，AGEM 约 −0.39，在线时间约 1.5 到 1.7 倍。GSS 未崩的两个 seed ΔP 约 −0.26，seed 2 两边 P 都是 0.011。稳定性更高，但没有实用优势。数字见 [G4_H2_NC](reports/fullmem_v4/G4_H2_NC.md)。

## 正在跑

`g4-h2-fixedopt-nc-mem5g`：NC 上 GSS、GEM、AGEM，EWC 固定打开，λ=100，控制器关闭，batch 16，replay 200。9 条。这是 H2 的固定可选策略，不是调优静态，也不是三档。

GSS seed 0 在 18:18 热身创建 CUDA context 时 OOM，约 12 秒，`resource_failure`。执行器已继续，当前是 GSS seed 1。其余格子若正常训练，这一批大约还要 2 到 3 小时。

## 现场

执行器从 2026-10-07 12:35 起一直 active。`mem=5G` 不变。调优静态和另外两档容量还没做。

[G5_CLAIMS](reports/fullmem_v4/G5_CLAIMS.md) 仍只覆盖已准入的 H4。后来的静态比较和这组 H2 基础静态写在各自报告和本文件里，还没有并进那份裁决。

`executor.py` 把 `cuda error: out of memory` 也记成 `resource_failure`。上面两次热身失败按这一行归类。

查询：`ssh zhuzetong@10.5.229.37` 后看 `experiments/fullmem_v4/state/executor.json`。
