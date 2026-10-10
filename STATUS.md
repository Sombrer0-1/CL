# 当前状态：调优静态已选出，正在跑 GEM 和 AGEM 的正式 seed

更新：2026-10-10 11:00（UTC+8）。`mem=5G`，boot_id=`13f1512f-57ca-4ce5-8054-f415f168afd3`，MemTotal ≈ 4.856 GiB。5G 是 tight 候选，8G 是 loose 候选，mid 未建立。

## 已完成

8G 上 CIFAR-10、CIFAR-100、NI、NC、NIC 的 S0 / S* / O-recon 都完成。5G 上 CIFAR-100、NC、NIC、CIFAR-10 也完成。各流 O-recon 相对 S* 都没有实用优势。见 [G4_STATIC_LOOSE](reports/fullmem_v4/G4_STATIC_LOOSE.md)、[G4_STATIC_TIGHT](reports/fullmem_v4/G4_STATIC_TIGHT.md)。

5G NI 的 O-recon 三条现在都有质量结果。seed 2 的前两次热身 CUDA OOM 保留；b3 已 completed，三条相对 S* 都更差更慢。数字在下面和 [G4_STATIC_TIGHT](reports/fullmem_v4/G4_STATIC_TIGHT.md)。

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

`g4-h2-fixedopt-nc-mem5g` 于 20:54 终态。8 条 completed，GSS seed 0 热身 CUDA OOM。固定 EWC 相对基础静态更差更慢：GEM 的 ΔP 约 −0.44，AGEM 约 −0.45，GSS seed 1 的 ΔP −0.374，在线时间大约 1.5 到 2.2 倍。见 [G4_H2_NC](reports/fullmem_v4/G4_H2_NC.md)。

## 进度

设计分母是 1320 个名额，不是执行队列。已经有终态的正式格大约 165 个，约占 12%：H4 的 48 格，五流 S0/S*/O-recon 在 5G 和 8G 上的 90 格，以及 mem5g 上 H2 三个对照的 27 格。已写入 `FREEZE.json` 的批次本身都已终态。

中间档、Endless 官方协议、Oracle 全搜索、LR/MAX-A/MAX-P 和 H5–H8 的正式矩阵还没建立。这些留在分母里，没有排进队列，也不能给出跑完日期。按单卡串行、每条大约 10 到 20 分钟估算，把这些未建立块也全部排上是数周，而且中间档和 Endless 官方协议现在不能靠加运行填上。

mem5g 上 H2 的调优静态已经选出。开发选择 18 条在 2026-10-09 22:29 终态。GSS 为 b16/r200，和基础静态相同，正式结果复用。GEM 为 b256/r200。AGEM 为 b16/r2000，`patterns_per_exp` 500。见 [G4_H2_NC](reports/fullmem_v4/G4_H2_NC.md)。

## 正在跑

`g4-h2-tuned-nc-mem5g`：GEM 与 AGEM 各 seed 0/1/2，共 6 条。当前是 GEM seed 0。开发选择里 GEM b256 大约 3 分钟，AGEM b16/r2000 大约 6 分钟；正式评价会更长，这一批大约 1 到 2 小时。这批结束前不换档。

## 现场

执行器 active。`mem=5G` 不变。

[G5_CLAIMS](reports/fullmem_v4/G5_CLAIMS.md) 仍只覆盖已准入的 H4。后来的静态比较和这组 H2 基础静态写在各自报告和本文件里，还没有并进那份裁决。

`executor.py` 把 `cuda error: out of memory` 也记成 `resource_failure`。上面两次热身失败按这一行归类。

查询：`ssh zhuzetong@10.5.229.37` 后看 `experiments/fullmem_v4/state/executor.json`。
