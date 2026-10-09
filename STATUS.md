# 当前状态：H2 的基础静态、Orion、固定 EWC 都已终态，执行器空闲

更新：2026-10-09 21:20（UTC+8）。`mem=5G`，boot_id=`13f1512f-57ca-4ce5-8054-f415f168afd3`，MemTotal ≈ 4.856 GiB。5G 是 tight 候选，8G 是 loose 候选，mid 未建立。

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

## 现场

执行器 active，mode=idle，inbox 与 running 为空。心跳 21:21。`mem=5G` 不变。H2 还缺调优静态，也还缺另外两档容量。没有下一条已提交的任务。

[G5_CLAIMS](reports/fullmem_v4/G5_CLAIMS.md) 仍只覆盖已准入的 H4。后来的静态比较和这组 H2 基础静态写在各自报告和本文件里，还没有并进那份裁决。

`executor.py` 把 `cuda error: out of memory` 也记成 `resource_failure`。上面两次热身失败按这一行归类。

查询：`ssh zhuzetong@10.5.229.37` 后看 `experiments/fullmem_v4/state/executor.json`。
