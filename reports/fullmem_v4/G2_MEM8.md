# G2 `mem=8G` 开发训练探针（未冻结）

批次：`g2-dev-mem8g`。9/9 `completed`，failed 目录无 mem8g。`kind=probe`。seed=17。**不是正式矩阵，不是 G3 冻结。**

MemTotal ≈ 7.799 GiB。空载三窗 MemAvailable 最低 ≈ 5.17 GiB。本档 occupancy 最少仍剩 **3286081536 B ≈ 3.06 GiB**（CIFAR100 batch256）。峰值 MemTotal−MemAvailable ≈ **4.74 GiB**。`factor_m` 全部 1.0。`pressure_scene: unestablished_at_mem8g`。

CIFAR100 S0 两次均完整完成。预注册高资源静态（batch256 / replay2000 / ER+GEM+EWC）均完成、无资源失败。按设计：最小已准入档对 ResNet-20 32×32 仍宽松，**不把 8G 写成 tight，不放大模型**。H4 才是共运行压力。

## 队列结果

| seq | task_id | status | P | S | 墙时 s | min avail GiB | peak used GiB |
|---:|---|---|---:|---:|---:|---:|---:|
| 1000 | `g2-idle-mem8g` | completed | — | — | 3×60 | 空载最低 ≈5.17 | — |
| 1001 | `g1-cifar100-exp1-mem8g` | completed | 0.405 | 1.000 | 7.4 | （未进 G2 occupancy 范围） | — |
| 1002 | `g2-dev-cifar100-s0-seed17-mem8g` | completed | 0.441 | 0.582 | 70.9 | 3.83 | 3.97 |
| 1003 | `g2-dev-cifar100-s0-repeat-seed17-mem8g` | completed | 0.444 | 0.580 | 71.5 | 3.84 | 3.96 |
| 1004 | `g2-dev-cifar100-batch256-seed17-mem8g` | completed | 0.0905 | 0.914 | 25.5 | **3.06** | **4.74** |
| 1005 | `g2-dev-cifar100-replay2000-seed17-mem8g` | completed | 0.397 | 0.696 | 63.1 | 3.84 | 3.96 |
| 1006 | `g2-dev-cifar100-er-gem-ewc-seed17-mem8g` | completed | 0.164 | 0.829 | 312.5 | 3.64 | 4.16 |
| 1007 | `g2-dev-core50-nc-s0-seed17-mem8g` | completed | 0.673 | 0.428 | 219.6 | 4.00 | 3.80 |
| 1008 | `g2-occupancy-mem8g` | completed | — | — | — | 本档汇总 | — |

原文：`reports/fullmem_v4/g2/occupancy_mem8g.json`。`pressure_scope: run_id_contains_capacity`，6 个本档 G2 成员。

## 选档含义（未冻结）

- S0 两次完整：满足 tight 的可行性一半。
- 高资源静态无资源失败，最少仍剩 ≈3.06 GiB，`factor_m=1.0`：可测压力未建立。
- 因此 8G **不能**记成该小模型的 tight。64/32/16/8 四档静态压力均未建立。
- 下一步按设计是 H4 专属后台占用，不是再切更小 `mem=`，也不是放大骨干。
