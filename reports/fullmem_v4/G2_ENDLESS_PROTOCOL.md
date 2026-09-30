# Endless 协议缺口（非正式）

日期：2026-09-24。对应 PLAN Table 2 与对齐表 `endless_protocol`。

盘上有 Zenodo 4899267 的 Endless-Sim 公开 zip，并已解压。现用划分是 **A22 grouped holdout**（`split_policy: class_ordered_holdout_embargo_v2`），经验数 IC/IL/WC = 4/5/5，与论文表口径一致，**但不是官方机器人协议，也不是实体闭环**。

未建立的是：

- 官方 train/test 划分与评测脚本（若作者另有）
- 机器人 onboard 时序、传感器与闭环控制
- 把 A22 高 P 写成论文 §5.3 场景成功

H3 189 格因此全部留在未建立分母。16G 上 A22 的 S0/LR/GSS/GEM/AGEM/插件/MAX-P 只是开发数据探针。
