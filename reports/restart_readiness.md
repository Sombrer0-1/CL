# Thor 第三阶段准备验收

2026-09-17。结论：**可进入第三阶段G1/G2开发工作；尚不可直接进入G3/G4正式比较。** 机器可读记录：[admission.json](effectiveness_v3/readiness/admission.json)。

## 本轮已执行

| 检查 | 结果 |
|---|---|
| 专用orion、Thor/sm_110、GPU前向/反向/参数更新 | 通过；driver595.78、torch2.14.0+cu132、Avalanche0.6.0 |
| 回归 | 165 passed，15库告警；[原始输出](effectiveness_v3/readiness/tests.txt) |
| allocator64MiB真实OOM与失败阶段峰值 | 通过；[GPU证据](effectiveness_v3/validation_gpu.json) |
| CIFAR-100 / CORe50 mini获取及划分恢复 | 官方MD5通过；开发划分核对、CORe50引用图片存在性检查通过；[数据证据](effectiveness_v3/readiness/data.json) |
| CORe50-NC真实首experience | 训练1次、评价1次，completed；端到端35.53秒 |
| CIFAR100真实首experience | 训练1次、评价1次，completed；端到端13.73秒 |
| 当前源码与两次真实运行身份 | 一致；运行ID及hash见admission.json |
| 首波次生成 / 未审查冻结拒绝 | 3个NC评价batch探针已生成；缺g2_review时明确拒绝 |
| git diff --check | 通过 |

真实数据验收采用独立`thor_readiness` revision、真实数据首experience、静态ER、开发验证反馈、1024MiB allocator配额。它验证迁移后的运行链路，**不是完整流校准、不是Orion有效性实验**。完整配置在`configs/effectiveness_v3/thor_readiness/dev/`，原始运行在`runs/`；关键summary副本在readiness目录。

本轮修复及影响见[A28](../docs/decisions/A28.md)。PLAN补充了论文描述与重建的距离、三层判定及实用效应阈值；SDD清楚区分已有实现和G3前剩余工作。未改动历史数据manifest、light24结果或旧开发模板；接手前已有pressure_v2收尾说明改动保留，inspect对此仍返回非零。

## 必须保留的边界

- `pip check`仍有cuSPARSELt SBSA标签告警，未声称全依赖检查通过；dense FP32实测通过。
- host专属cgroup未配置/不可用，H组尚未建立。allocator不等于统一内存板级硬上限。
- SDD §10的原始证据闭包、完整候选/场景验证和完整配对报告必须在正式冻结前完成；不是通过测试就自动具备。
- 153为计划分母，不是现成可执行矩阵。预算/L_cal/S*未冻结，没有本机正式结论。
- 当前保留原有及本轮工作树修改，未混合提交。raw/processed/runs需单独备份。

下一步按[HANDOFF](../docs/HANDOFF.md)使用`thor_r1`逐波推进；完整流耗时在G2实测估计，不能把上述首experience计时线性当成含插件的153格预算。5090旧准备报告见[归档](../docs/environments/linux_rtx5090/restart_readiness.md)。
