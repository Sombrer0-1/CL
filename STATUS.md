# 当前状态：Thor 第三阶段准备准入

更新：2026-09-17。执行设备为 Jetson AGX Thor，独立环境 `/home/zhuzetong/miniconda3/envs/orion`。接手时已有的迁移修改已保留；本轮没有提交或覆盖历史实验。决定见[A27](docs/decisions/A27.md)、[A28](docs/decisions/A28.md)。

## 已完成

- 重新梳理训练、评价、控制、资源、预取、开发、冻结与报告链路，更新[PLAN](PLAN.md)、[SDD](docs/SDD_effectiveness_v3.md)、[仓库地图](docs/REPO_MAP.md)和[交接命令](docs/HANDOFF.md)。
- 修复开发O00/O10阈值覆盖、L中位数、DYN静态搜索未施加预留、缺失候选误入校准、正式运行缺冻结对象、源码哈希包含自身产物等问题；新增迁移回归。
- 本机全套测试 **165 passed**；GPU参数更新和真实allocator OOM失败峰值诊断通过。证据在[readiness](reports/effectiveness_v3/readiness/)。
- CIFAR-100和CORe50 mini已下载并校验；NC9开发划分已恢复，历史manifest未覆盖；数据准入检查通过。
- `thor_r1`首个开发波次已生成（3个NC评价batch诊断），尚未运行完整开发校准。`thor_readiness`单独存放真实数据首experience验收，不进入校准或正式均值。
- 两套真实数据首experience训练/评价均completed（35.53s / 13.73s），源码身份一致；见[准备验收](reports/restart_readiness.md)。
- G2审查缺失时freeze拒绝执行；没有正式冻结预算/L_cal/S*，没有正式矩阵。

## 准入边界与下一步

可以进入第三阶段 **G1真实数据验收 / G2开发工作**。正式G3/G4尚未准入；先完成[SDD §10](docs/SDD_effectiveness_v3.md#10-thor-准入实施清单2026-09-17)中的原始证据闭包、完整开发候选与场景判定、配对统计报告，再审查冻结。不能把当前报告框架当成完整有效性分析器。

1. 读取本机准备验收报告，使用 `thor_r1`逐波开展完整真实开发流。
2. 保存失败和未建立场景；统一内存上的allocator限额不代表板级共享内存限制。
3. 完成完整开发与SDD实现清单，生成有证据的`g2_review.json`后freeze→emit→run→report。
4. A–H共153设计名额、主要比较3 seeds；按PLAN §9逐项裁决C01–C08，不预设Orion有效。

## 已知限制

- `pip check`报告cuSPARSELt 0.8.1的SBSA wheel标签不匹配；当前dense FP32训练路径已验证，未篡改元数据或更换驱动。
- host专属cgroup未配置且当前探测不可用，H组尚不能执行；不以RLIMIT或allocator冒充共享RAM硬限额。
- `inspect`仍报告接手前已有的`reports/pressure_v2/CLOSEOUT.md`迁移说明改动；不删除保护检查换取绿灯。
- raw/processed/runs不纳入Git；迁移需另备份。当前工作树含原有迁移修改和本轮修改，未做混合提交。

## 历史边界

| 阶段 | 状态 |
|---|---|
| light24_v1 / RTX5060 Ti | 96格，90完成+6评价OOM；未证明Orion有效性，原证据保留 |
| pressure_v2 / RTX5060 Ti | 中断收尾、未验收，不跨机续跑 |
| effectiveness_v3 / RTX5090 | A26放弃校准与跑数，不混入Thor |
| effectiveness_v3 / Thor | 准备与开发入口已具备；完整开发/正式结果尚无结论 |
