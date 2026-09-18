# effectiveness_v3 设计前代码审查与补丁

日期：2026-09-16。审查基线：`ecc004c`。范围：训练/评价/控制时序、失败阶段、预取计时、校准源域、环境验收与checkpoint。这是有边界的审查，不代表仓库不存在其他缺陷，也**不是**方法有效性结论。

**换机说明（A26/A27）：** 本文记录源码修补。其中 5090 上的 pytest 计数、envcheck 和 GPU OOM 诊断已归档到 [linux_rtx5090](../../docs/environments/linux_rtx5090/)，不能作为 Thor 冻结或正式结果。当前环境见仓库根 `reports/envcheck.json` 与 [A27](../../docs/decisions/A27.md)。G2–G4 须在本机重新校准。

## 已确认并修复

| ID | 缺陷与影响 | 修补 | 验证 |
|---|---|---|---|
| B01 | OOM处理取recorder.last，开放的失败phase尚未end，可能把训练/过渡峰值当成评价/训练失败峰值 | PhaseRecorder.end_failed按phase/index闭合当前phase，禁止回退到不同阶段；测量异常保留原OOM且记error；拒绝嵌套phase重置峰值 | 失败阶段回归、重叠phase拒绝；真实cuda:0 allocator OOM检查 |
| B02 | 无batch_sampler回退路径在next之后才计时；queued回退完全未记录producer耗时 | serial/queued均围绕next(inner_iter)计时，保留次序及消费次数 | 虚拟时钟只在取数前进，两路径均测得6s而非0；现有顺序/早停测试 |
| B03 | 环境检查只判断梯度存在；opt.step未更新参数也可能通过，Avalanche导入失败仍打印PASS | 检查loss/梯度/参数有限且实际参数变化；必需Avalanche/策略导入失败直接非零退出 | noop优化器拒绝、真实更新通过、模拟缺库拒绝；GPU envcheck通过 |
| B04 | 开发校准允许feedback_source缺失，且部分资源/IO探针未检查来源 | 所有probe都要求明确development_val_seen；两种来源字段若冲突/空缺也拒绝 | 缺失/None/别名冲突回归；既有非开发反馈拒绝与完整校准测试通过 |
| B05 | 预取校准把producer工作与consumer等待相加，两者可重叠，会夸大阻塞比例 | serial用produce，queued用wait，producer单列；不重算旧冻结材料 | 10s学习、6s producer、2s wait：off=0.6、on=0.2，而非0.8 |
| B06 | 控制器末轮记录后continue，跳过experience_boundary checkpoint，latest落后一轮或不存在 | 末轮不重配，但启用checkpoint时仍保存当前experience和latest | 真实Avalanche CPU小型单experience跑完整runner，两个checkpoint均completed_experiences=1 |

这些修补涉及6个源码文件及新增 `tests/test_restart_regressions.py`。B06功能检查使用小型张量数据，只验证保存时序，不是持续学习正式结果。B01 GPU检查是真实分配和OOM，但没有基准训练，不代表资源自适应有效性。

## 实际验证

- 全套（**5090 当时**）：**134 passed，10 warnings，5.96s**。原始输出已归档：[validation_tests.txt](../../docs/environments/linux_rtx5090/validation_tests.txt)。新增13项回归；库告警来自未配置logger及Avalanche replay update弃用。
- 当时 `orion_repro.envcheck`：Python3.11.16、torch2.11.0+cu128、torchvision0.26.0+cu128、Avalanche0.6.0。这不是本机 Thor 证据。
- 当时 CUDA 失败阶段检查：进程allocator配额64MiB，前阶段峰值2MiB，评价阶段保留6MiB后请求128MiB触发OOM，失败记录峰值为6MiB而非上一阶段2MiB。见归档 [validation_gpu.json](../../docs/environments/linux_rtx5090/validation_gpu.json)；复核脚本仍为 [verify_failure_peak.py](verify_failure_peak.py)，在 Thor 上重跑会覆盖本机新产物。

```bash
ORION_PY=/home/zhuzetong/miniconda3/envs/orion/bin/python
WANDB_MODE=disabled "$ORION_PY" -m pytest tests -ra
WANDB_MODE=disabled "$ORION_PY" -m orion_repro.envcheck
WANDB_MODE=disabled "$ORION_PY" reports/effectiveness_v3/verify_failure_peak.py
```

## 对历史证据的影响

- B01限定旧失败摘要的峰值可信度：failure_phase和原始堆栈仍保留，旧summary峰值不再直接用于新预算校准。未重写旧摘要。
- B02影响无可规划sampler回退路径，不能一概声称历史计划索引路径的供给时间全部错误。
- B04是验证漏洞；现有pressure_v2 probe来源标注仍保留，未发现就不能推断历史确实用了正式测试选参。
- B05的串行off场景wait通常为0；不得仅因修补就宣称旧IO门槛必然失效。异步on的旧和式不作为新阻塞指标。
- B06不影响checkpoint_policy=none的历史主比较；不重新计算旧P/S。
- 新源码身份已变化。两个历史阶段的配置、协议、run映射与数字保持原样；不重开96格或续补54格。

## 已评审但尚待实现/验证的能力

1. 新StageContext/执行器/冻结/报告全链路；pressure_*继续绑定旧阶段，不能用于新正式运行。
2. F11的fixed_advanced尚无schema/runtime支持；host专属cgroup执行器尚未实现。
3. PhaseRecord的host值目前是结束快照；异步采样聚合与phase隔离、EWC完整辅助访问统计需按SDD补齐。
4. 预取线程阻塞退出、重复迭代、experience+version双校验等边界需系统验收；现有基础顺序测试通过不等于全部通过。
5. 旧报告器只按部分字段分组，不能用它跨新revision配对；新报告必须矩阵驱动、核对完整控制签名。
6. 新平台完整真实数据校准、八场景覆盖、正式有效性比较均尚未执行。

## PLAN→SDD自审

已逐项核对：S01–S08来源及可观测证据、三层验收、两数据集完整流/3seeds、因素消融、敏感性预定值、开发反馈隔离、失败/突变归属、count-space/插件驻留局限、host/device分离、预取交互、身份/路径闭包、153名额及条件分母。未发现需要先改研究目标才能实施的设计冲突。

交叉检查产物见 [design_review.json](design_review.json)：153名额计数、12项验收状态、修改文档链接和对齐CSV检查通过；474个历史受保护文件逐字节与Git基线一致，未提前生成formal配置。该文件还保存本次源码/测试文件哈希，不能当作正式冻结协议。

剩余不确定性属于待测问题：本机能否建立可辨识训练压力/自然IO、URGE是否实际扩张、host权限及插件成本。全部有明确诊断和未建立出口；不保证正向结论，不把设计自审称为实验验收。

## 2026-09-17 Thor 修订

前文“尚待实现”是5090设计审查时点，不能作为当前源码清单。fixed_advanced、StageContext/CLI、HostEnvelope和host phase采样已有实现及本机回归；原始证据校验、完整开发候选和最终配对报告仍未完成，见[SDD §10](../../docs/SDD_effectiveness_v3.md#10-thor-准入实施清单2026-09-17)。

本轮新增科学对照/迁移问题与修补见[A28](../../docs/decisions/A28.md)，新增`tests/test_v3_migration.py`。当前Thor证据存于[readiness](readiness/)，GPU失败峰值诊断为本机新生成的[validation_gpu.json](validation_gpu.json)。旧5090证据仍在宿主归档目录。本轮未改写历史实验数字。
