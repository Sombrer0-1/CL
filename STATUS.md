# 当前工作状态

更新：2026-09-14。有效范围为 **light24_v1 代表性方法复现**，见 PLAN.md。旧全覆盖目标已被用户授权替换，不再恢复旧MAX-A队列。

## 当前阶段

**仓库已完成轻量化重构及恢复前验证，可交接运行。当前没有项目训练进程，新正式矩阵尚未启动。**

- 90个配置和分阶段矩阵已生成；有限开发搜索完成后另生成6个selected配置，共96格。
- 主场景：CIFAR100、CORe50-NC；核心对照、显存预算、偏好、固定配置预取、阈值敏感性；Endless三场景作为开发划分辅助验证。
- 历史71次正式运行保留为背景；CIFAR100 MAX-A seed2的两个中断产物保留，已移出当前范围，不是待补齐任务。
- 约24小时是工作量估算，**没有时间截止或超时杀训练规则**。矩阵训练预计4–15小时，验证分析另2–5小时，需根据实测更新。

## 恢复前验证

100项测试通过（5条Avalanche无logger警告）；90个配置及全部活动矩阵校验通过；两次真实GPU小型训练完成，验证原始advanced建议在普通策略中应用advanced、在资源消融中应用default；CLI入口已修复并验证。测试确认累计耗时超过24小时仍正常执行。软件验证不计入96格研究实验。

## 已具备的执行能力

- `orion_repro.study`：串行独立子进程，自动复用、累计耗时、原始日志、资源失败保留、实现错误暂停、用户中断处理。
- reuse v2：新增无关配置、矩阵、文档不使已完成格失效；源码/实际配置/依赖锁/manifest变化仍失效。旧身份不强行转换。
- `controller.plugin_policy=fixed_default`：显式关闭插件切换的资源自适应消融；保留原始URGE建议与实际应用模式。
- `orion_repro.light24 --select`：只按当前身份的12个开发候选选择静态基线，不用正式测试选参。
- `orion_repro.study_report`：当前身份覆盖、全部尝试、失败及均值/样本标准差；历史结果不混入。

## 版本控制

当前状态纳入初始 Git 提交 `Establish light24 reproduction baseline`。具体提交号用 `git log -1 --oneline` 查看。接手先检查 `git status --short`；必要数据 manifests 纳入版本控制，原始数据、runs、权重及缓存保留在本地并忽略。本次仅固定版本基线，未改变训练代码/协议或启动实验，未推送远端。

## 接手顺序

命令直接见 README。先确认GPU没有其他项目训练，再依次执行：

1. `study --matrix experiments/light24/core.yaml`。
2. `study --matrix experiments/light24/search.yaml`。
3. `light24 --select`，然后 `study --matrix experiments/light24/selected.yaml`。
4. `study --matrix experiments/light24/extensions.yaml`（budget→preference→prefetch→sensitivity→endless）。
5. `study_report`，完成图表及C01–C08限定结论，更新验收表。

以上模块均用 `/home/admin/miniconda3/envs/orion/bin/python -m orion_repro.<模块>`。无需重新生成已有配置。阶段可分开运行，不必等一个无输出的大任务；原始子进程日志在 `runs/_light24_executor/`。

## 当前边界与待验收

- 新矩阵覆盖情况见 `reports/light24/progress.json`，当前为0/96；软件测试不计正式实验。
- 当前新增代码验证摘要见 `reports/light24/readiness.json`；配置dry-run不代表训练完成。
- source更新后不能强行复用旧结果；同一科学比较使用一致实现。若未来修复改变训练行为，重跑受影响配对。
- search目前要求12格全部completed才自动选择；有真实资源失败须形成记录并明确可行候选规则后再选，不能静默漏格。
- 预取对照只覆盖静态轨迹，动态控制下预取等价性不在本轮承诺内；主实验预取关闭。
- allocator不等于host/全GPU硬限制；Endless排序代理不证明真实时间隔离；主要协议含官方测试反馈；这些局限保留。
- GSS历史崩塌、预取变慢、MAX-A弱结果仍须如实解释。LR×非ER、完整Oracle、全成本模型和全checkpoint验收已移出本轮范围。

完成以PLAN和docs/acceptance.csv为准，不以达到小时数或复现出正向结论为准。
