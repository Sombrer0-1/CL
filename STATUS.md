# 当前工作状态

更新：2026-09-14。有效范围为 **light24_v1 代表性方法复现**，见 PLAN.md。旧全覆盖目标已被用户授权替换，不再恢复旧MAX-A队列。

## 当前阶段

**light24_v1 全部 96 格执行完毕并完成结果分析：90 completed + 6 个 128MiB 启动 OOM（有证据的资源失败，保留）。最终结果见 [reports/light24/RESULTS.md](reports/light24/RESULTS.md)，C01–C08 判定见 [docs/claims_status.md](docs/claims_status.md)。**

- 执行顺序 core → search → selected → extensions（budget→preference→prefetch→sensitivity→endless）全部完成；累计正式训练 6725.6s（约 1.87h，见 runs/light24_progress.json）。
- 主要比较 3 seeds，均值/样本标准差/配对差/失败数齐备：means.csv、paired_diffs.csv、run_artifacts.csv、overhead.csv、budget_summary.csv；图在 reports/light24/figures/。
- 核心 finding：URGE 峰值 0.006 远低于最低受试阈值 0.025（延迟因子主导），自适应从未触发；完整 Orion ≈ 初始资源静态 ER；6 格搜索选出的 b16/r2000 在两场景 (P+S)/2 均最优；预取语义严格等价但 +31–44% 耗时；128MiB 配额两方法全部启动 OOM。
- Endless IC 的 Orion 时间优势经 4 次重跑探针证伪为进程级主机计时方差（同配置 t0 波动 5.9–9.0s）；探针保留在 runs/。

## 边界与遗留

- 96 格中 6 格为 cuda_oom 资源失败（budget 128MiB×两方法×3 seeds），按协议视为有证据的真实结果，不补跑、不调参隐藏。
- C04（多算法普适性）本轮范围外；Jetson 能耗、原设备耗时、机器人闭环仍为平台外。
- paper_feedback 口径含官方测试反馈；开发选择未使用该数据源。
- 未构造出“配额可满足且内存紧张”的受压区（可行下界 ~129MiB > 128MiB 配额）；控制器在真实内存压力下的行为未验证，列为后续研究项（RESULTS.md §6）。
- 若未来源码修复改变训练语义，按 reuse v2 规则重跑受影响配对组；当前无未完成执行队列。

## 版本控制

代码、配置、文档与研究记录纳入 Git；数据、runs、权重、缓存仅本地（.gitignore 排除）。远端 `git@github.com:Sombrer0-1/CL.git`（origin/master）。本轮交付提交包含 light24 报告产物与分析脚本；原始子进程日志在 `runs/_light24_executor/`。

## 复现命令

全部矩阵与报告可用 README「从仓库根目录恢复」一节的命令重跑；`python reports/light24_analysis.py`（orion 环境）重建配对差、开销表与图表。
