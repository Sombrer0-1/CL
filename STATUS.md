# 当前工作状态：pressure_v2 已按中断矩阵收尾，未验收

light24_v1 结项证据保持不变。2026-09-15 已保存 I1 修补，并按 SDD 实现 PhaseRecorder、ResourceEnvelope、协议冻结/发射、独立执行器与报告。开发校准与冻结已完成。正式 54 格在原 WSL2 + RTX 5060 Ti 上于用户指示下中断后收尾；**矩阵未跑完，不得声称第二阶段验收通过。**

仓库已迁到 Linux 服务器（双 RTX 5090）。**不要**在本机续跑剩余格并并入同一 54 格比较。本机 `orion` 环境已于 2026-09-15 建立（`torch==2.11.0+cu128`，见 [A24](docs/decisions/A24.md) 与 `reports/envcheck.json`）。下一步是新阶段设计与校准，不是续跑 `formal.yaml`。

## 第二阶段收尾入口

- 收尾说明：[reports/pressure_v2/CLOSEOUT.md](reports/pressure_v2/CLOSEOUT.md)；进度摘要：`reports/pressure_v2/progress.json`、`attempts.csv`。原始 runs 仍在 `runs/`（不纳入 Git）。
- 方案：[PLAN.md](PLAN.md)；设计清单：`experiments/pressure_v2/design.yaml`；架构：[docs/SDD_pressure_v2.md](docs/SDD_pressure_v2.md)；准备验收：[docs/pressure_v2_acceptance.csv](docs/pressure_v2_acceptance.csv)。
- 冻结协议 `experiments/pressure_v2/frozen_protocol.json`，hash `863ac7b0f08570d985be0c2a5e0dd4e692088d0812854ce57455f58a1907471a`。eval_batch=32；Q_tight=176 MiB；Q_loose=352 MiB；L_cal≈5.94s（原 5060 Ti）；各预算 S* 均为 b16/r2000；DYN 在 k=3..5 预留 32505856 B；IO 开发 wait_ratio≈0.50。
- 执行器（仅原平台复核，不是本机待办）：`PYTHONPATH=src conda run -n orion python -m orion_repro.pressure_study --matrix experiments/pressure_v2/formal.yaml`。进度文件 `runs/pressure_v2_progress.json`，不写 light24。
- 正式进度（收尾时）：54 格中已记录 45（completed 30、cuda_oom 14、interrupted 1）；未启动 9 格（剩余 3 个 PREF + 全部 6 个 IO）。中断格为 `pref/O11_ps_s1`，进度 `active=null`。
- 下一步：新阶段设计与本机资源校准。不以 URGE 触发代替有效性，也不把未完成矩阵续成跨机混合结果。旧 5060 Ti 的 `L_cal`、配额与耗时不自动沿用。

## 本轮修补与证据保护

第一阶段完整配置没有装载可切换插件；此前advanced字符串检查不足以证明启用了GEM/EWC。现运行接口在无插件时拒绝advanced，并返回真实开关状态。第一阶段核心轨迹始终default，P/S保留，但完整插件能力仍未验证。源码修补使reuse身份变化，不把冻结成果变成“待重跑96格”。冻结索引为reports/light24/stage_manifest.json，原标签不动；不要用当前源码study_report覆写第一阶段覆盖表。

空buffer初次resize越界与OOM摘要缺失训练/评价阶段也已修补。新模块主体、资源压力场景、全GPU插件行为仍须按SDD验收，不能将单元测试通过写成方法有效性。

本轮验证：104项测试通过（9条库告警，含真实CPU插件生命周期）；GPU小型检查确认evaluation OOM时已训练1个experience、完成评价0个，失败阶段记录正确。摘要见reports/pressure_v2/readiness.json。

开发校准要点（正式前冻结，不按正式 P/S 选参）：CIFAR100 eval_batch=128 评价 OOM，32/8 可完成；CORe50 128 MiB OOM，176 MiB 训练 reserved/配额≈0.886；O00（L=30）可完成且收缩；O10/O11 开发路径启用 GEM/EWC 后于 experience 6 训练 OOM（非未公告过渡）。正式矩阵已中断收尾，未完成验收。

## 第一阶段成果

- 96格全部执行：90 completed + 6个128MiB CUDA OOM，失败证据保留，符合既定资源失败验收口径。
- 六例OOM均发生在首个experience训练后的评价阶段（eval_batch=128），不是训练启动失败；详见 `reports/light24/budget_failure_stages.csv`。
- 矩阵子进程累计墙时6725.6s（约1.87小时，含开发搜索、启动、训练、评价；不含软件验证及4次额外重跑探针）。
- 两场景主要比较、有限开发搜索及获选配置评价、预算、偏好、预取、阈值敏感性、Endless辅助验证齐备。主要比较3 seeds；既有100项测试和两次小型GPU验证有记录，本次结项不重新训练。
- 配置、代码、汇总脚本、逐格表、配对差、开销表、7张图和验收表已交付；历史71次正式运行仍是背景，不混入新比较。

## 核心认识与边界

1. 控制器发生batch16→15与replay收缩，但未进入资源扩张或高级插件切换分支；不能称完全没有自适应。
2. 有限搜索b16/r2000在本轮(P+S)/2指标下优于受试Orion；更大replay是否更好未知，不能把搜索边界当作未饱和的证明。
3. 128MiB完整协议在评价阶段失败、256/512MiB完成；训练精确可行边界和真正受压时的自适应能力未验证。
4. 预取on/off完整准确率矩阵和访问计数逐seed一致，时间增加31–44%；不据此证明每步训练严格等价，具体内部开销尚未直接剖析。
5. Endless为开发划分辅助证据，真实时间隔离未经证明；同配置重跑存在明显计时波动，小幅速度差不能直接归因于方法。
6. paper_feedback含官方测试反馈；C04新多算法扩展、Jetson能耗、机器人闭环等不在本阶段范围内。
7. 第二阶段在紧配额下观察到开发/正式路径的训练期 cuda_oom（含 GEM/EWC），以及 30 格完成与 14 格正式 OOM；因矩阵未跑完且换机，不据此给出方法有效性结论。

完整报告：[reports/light24/RESULTS.md](reports/light24/RESULTS.md)。限定结论：[docs/claims_status.md](docs/claims_status.md)。验收：[docs/acceptance.csv](docs/acceptance.csv)。第二阶段收尾：[reports/pressure_v2/CLOSEOUT.md](reports/pressure_v2/CLOSEOUT.md)。

## 结项与后续边界

- 第一阶段结项时仅修正文档解释、补充OOM阶段索引并固定阶段版本；原始数据、训练代码、配置和结果数值不变。
- 阶段标记：`light24-v1-complete`。此前实验/分析提交为 `8aa6654`、`8036f59`；标记指向包含评审修订的结项提交。
- Git远端：`git@github.com:Sombrer0-1/CL.git`。代码、文档、必要结果表和图纳入Git；data/raw、data/processed、runs、模型和缓存仅保留本地，远端不是完整原始数据备份。
- 第二阶段实现、冻结协议与中断进度已纳入仓库记录；正式比较未完成、未验收。第一阶段不因此重新打开为未完成状态。
- 本机 `orion` 环境已建立，下一步是新阶段设计，不是续跑 `formal.yaml`。

README中的light24命令仅供第一阶段复核，不能用本机新环境身份覆盖冻结结果。
