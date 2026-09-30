# Orion 论文复现

本仓库依据本地原论文重建 Orion，不是作者官方实现。当前进入 **阶段4 fullmem_v4 的计划与平台准备**：采用真实整机共享内存约束，由远程控制机的agent通过SSH调度Thor。

**尚未限制整机内存、尚未重启、尚未冻结预算或启动阶段4正式实验。** 已验证host cgroup限制，但普通CUDA分配未被完整计费，不能用它冒充总内存基准。

| 阶段 | 状态 |
|---|---|
| light24_v1 | 已结项，未证明Orion优势 |
| pressure_v2 | 中断收尾，未验收 |
| effectiveness_v3 / Thor thor_r2 | 已结项，129终态（75完成/54 OOM），主比较不支持优势 |
| fullmem_v4 | 计划建立；真实总预算与远程运行准入待完成 |

## 当前入口

- [PLAN](PLAN.md)：阶段4完整计划、预算机制、远程模式、四条主线、矩阵与验收。
- [STATUS](STATUS.md)、[HANDOFF](docs/HANDOFF.md)、[MEMORY](docs/MEMORY.md)：当前事实与下一步。
- [A30](docs/decisions/A30.md)：阶段4选择及对旧约束的替代范围。
- [host实测](reports/host_feasibility/20260920/README.md)：为什么cgroup不足以成为总内存限额。
- [仓库地图](docs/REPO_MAP.md)、[清理记录](docs/cleanup_20260920.md)：保留证据、历史入口和清理边界。
- [AGENTS](AGENTS.md)：项目稳定约束；[环境](docs/environments/jetson_agx_thor/README.md)、[数据](docs/DATA.md)、[结果](reports/README.md)。

## 运行边界

项目脚本与实验显式使用 `/home/zhuzetong/miniconda3/envs/orion/bin/python`；禁止mineru/base和擅自升级驱动。没有阶段4正式执行命令；旧入口不是新队列。

控制机上的agent负责调度和分析；Thor只保留实验、轻量执行器和记录器。16GiB仅是候选，整机启动限制须先验证并安排重启确认。

数据及原始runs位于 `data/raw`、`data/processed`、`runs`，不纳入Git；仓库备份不等于证据备份。历史失败与开发探针保留，不能跨阶段合并统计。

第三阶段最终裁决见 [CLAIMS](reports/effectiveness_v3/thor_r2/CLAIMS.md)，旧计划见 [effectiveness_v3](docs/plans/effectiveness_v3.md)。第一、二阶段见 [light24](docs/plans/light24_v1.md)、[pressure_v2](docs/plans/pressure_v2.md)。
