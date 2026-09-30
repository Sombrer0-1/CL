# Thor host 限额可行性实测（2026-09-20）

结论：用户级 systemd cgroup v2 的 host 硬限额已跑通，不需要 sudo、重启、驱动修改或全局配置修改。但本机普通 PyTorch CUDA allocation 未被该 cgroup memory controller 完整计费，因此这不是 CPU+GPU 总统一内存硬上限。这里仅做能力验收，未启动阶段 4 正式持续学习实验，也未修改第三阶段源码、冻结协议或裁决。

## 原因定位

- 当前会话：`/user.slice/user-1001.slice/session-221.scope`。
- `/sys/fs/cgroup` 已挂载 cgroup2，可用 controller 含 memory。
- `user@1001.service` 已有 `Delegate=yes`，用户可用 controller 为 cpu/memory/pids。
- 旧 `src/orion_repro/memory/host_enforcement.py:cgroup_memory_max` 仅找层级根的 memory.max 或 v1 路径；根目录没有 memory.max 并不意味着子 cgroup 无能力。
- 旧阶段 `record_host_capability` 调用 `probe_delegated_cgroup(None)`，没有尝试用户级 systemd 入口。这解释了旧 unavailable 的探测缺口，但不改写旧阶段未执行 H 的事实。
- `sudo -n -l` 要求密码；实际完成以下验证没有使用 sudo。

## 实测

所有工作负载在 systemd 启动时就进入限额组，再导入 torch/初始化 CUDA；解释器固定为 orion。`MemorySwapMax=0`，本机系统本身也无 swap。普通 CPU/CUDA 探针分配均实际写入，CUDA 写入后同步；不会仅靠虚拟地址预留判断。

| 探针 | memory.max | 结果 |
|---|---:|---|
| CPU bytearray 逐步分配 | 128 MiB | 峰值触及 128 MiB，memory.events 的 oom=1、oom_kill=1 |
| 由子进程分配 CPU 内存 | 128 MiB | 同样触发 cgroup OOM，子进程受同一预算约束 |
| CUDA tensor 逐步分配并 fill/synchronize | 2 GiB | 3 GiB 张量同时存活仍成功，cgroup 峰值约 384 MiB，oom=0 |
| CUDA 复核 | 512 MiB | 同样持有 3 GiB CUDA 张量成功，cgroup 峰值约 371 MiB，oom=0 |
| pinned CPU tensor | 2 GiB | 1.5 GiB payload 后下一次分配触发 cgroup OOM，oom_kill=1 |
| 新文件写入 384 MiB | 128 MiB | 成功，cgroup 峰值 128 MiB、max 事件增加、无 OOM；日志保存 memory.stat 的 file/回收项，说明可回收页缓存不是必然导致 OOM |
| 小型 GPU 网络 5 次 SGD 更新 | 2 GiB | 成功，参数确实改变；cgroup 峰值约 794 MiB。这仅是兼容性检查，非正式复现实验 |

外部父进程每 20 ms 采样受试 cgroup，并保存 worker 每步快照、memory.peak/stat/events 和进程组信息。子进程 OOM 不会杀死外部记录器。`systemd-run` 结束时打印的 Memory peak 明显低于实时 cgroup 读数，本报告使用保存的 cgroup 文件读数；不能依赖其结束摘要。cgroup 生命周期结束后消失，最后事件采用消失前采样；更稳健的正式执行器仍需保证退出证据完整回收。

主结果见 `summary.json`，复核和页缓存见 `summary_extra.json`；各 `.log` 包含逐步证据，`*_samples.json` 保存外部采样。三个预期 OOM 已保留，不能当成实现失败或删除。另一次手动 CUDA 复核用了相对脚本路径，systemd 默认工作目录导致文件未找到（exit 2）；随后用绝对路径完成上述复核。正式入口必须指定工作目录或全部使用绝对路径。

## 可用入口与重放

从项目目录运行：

```bash
/home/zhuzetong/miniconda3/envs/orion/bin/python reports/host_feasibility/20260920/probe.py
/home/zhuzetong/miniconda3/envs/orion/bin/python reports/host_feasibility/20260920/probe.py --extra
```

脚本默认覆写本目录同名探针输出；如需保留本次证据，先复制整个探针目录到新的日期或 revision 再运行。临时文件由 TemporaryFile 自动回收；受试 transient services 已退出，失败状态已清理。没有持久 systemd unit 或系统配置变更。

未来训练入口可以使用：

```bash
systemd-run --user --wait --pipe --unit=orion-h-UNIQUE_RUN_ID \
  -p MemoryMax=CALIBRATED_BYTES -p MemorySwapMax=0 \
  -p MemoryAccounting=yes -p OOMPolicy=stop \
  -p WorkingDirectory=/home/zhuzetong/research/cl/Reproduce-Orion \
  /home/zhuzetong/miniconda3/envs/orion/bin/python ABSOLUTE_ENTRYPOINT
```

这是命令模板，不是冻结配置；预算、入口与唯一 ID 必须替换。正式阶段还需接入配置身份、退出分类、组外记录器、replay 物理存储审计和资源校准。

## 对阶段 4 的影响

1. H 的 host 预算路线可开展，目前不需要用户提供权限。systemd 管理实验专属组即可，不必向旧 HostEnvelope 手动提供根层权限。
2. 单纯增加 sudo 权限不能被当作修复 CUDA 计费缺口的办法。总内存约束必须先找到并验证覆盖 CUDA 的执行机制。
3. 可立即研究的是 host cgroup 与 device allocator 分别限额的联合预算，但应明确称为分池限额；固定拆分不能等同允许 CPU/GPU 互借的一个总预算。
4. cgroup 计费与 device reserved 不能未经重叠/遗漏核验直接相加。采样 watchdog 也不能称硬总内存限额。若选择此类工程方案，需要独立命名并审计计费与超调。

参考：[Linux cgroup v2 官方文档](https://docs.kernel.org/admin-guide/cgroup-v2.html)。文档描述 memory controller 与层级语义；本机 CUDA 是否计费的判断来自以上实际分配试验，不从统一内存架构或文档泛化推断。
