# Linux + 双 RTX 5090 环境锁与离机快照（归档）

**不是当前执行环境。** 当前宿主是 Jetson AGX Thor，见 [A27](../../decisions/A27.md) 与 [jetson_agx_thor/](../jetson_agx_thor/)。

2026-09-15 在该宿主建立 `orion` 环境（A24）。2026-09-16 按 [A26](../../decisions/A26.md) 放弃此机上的 effectiveness_v3 校准与正式跑数。不要按此清单安装，也不要把这里的 PyTorch/CUDA 选择当成 Thor 的必选值。

- 解释器当时为 `/home/zhuzetong/.conda/envs/orion/bin/python`
- `torch==2.11.0+cu128`、`torchvision==0.26.0+cu128`、`avalanche-lib==0.6.0`

归档材料：

| 文件 | 内容 |
|---|---|
| `requirements.lock.txt` / `environment.lock.yml` | 当时 pip/conda 锁 |
| `validation_gpu.json` / `validation_tests.txt` | 当时 allocator OOM 诊断与 pytest 日志 |
| `restart_readiness.md` | 迁入 5090 时的仓库整理快照 |

锁文件哈希（归档目录现行文件；仅文件头从 Current host 改为 Archived host，包列表未改）：

- `requirements.lock.txt` `66095c090cb7a3449e97103d5ac74a531c1d6850b75e80cdc48e6cc0180ae659`
- `environment.lock.yml` `e134b20be9e65f9a7bc533c6da40cd006b140860d891d02b14f3e41d52b1293a`
