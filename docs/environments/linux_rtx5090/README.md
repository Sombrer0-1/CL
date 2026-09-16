# Linux + 双 RTX 5090 环境锁（归档）

2026-09-15 在该宿主建立 `orion` 环境（A24）。2026-09-16 按 [A26](../../decisions/A26.md) 放弃此机上的 effectiveness_v3 校准与正式跑数。不要在新机器上按此清单安装，也不要把这里的 PyTorch/CUDA 选择当成下一宿主的必选值。

- 解释器当时为 `/home/zhuzetong/.conda/envs/orion/bin/python`
- `torch==2.11.0+cu128`、`torchvision==0.26.0+cu128`、`avalanche-lib==0.6.0`
- 根目录 `environment.lock.yml` / `requirements.lock.txt` 在换机前仍可能是这份快照；新宿主应重新锁定并更新哈希

锁文件哈希（归档时，与 `docs/environment_hashes.txt` 中 Current host 两项相同）：

- `requirements.lock.txt` `47a32c4df58522a909c6643fe5005c31b6cc2e168240715aa883ad18eff4cd69`
- `environment.lock.yml` `44b137cb87153aa0bdd645051616f89c279e663b1b90aba2cc661299d44a7bff`
