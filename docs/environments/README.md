# 环境锁文件

根目录 `requirements.lock.txt` 与 `environment.lock.yml` 曾对应 **Linux + 双 RTX 5090**（`~/.conda/envs/orion`，`torch==2.11.0+cu128`）。该宿主快照另存 [linux_rtx5090/](linux_rtx5090/)。哈希见 `docs/environment_hashes.txt`。换机后不要 `pip install -r` 这份清单；按新 GPU/驱动重建 `orion` 并更新锁与哈希。

`wsl2_rtx5060ti/` 是第一、二阶段原平台锁文件（`torch==2.14.0+cu130`，`/home/admin/miniconda3/envs/orion`）。只用于复核当时身份。

`reports/envcheck.json` 如存在，只是某次 T01 快照，会被新的 envcheck 覆盖。
