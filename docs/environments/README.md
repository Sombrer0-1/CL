# 环境锁文件

根目录 `requirements.lock.txt` 与 `environment.lock.yml` 对应**当前主机**（Linux + 双 RTX 5090，`~/.conda/envs/orion`，`torch==2.11.0+cu128`）。哈希见 `docs/environment_hashes.txt`。

`wsl2_rtx5060ti/` 是第一、二阶段原平台锁文件（`torch==2.14.0+cu130`，`/home/admin/miniconda3/envs/orion`）。只用于复核当时身份，不要在本机 `pip install -r` 那份清单。

`reports/envcheck.json` 是本机最近一次 T01 结果，会被新的 envcheck 覆盖。
