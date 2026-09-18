# 环境锁文件

当前宿主：**Jetson AGX Thor**，锁在 [jetson_agx_thor/](jetson_agx_thor/)，与根目录 `requirements.lock.txt` / `environment.lock.yml` 相同。解释器 `/home/zhuzetong/miniconda3/envs/orion/bin/python`，`torch==2.14.0+cu132`。哈希见 `docs/environment_hashes.txt`。

其它目录是**离机归档**，只用于核对历史实验身份，禁止当作安装配方：

- [linux_rtx5090/](linux_rtx5090/)：Linux + 双 RTX 5090，`torch==2.11.0+cu128`
- [wsl2_rtx5060ti/](wsl2_rtx5060ti/)：第一、二阶段原平台，`torch==2.14.0+cu130`

`reports/envcheck.json` 是本机最近一次 T01 快照，会被新的 envcheck 覆盖。
