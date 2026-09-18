# Jetson AGX Thor 环境锁（当前宿主）

2026-09-16 按 [A27](../../decisions/A27.md) 建立。不要在 x86_64 / RTX 5090 / 5060 Ti 上按此清单安装。

- 解释器：`/home/zhuzetong/miniconda3/envs/orion/bin/python`
- Python 3.11.16，`torch==2.14.0+cu132`、`torchvision==0.29.0+cu132`、`avalanche-lib==0.6.0`
- 平台：aarch64，JetPack 7.2.1，L4T R39.2.1，驱动 595.78，CUDA 13.2，GPU NVIDIA Thor / sm_110
- 重建：见仓库根 `environment.yml` 注释；先装 Miniconda aarch64 到 `~/miniconda3`

锁文件哈希（建立时，与 `docs/environment_hashes.txt` 中 Jetson AGX Thor 两项相同）：

- `requirements.lock.txt` `2441b78017080812aecdee490a6fa44e18b03635aa5e64f0a8c249caee7424b4`
- `environment.lock.yml` `37206057b24e358bf2ea57867df55b9105802ce3bc67af5b4f318bff02a4ff9a`

2026-09-17复查：真实GPU参数更新与64MiB allocator失败阶段诊断通过；全套回归见`reports/effectiveness_v3/readiness/tests.txt`。`pip check`不是全绿：cusparselt 0.8.1携带`manylinux2014_sbsa`标签，被pip报告平台不支持。未重写wheel元数据；当前dense FP32路径已验证，稀疏算子不在此次验证范围。详见[A28](../../decisions/A28.md)。无驱动、base或mineru改动。
