# 预取正确性与性能协议

正确性验证固定 manifest、初始化、采样、数据增强及训练配置，比较实际输入 hash、标签、loss 与完整模型状态。当前小型真实 ER 的确定性 GPU 两经验路径 76 步逐位一致；该证据不覆盖所有算法和完整流。

显式诊断选项 `training.deterministic_algorithms=true` 同时开启 Torch deterministic algorithms 与 cuDNN deterministic；进程启动前设置 `CUBLAS_WORKSPACE_CONFIG=:4096:8`。不支持的确定性算子应报错，不自动回退。该诊断变体不混入默认吞吐主表。

正式 E07 需固定配置/轨迹，在开发集至少三组 off/off 重复上量化方差，然后预先确定容差及统计方法，再执行配对 on/off。不能用单次参数距离作为通用容差或归因全部历史差异。需要全流、多算法和队列配置覆盖；C07 仍尚未验证。
