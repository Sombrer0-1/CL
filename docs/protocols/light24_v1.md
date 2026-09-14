# light24_v1 当前实验协议

2026-09-14 冻结范围与配置。当前研究范围仅见 PLAN；此协议用于代表性方法验证，不承诺论文所有结论成立。

| 项目 | 采用设置 |
|---|---|
| 主场景 | SplitCIFAR100 10 experiences；CORe50-NC run0 9 experiences |
| 模型与训练 | ResNet-20，FP32，SGD lr0.01/momentum0.9；新数据1 epoch；约定完整训练流 |
| Seed | model/stream/replay/augmentation 同为0、1、2；已冻结类序与数据划分不随选参改变 |
| URGE | balanced k=0.25；P/S阈值0.5；L_th=30s；M_max=4096MiB；Thr0=0.05；delta=0；alpha0.1/beta0.2 |
| 初始资源 | new batch16；replay batch16；capacity200；MB0=16/MR0=200；保留既有单位和映射假设 |
| 主反馈 | official_test_seen，P_diag/S_initial；论文口径重建，存在测试反馈，非独立泛化评价 |
| 主预取 | off；预取对照单独开启确定性算法，不能跨组直接比耗时 |
| 仅资源消融 | controller.plugin_policy=fixed_default，URGE/阈值/batch/replay原式不变，仅实际插件模式固定default；保留原始建议 |
| 有限静态搜索 | 每场景6格：batch16/64/256 × replay200/2000；开发seed0；同训练实现，控制器关闭 |
| 选择规则 | (P_diag+S_initial)/2最大；平局online_total_s更短，再配置路径字典序；独立于正式测试；获选配置另跑3 seeds |
| 显存实验 | CIFAR100静态ER/Orion，allocator配额128/256/512MiB，M_max同步配额。启动OOM也是真实结果；不通过调整正式预算隐藏失败 |
| 偏好 | CORe50-NC：latency优先顺序L/M/P/S；P-S优先P/S/M/L；按4/3/2/1归一化。Balanced复用core |
| 敏感性 | 两场景Thr0=0.025/0.1；主值0.05复用core；其他参数固定，不选最好者回填主表 |
| 预取 | CORe50-NC静态ER，on/off三seed，队列depth2、worker0，其余配置相同；检验指标、new/replay访问、等待和内存 |
| Endless | grouped_v2完整开发流，split_seed17/val0.2/embargo32；IC/IL/WC × ER/Orion ×3seeds；辅助验证，不能混称官方测试结果 |
| 时间 | 24小时左右的工作量目标，仅估算，不强制停止正常训练 |

## 证据与解释

所有比较保留准确率矩阵、URGE建议/实际控制、运行配置、环境/源码/manifest和资源轨迹。静态方法与Orion使用同种数据加载及训练后端。报告学习质量、时间、内存的联合权衡；不只选一个有利指标。

有限搜索的开发成本与选中配置的正式单次成本分开。选择器拒绝未完成候选；发生资源失败时先形成具体记录，再明确修订候选可行性处理，不静默漏格。

PyTorch allocator实验不等于host硬限制或完整GPU物理限制。本轮不依赖complete=false成本模型施加虚假的硬保证。不能从单条成功运行推出所有预算下均无OOM。

Endless排序代理无法证明真实帧级独立性；如不能核实时间映射，则结果只用于该已声明划分的探索，不能当作无偏迁移证据。

旧正式结果与新study分开汇总。主比较与消融配对若受实现修复影响须一起重跑。训练参数相同不代表源码不同的运行可以直接合并。
