# 数据与划分

已有数据保存在 `data/raw/`，派生文件在 `data/processed/`，来源、校验和和划分统计在 `data/manifests/`。不需要重新下载或把完整图像集装入 RAM。

| 数据 | 当前用途与协议 |
|---|---|
| CIFAR10 / CIFAR100 | 32×32，均 10 experiences；开发集从官方 train 按类别保留 20% val，split seed=17 |
| CORe50 mini | 32×32、50 对象类；NC/NI/NIC 为 9/8/79 experiences；开发划分按 `s*/o*` 整段序列 |
| Endless classification | IC/IL/WC 为 4/5/5 experiences；32×32；当前分组协议见 [A22](decisions/A22.md) |

Endless 每次运行保存 `runs/<run_id>/split_manifest.json`，包含路径及 train/val/excluded 索引。数字 patch ID 的时间含义尚未确认。CORe50 NIC 映射到 Avalanche `nicv2_79` 是重建选择，不是已确认的作者配置。

需要在新环境准备数据时，在仓库根目录按需执行：

```bash
/home/admin/miniconda3/envs/orion/bin/python -m orion_repro.prepare_data --dataset cifar10
/home/admin/miniconda3/envs/orion/bin/python -m orion_repro.prepare_data --dataset cifar100
/home/admin/miniconda3/envs/orion/bin/python -m orion_repro.prepare_data --dataset cifar10_dev
/home/admin/miniconda3/envs/orion/bin/python -m orion_repro.prepare_data --dataset cifar100_dev
/home/admin/miniconda3/envs/orion/bin/python -m orion_repro.prepare_core50
/home/admin/miniconda3/envs/orion/bin/python -m orion_repro.prepare_data --dataset core50_dev_nc
/home/admin/miniconda3/envs/orion/bin/python -m orion_repro.prepare_data --dataset core50_dev_ni
/home/admin/miniconda3/envs/orion/bin/python -m orion_repro.prepare_data --dataset core50_dev_nic
/home/admin/miniconda3/envs/orion/bin/python -m orion_repro.prepare_endless --scenario all
/home/admin/miniconda3/envs/orion/bin/python -m orion_repro.prepare_endless --scenario all --extract
```

开发校准仅使用训练集内验证数据。官方测试曾在早期探索中暴露，不能宣称从未接触；正式 `paper_feedback` 的测试反馈也不能用于人为搜索配置。
