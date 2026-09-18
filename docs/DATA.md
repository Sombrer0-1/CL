# 数据与划分

`data/raw/` 与 `data/processed/` **不纳入 Git**（`.gitignore`）。clone 后必须在新机器重新下载并生成划分；Git 只保留 `data/manifests/` 里的来源、校验和与划分统计。不要把全集原图载入 RAM。

历史宿主磁盘占用参考（不代表本机数据已准备）：CIFAR-10 约 341 MiB，CIFAR-100 约 339 MiB，CORe50 mini 32×32 约 3.4 GiB，Endless 分类约 14 GiB。第三阶段正式比较只需 **CIFAR-100 + CORe50 mini**（约 3.7 GiB）；Endless 不是 effectiveness_v3 必下集合。

| 数据 | 当前用途与协议 |
|---|---|
| CIFAR10 / CIFAR100 | 32×32，均 10 experiences；开发集从官方 train 按类别保留 20% val，split seed=17 |
| CORe50 mini | 32×32、50 对象类；NC/NI/NIC 为 9/8/79 experiences；开发划分按 `s*/o*` 整段序列 |
| Endless classification | IC/IL/WC 为 4/5/5 experiences；32×32；当前分组协议见 [A22](decisions/A22.md) |

第三阶段迁移使用 `"$ORION_PY" -m orion_repro.stages.effectiveness_v3.prepare`，它不覆写历史manifest。下述旧命令会重写manifest，仅供历史复核，不作为本次迁移命令。

获取渠道写在准备脚本里：

| 数据 | 脚本 | 下载来源 |
|---|---|---|
| CIFAR-10/100 | `orion_repro.prepare_data` | torchvision 官方镜像（`download=True`） |
| CORe50 mini 32×32 | `orion_repro.prepare_core50` | Avalanche `CORe50Dataset(mini=True)`：`http://vps.continualai.org/data/core50_32x32.zip`，划分文件来自 [CORe50 官方页](https://vlomonaco.github.io/core50/) |
| Endless 分类 zip | `orion_repro.prepare_endless` | Zenodo [4899267](https://zenodo.org/record/4899267)；三条 URL 与 md5 写在脚本里 |

换机后先建好 `data/raw`、`data/processed`（普通目录或新数据盘的符号链接），再在仓库根目录执行：

```bash
ORION_PY="${ORION_PY:-$HOME/miniconda3/envs/orion/bin/python}"
"$ORION_PY" -m orion_repro.prepare_data --dataset cifar10
"$ORION_PY" -m orion_repro.prepare_data --dataset cifar100
"$ORION_PY" -m orion_repro.prepare_data --dataset cifar10_dev
"$ORION_PY" -m orion_repro.prepare_data --dataset cifar100_dev
"$ORION_PY" -m orion_repro.prepare_core50
"$ORION_PY" -m orion_repro.prepare_data --dataset core50_dev_nc
"$ORION_PY" -m orion_repro.prepare_data --dataset core50_dev_ni
"$ORION_PY" -m orion_repro.prepare_data --dataset core50_dev_nic
```

仅当需要历史 Endless 辅助实验时再执行：

```bash
"$ORION_PY" -m orion_repro.prepare_endless --scenario all
"$ORION_PY" -m orion_repro.prepare_endless --scenario all --extract
```

Endless 每次运行还会在 `runs/<run_id>/split_manifest.json` 保存路径及 train/val/excluded 索引。数字 patch ID 的时间含义尚未确认。CORe50 NIC 映射到 Avalanche `nicv2_79` 是重建选择，不是已确认的作者配置。

`data/manifests/` 中的 `/home/admin/...` 绝对路径是原机准备时的溯源字段，不要为搬家改写（会改变哈希）。训练配置使用相对 `data/raw/...` 与 `split_dir`，经仓库根与符号链接解析。

开发校准仅使用训练集内验证数据。官方测试曾在早期探索中暴露，不能宣称从未接触；正式 `paper_feedback` 的测试反馈也不能用于人为搜索配置。
