# G1 日志盘满模拟

日期：2026-09-22。`mem=16G`。不是内存压力场景，不是 G3 冻结，不填系统盘。

## 做了什么

- 单测：`process_one` 在 `disk_low` 时 pause，inbox 保留、不进 running。
- 单测：`health_check` 报告 `disk_low`。
- 真机：在仓库下挂 **256MiB tmpfs** 作为独立探针根，对 `g1-diskfull-a` 调 `process_one`。系统盘 `/` 仍约 791GiB 空闲。

## 结果

| 项 | 值 |
|---|---|
| 探针根空闲 | 256MiB（`268435456` 字节） |
| `MIN_FREE_BYTES` | 10GiB |
| `process_one` | `paused` / `disk_low:268427264` |
| inbox | 保留 `g1-diskfull-a` |
| running / done | 无 |
| 生产队列 | 未动，仍 idle |
| `/` | 97G/936G used，791G avail |

证据：`reports/fullmem_v4/g1/diskfull_probe.json`。

这只证明执行器在日志盘不够时停发并留证。不能把它写成 URGE 压力已建立。

## 仍未做（本报告时）

- 运行中整机重启
- 硬整机 OOM
