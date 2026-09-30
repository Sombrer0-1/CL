# G1 运行中整机重启

日期：2026-09-22。`mem=16G`。不换档，不改 nvpmodel。不是 G3 冻结，不是硬整机 OOM。

## 做了什么

- 先修执行器：`host_reboot_with_open_work` 必须一次性记下新 boot_id，否则会挡住 `finalize_dead`。单测已覆盖。
- 提交 `g1-reboot-a`（dummy 180s）与 `g1-reboot-b`（dummy 5s）。
- a 已在跑时 `sudo reboot`。DEFAULT 仍是 `orion-mem16g`。

## 结果

| 项 | 值 |
|---|---|
| 重启前 boot_id | `7b3d596d-3424-4a78-854c-18fd5cc5833e` |
| 重启后 boot_id | `febc65c9-b79c-425c-9d2c-e959958e09da` |
| SSH | 约一次超时后重连，IP 仍 `10.5.229.37` |
| `mem=` | 仍 `16G`，MemTotal 仍 16435592 kB |
| nvpmodel | 仍 120W / mode 1 |
| Linger | yes；执行器自行 active（MainPID 2444） |
| 重启后首态 | `paused` / `host_reboot_with_open_work`；a 已收尾；b 留在 inbox |
| reboot-a | done，`implementation_error`（被内核重启打断，无 marker；探针记录，不是训练实现错误） |
| reboot-b | `resume` 后 completed，marker 已写 |
| failed_n | 4 → 5，新增的是本探针 a |

符合契约：运行中重启后核对 boot 身份；中断 attempt 保留；inbox 不自动吞；全局故障保持 paused，人工 resume 后才继续。

## 仍未做

- 硬整机 OOM（设计允许报告未确认，不主动撞死整机）
