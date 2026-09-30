# G1 注销存活（linger / terminate-user）

日期：2026-09-22。`mem=16G`。不是 G3 冻结，不是硬整机 OOM。

## 做了什么

- Linger 已是 yes；用户服务 `orion-fullmem-v4-executor.service` enabled。
- 提交 `g1-linger-a`（dummy 90s）与 `g1-linger-b`（dummy 5s）。
- a 已在跑时执行 `loginctl terminate-user zhuzetong`。
- 未换档、未改 nvpmodel、未重启内核。boot_id 仍是 `7b3d596d-3424-4a78-854c-18fd5cc5833e`。

## 结果

| 项 | 值 |
|---|---|
| SSH 重连 | 成功，IP 仍 `10.5.229.37`，`mem=16G` |
| Linger | 仍 yes |
| 执行器 | 重新起来（MainPID 2327 → 49338），无需人工装服务 |
| linger-a | 中断，attempt 保留在 done，`implementation_error`（dummy 被杀掉，未写 marker） |
| linger-b | 暂停期间留在 inbox；`resume` 后 completed，marker 已写 |
| failed_n | 3 → 4，新增的是本探针 a，不是训练矩阵失败 |

这符合现有执行器契约：全局故障保持 paused，不自动吞 inbox；中断的 attempt 保留。注销后用户服务因 linger 自行回来。

分类器把非 137 的中断记成 `implementation_error`，这里只作探针记录，不把它写成训练实现错误。

## 仍未做

- 硬整机 OOM
