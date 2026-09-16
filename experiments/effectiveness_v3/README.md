# effectiveness_v3 实验对象

| 路径 | 是否随仓库迁移 | 说明 |
|---|---|---|
| `design.yaml` | 是 | 不可执行设计：153 名额、场景、校准规则。无冻结预算 |
| `revisions/<revision>/` | 否 | 当前宿主的 probe、校准、冻结协议和矩阵；换机后重建 |

不要把某一台机器上的 `frozen_protocol.json` 拷到新机器继续 `run`。
