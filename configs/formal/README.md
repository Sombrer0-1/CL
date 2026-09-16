# 既有正式协议配置

早期协议配置，保留作运行溯源和阶段生成器输入模板，不是当前待执行队列。`light24.py` 与 `pressure_protocol.py` 仍读取其中模板，不能整体删除或改路径。

第一阶段配置在 `../light24/`，第二阶段在 `../pressure_v2/`，均属历史阶段。第三阶段正式 YAML 在执行宿主 freeze 后写入 `../effectiveness_v3/`，见 [PLAN](../../PLAN.md)。

