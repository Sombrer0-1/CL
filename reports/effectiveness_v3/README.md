# effectiveness_v3 报告

| 材料 | 迁移 | 说明 |
|---|---|---|
| `IMPLEMENTATION_AUDIT.md` | 代码补丁记录 | B01–B06；5090 当时的 GPU 测量已归档到 `docs/environments/linux_rtx5090/` |
| `design_review.json` | 设计自审 | 名额与文档交叉检查，不是冻结协议 |
| `verify_failure_peak.py` | 脚本 | 可在本机 Thor 重跑；勿把归档 JSON 当当前证据 |
| `<revision>/` | 不迁移 | 开发校准与正式报告；5090 的 r1 已按 A26 删除 |

有效性结论只来自新宿主冻结后的正式矩阵，不来自本目录的审查文件。
