# v3文档与工件构建支持

当前交付入口在[仓库README](../../README.md)和[研究主线](../../research-v3/RESEARCH.md)。本目录放工作簿构建器与整理核验工具，不放原始模型/数据。

| 脚本 | 产物 |
| --- | --- |
| `build-analysis.mjs` | `outputs/research-analysis-20261108/研究问题定量分析.xlsx` |
| `build-inventory.mjs` | `outputs/experiment-inventory-20261108/累计训练验证推理清单.xlsx`；读取当前42-Q清单 |
| `build-timings.mjs` | `outputs/training-inference-timings-20261108/训练与推理耗时记录.xlsx`；仅后续优化轮 |
| `check-organization.py` | 当前入口链接、计数与整理前后受保护文件SHA核验 |

构建器从项目根目录运行，依赖Codex本机的Artifact Tool运行时。`node_modules`为本机运行时链接，不属于需复制的研究数据。原`outputs/training-inference-timings-20261108/build-*.mjs`保留兼容转发，避免旧同步入口失效。

数值重算和报告图构建仍在[analysis-v2/](../../research-v3/reports/overnight/analysis-v2/README.md)，以维持已有研究复现路径。原始数据、模型、冻结协议和逐样本证据位置不动。

完整图册的独立生成器和逐图数据在[complete-atlas-v1/](../../research-v3/reports/overnight/complete-atlas-v1/README.md)：共12页，包括1页交接前后接续与11张结果图。`build-atlas.py`校验原交接归属和现有来源SHA后绘图，不运行模型；PDF渲染预览位于其`support/`。

整理前后保护清单、移动记录和核验结果保存在项目`.superpowers/repo-organization-20261108/`。旧README原文保存在[历史入口](../../docs/PROJECT-HISTORY.md)指向的归档中。
