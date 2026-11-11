# 当前研究入口

研究主题：**文本分类模型在不同数据条件下的能力边界与失败机制。** `research-v3/`保存数据、模型工件和结果；相邻的[research_v3/](../research_v3/README.md)是Python源码包。

## 建议阅读顺序

1. [研究主线与结论](RESEARCH.md)：六组证据对两个核心问题分别意味着什么。
2. [统一分析报告](reports/overnight/results-analysis-v1-zh.md)：42个Q模型、受控机制对照、具体错误案例及结论范围。
3. [12页完整研究图册](reports/overnight/complete-atlas-v1/完整研究图册.pdf)先说明交接前后的研究接续，后接11张结果图。[逐图解释与PNG](reports/overnight/complete-atlas-v1/README.md)与[研究逻辑链](reports/overnight/research-logic-v1/当前研究问题与实验逻辑链.pdf)可单独阅读。原[三页重点Q图](reports/overnight/analysis-v2/Q测试与机制对照分析.pdf)保留为简版。
4. [研究对照Excel](../outputs/research-analysis-20261108/研究问题定量分析.xlsx)与[累计模型Excel](../outputs/experiment-inventory-20261108/累计训练验证推理清单.xlsx)。

## 当前范围

| 批次 | 完成模型记录 | 有V指标 | 完成Q |
| --- | ---: | ---: | ---: |
| 接手后v3首批固定预算 | 184 | 178 | 4 |
| S1学习率×预算 | 16 | 16 | 16 |
| S3支持×类别贡献 | 8 | 8 | 8 |
| 后续优化与C网格新增 | 24 | 24 | 14 |
| 当前研究合计 | 232 | 226 | 42 |

184指接手后v3第一批，不是原交接已有模型；很多为固定短预算探索。另181条早期选参额外记录及15条工程检查，使全部v3完成记录为428，2条中断不计完成。模型记录数量不等于模型种类或独立结论数量。[完整来源和去重说明](../docs/PROJECT-HISTORY.md)。

42个Q结果来自三套已冻结范围：优化/历史参考18、S1 16、S3 8；最新一组于2026-11-08 06:58 SGT完成。6条strict_no_val分支按设计没有V指标。原始2,859项主清单仅完成其中184项，补充实验另外登记。

## 目录用途

| 目录 | 用途 |
| --- | --- |
| `assets/`、`data/`、`inputs/` | 固定公开资产、划分数据及统一可见输入 |
| `matrix/` | 实验设计、条件与训练ID |
| `runs/`、`features/` | 本机模型状态/权重、特征工件；集群模型权重不保证已全部下载 |
| [reports/](reports/README.md) | 当前分析、图表、预测与证据 |
| `training/` | 授权、配方、集群脚本、模型清单与执行记录 |
| `audit/` | 审核协议和标注结构 |
| `history/` | 早期验收与历史状态 |

原始预测、冻结清单和模型路径保留。目录内的训练前验收/旧状态JSON是当时快照；当前进展看上表和[当前清单](reports/overnight/convergence-v1/experiment-inventory-current.json)，不要从历史文件的pending字段推断现在仍在运行。

[研究设计与执行记录](../../../docs/superpowers/README.md) · [原交接文档](../legacy-root/HANDOFF.md)
