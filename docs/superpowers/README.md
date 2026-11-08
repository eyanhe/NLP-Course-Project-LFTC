# 研究设计、执行与审查

当前研究主题为文本分类模型在不同数据条件下的能力边界与失败机制。研究已经完成基准训练、补充对照、优化网格和42个固定模型的独立Q评估；本目录保留设计与实施过程，当前结果从[研究主线](../../research-v3/RESEARCH.md)和[统一分析](../../research-v3/reports/overnight/results-analysis-v1-zh.md)阅读。

[交接前后的研究接续](../RESEARCH-CONTINUITY.md)说明原常规任务、MultiNLI与AG News长尾比较怎样引出当前规模、优化和机制实验；原始发现与后续追问分别标注。

## 设计与执行文档

- [完整研究设计](specs/2026-11-07-text-classification-research-design.md)：问题、实验矩阵、失败机制和统计口径。
- [原执行计划](plans/2026-11-07-text-classification-research-plan.md)、[机器协议](specs/2026-11-07-research-protocol.json)、[需求追踪](requirements-traceability.csv)。
- [设计独立审查](reviews/2026-11-07-research-design-review.md)、[来源核查](reviews/2026-11-07-source-verification.json)、[设计交付核验](reviews/2026-11-07-delivery-verification.json)。
- [补充实验1—4计划](plans/2026-11-07-supplement-1-4.md)与[实际补充结果](../../research-v3/reports/overnight/supplement-1-4/README.md)。
- [优化与正式Q结果](../../research-v3/reports/overnight/convergence-v1/README.md)、[S1/S3独立Q结果](../../research-v3/reports/overnight/supplement-q-v1/README.md)。
- [仓库整理计划](plans/2026-11-08-repository-organization.md)、[工作沿革与数量解释](../PROJECT-HISTORY.md)。

原2859项设计是规划范围，首批固定预算仅完成184项；后续S1、S3与优化新增48条，当前研究共232条。不能把原计划或准备阶段验收写成所有实验已完成。

## 历史准备阶段

- [训练前机器验收](../../research-v3/pretraining-verification.json)、[当时操作说明](../../research-v3/PRETRAINING.md)。
- [实施补充规则](specs/2026-11-07-preparation-clarifications.md)、[准备实现审查](reviews/2026-11-07-pretraining-code-review.md)、[准备完成记录](reviews/2026-11-07-pretraining-completion.md)。
- [本入口更新前的原文](archive/README-pretraining-20261108.md)。其中“训练未开始”等语句仅描述当时阶段。

设计、冻结文件和历史验收保留原文，当前进度使用结果目录的带时间记录；本页不充当实时调度查询。
