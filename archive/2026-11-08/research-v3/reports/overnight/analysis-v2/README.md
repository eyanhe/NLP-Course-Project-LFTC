# 当前统一分析

这批分析只读取已保存的预测与状态，不启动训练或模型推理。

需要完整图表时，阅读[12页研究图册](../complete-atlas-v1/完整研究图册.pdf)及[逐图解释](../complete-atlas-v1/README.md)：先看交接前后研究接续，再看11张结果图。本目录保留核心Q三页简版与统一数值分析。

- [研究主线与结论](../../../RESEARCH.md)：六组证据的研究含义。
- [统一分析报告](../results-analysis-v1-zh.md)：现象、假设、实验和解释。
- [三页Q图](Q测试与机制对照分析.pdf)：优化后比较、S1配方敏感性、S3机制。
- [Excel对照](../../../../outputs/research-analysis-20261108/研究问题定量分析.xlsx)：42个Q模型、44组差值及V/D证据。
- [analysis-data.json](analysis-data.json)：从42份预测重算的指标、差值、案例和198个来源哈希。

`analyse.py`重算和校验预测；`publish.py`生成当前报告与三页图。旧报告保存在[Q完成前原文](results-analysis-before-42Q.md)。`support/`存PDF渲染预览，不是另一套研究结果。

历史输入`convergence-v1/experiment-inventory.json`在分析时为18-Q快照，其训练记录用于M2汇总。当前模型覆盖另看[42-Q累计清单](../convergence-v1/experiment-inventory-current.json)，不要改写历史输入来更新状态。
