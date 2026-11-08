# 后续优化与首批Q

本目录保存BANKING77 1k、GoEmotions 6k的学习率/预算/早停与SVM C网格证据。该轮26条记录含24次新训练和2条旧SVM复用。

- [首批Q结果](cluster-snapshot/reports/overnight/convergence-v1/test-results.json)：14个主比较模型加4个历史预算参考，18个模型于05:57 SGT完成。
- [当前累计清单](experiment-inventory-current.json)：合并S1/S3后42个Q模型，当前计数入口。
- [18-Q历史清单](experiment-inventory.json)：06:06 SGT快照；后续分析引用其训练记录，因此不覆盖。
- [计时原始汇总](training-inference-timings.json)：仅该轮26条加4个历史参考，不是全仓库训练计时总表。
- [累计模型Excel](../../../../outputs/experiment-inventory-20261108/累计训练验证推理清单.xlsx)、[该轮计时Excel](../../../../outputs/training-inference-timings-20261108/训练与推理耗时记录.xlsx)。

`cluster-snapshot/`保存下载的原始证据，`sync-live.py`只读同步，`audit-experiments.py`生成当前累计清单。完整分析见[统一报告](../results-analysis-v1-zh.md)。
