# S1/S3独立Q结果

两组均使用已完成的固定模型，未新增训练、Q选参或阈值调整。

| 组别 | 模型数 | Slurm作业 | 完成时间（2026-11-08 SGT） |
| --- | ---: | ---: | --- |
| S3固定表示×标注支持×类别贡献 | 8 | 278262 | 06:56 |
| S1学习率×预算 | 16 | 278263 | 06:58 |

- [S3结果](cluster-snapshot/reports/overnight/supplement-q-v1/S3/test-results.json)、[S1结果](cluster-snapshot/reports/overnight/supplement-q-v1/S1/test-results.json)。
- 每组同目录保存各自冻结、Q访问记录、进度与`predictions/`；逐样本文件和计时已下载校验。
- [执行记录](execution-record.json)与[同步完成记录](sync-status.json)。
- 主要解释见[研究主线](../../../RESEARCH.md)及[统一分析](../results-analysis-v1-zh.md)。

`sync-live.py`是本机只读同步入口；训练和测试不在登录节点执行。
