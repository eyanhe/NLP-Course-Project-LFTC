# v3研究源码

这是Python包，数据和结果位于相邻的[research-v3/](../research-v3/README.md)。

| 模块 | 职责 |
| --- | --- |
| `assets.py`、`preparation.py`、`inputs.py`、`sampling.py`、`matrix.py` | 资产、划分、公共输入、抽样与设计矩阵 |
| `features.py`、`models.py`、`train.py`、`batch.py` | 特征、学习器、固定预算训练与清单执行 |
| `diagnostics.py`、`supplemental.py` | 机制对照与S1—S4补充流程 |
| `convergence_training.py`、`convergence.py`、`budgeted_finish.py` | 后续学习率/预算/早停与检查点选择 |
| `scoped_evaluation.py` | 首批18个模型的独立Q范围 |
| `supplement_q.py` | S1/S3各自的独立Q范围 |
| `timed_inference.py`、`evaluate.py`、`metrics.py` | 计时、逐样本预测、指标 |
| `storage.py`、`execution.py` | 有界原子写入和运行租约 |

模块职责表是源码导航；具体可执行参数以各入口为准。已冻结的学习器、评估代码和模型请求有哈希绑定；组织文档时不改这些源码或已有模型身份。测试位于[tests/research/](../tests/research/)。

当前结果从[研究主线](../research-v3/RESEARCH.md)进入；分析与工作簿构建说明位于[开发支持](../development-support/research-v3/README.md)。
