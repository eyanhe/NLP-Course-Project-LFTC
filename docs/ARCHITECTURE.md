# 工程架构

## 代码职责

| 层 | 包 | 职责 |
| --- | --- | --- |
| 0 | `domain` | 任务、资产、样本、准备后数据集和异常类型 |
| 1 | `governance` | 公共路径边界处理 |
| 1 | `config` | 路径、资产注册表和实验网格 |
| 2 | `data` | Parquet 适配、划分、保存与训练集抽样 |
| 2 | `features` | 训练集拟合的 TF-IDF / Word2Vec 与文本对表示 |
| 2 | `models` | 全类别传统分类器与 BiLSTM 结构 |
| 2 | `evaluation` | 指标、阈值、配对统计、错误诊断和文本干预 |
| 3 | `training` | 训练、早停、分批推理、可重载检查点和原研究模型加载 |
| 3 | `reporting` | 实验报告、统计图、表格导出和 PPT 原研究证据读取 |
| 4 | `experiments` | 组合下层能力，完成 prepare/run/predict/evaluate/diagnose/report |
| 4 | `cli` | 唯一命令行入口 |
| 应用 | `apps` | 基于原 PPT 的 Streamlit 交互展示 |
| 工具 | `ops` | 运行包装、集群模板与源码打包 |

高层依赖低层，底层不导入训练工作流或界面。配置层不会加载训练模型；基础 CLI 和历史研究元数据不要求加载 PyTorch。新代码不导入旧包，历史资料以 JSON、Markdown、CSV 和成品文件读取。

## 数据流

新实验：`configs/assets.yaml` → 本地 Parquet → `PreparedDataset` → 训练集条件抽样 → 训练集拟合特征与模型 → 验证集早停/阈值 → 保留测试预测 → 指标和报告。

PPT 展示：原归档结果 → `reporting.presentation` → 七个研究页面；原五模型检查点 → `training.benchmarks` / `models.benchmark_compat` → 同一输入的实时预测对比。兼容类只负责读取已有模型，不调用旧训练流程。

文本对保留两个独立字段。单标签分数和多标签分数始终包含完整声明的类别列；传统模型遇到训练集中未出现的类别或恒定标签时提供明确的常量分数。空词汇特征有可预测的回退行为。

## 历史与当前

- `archive/2026-11-08/` 保存旧材料，数据和模型供查阅与演示，不提供归档 Python 的兼容导入。
- `artifacts/assets/` 保存可复用原始数据和底座模型。
- `artifacts/runs/`、`results/`、`reports/` 保存新流程产物。
- 前端读取 PPT 的原研究证据，新流程产物通过 CLI 读取与导出，不将示例运行计入原研究数量。

## 扩展

新数据集在 `configs/assets.yaml` 注册；非标准字段在 `data` 增加适配。新模型通过 `training.train_model` 接入统一 Predictor 分数与保存/加载接口。新统计方法放在 `evaluation`，新的呈现方式放在 `reporting` 或 `apps`，不在 CLI 内复制算法。

[使用说明](USAGE.md) · [历史资料](archive/2026-11-08.md)
