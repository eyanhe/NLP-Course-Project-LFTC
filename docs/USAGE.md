# 使用说明

## 环境

在项目根目录使用 `.venv-research/Scripts/python.exe -m lftc`。以下示例写作 `python -m lftc`，执行时使用该环境或已经安装本项目的 Python。

基础包只依赖 PyYAML。模型和图表使用 `ml` / `train` 依赖，界面使用 `ui` 依赖；完整测试需要现有研究环境或这些依赖组。

## PPT 前端

运行 `.\apps\start.ps1` 或 `python -m lftc app --port 8501`，访问 `http://127.0.0.1:8501`。前端以项目 PPT 为依据，提供 Overview、Performance & cost、Robustness、Mechanisms、Live prediction、Methods & evidence、Conclusions 七个英文页面，图表和说明对应原 PPT 的研究问题与证据。

Live prediction 比较原研究保存的 TF-IDF + LR / SVM、Word2Vec + SVM / BiLSTM 和 DistilBERT，支持 IMDb、Rotten Tomatoes、Yelp、AG News、MultiNLI。选 MultiNLI 时输入 Premise 和 Hypothesis，再按 Compare models。首次加载检查点需要等待，后续使用缓存；CPU 耗时是现场演示值，正式成本对比见 Performance & cost。SVM 显示决策分数，其余显示未校准概率。

原研究结果和检查点从 `archive/2026-11-08/` 读取，不重新训练。侧栏提供源 PPT，Methods & evidence 提供原报告。新环境浏览页面安装 `.[ui]`，全部模型推理还需要 `.[train]` 及本地权重。[前端说明](../apps/README.md)包含各页内容和数据来源。

## 命令

```powershell
python -m lftc env
python -m lftc assets list
python -m lftc matrix --config configs/experiments/baselines.yaml
python -m lftc prepare --config configs/experiments/quickstart.yaml
python -m lftc run --config configs/experiments/quickstart.yaml --run-id demo
python -m lftc runs list
python -m lftc evaluate --run-id demo
python -m lftc predict --run-id demo --text "An excellent film."
python -m lftc diagnose --run-id demo
python -m lftc report --run-id demo
python -m lftc history
python -m lftc report --history
python -m lftc app --port 8501
```

`train` 是 `run` 的同义命令。`evaluate` 使用保存的测试预测重算指标，不重新训练。`diagnose` 输出完整标签空间的支持数、长度分组错误率和逐条错误；Python API 的 `perturb_examples` 可生成线索遮盖、截断、词序打乱或小写变体供机制实验使用。

配对模型预测需提供两个字段：

```powershell
python -m lftc predict --run-id mnli-demo --text "A person is running." --text-pair "Someone is moving."
```

对齐同一测试样本的两个实验可比较：

```powershell
python -m lftc compare --run-a model-a --run-b model-b --resamples 1000
```

该 Bootstrap 区间以固定的已训练模型为条件，抽样单位是配对测试行，不代表训练随机种子或跨数据集不确定性。多标签比较保留各自验证集选出的阈值。`holm_adjust` 提供多重比较校正接口。

## 实验配置

所有配置采用 YAML，`schema_version: 1`。

```yaml
schema_version: 1
name: review-baseline
dataset: rotten_tomatoes
model: tfidf_lr
seed: 42
sampling:
  mode: scale
  size: 200
training:
  max_features: 10000
  ngram_range: [1, 2]
  max_iter: 1000
```

可将 `dataset` / `model` / `seed` 改为 `datasets` / `models` / `seeds` 列表，并使用 `conditions` 列出多个抽样设置。`matrix` 只展示展开的配方，不启动训练；`run` 顺序执行展开的全部配方。

已有配置：

| 配置 | 用途 |
| --- | --- |
| `quickstart.yaml` | Rotten Tomatoes，随机抽取 200 条训练样本，保留完整验证/测试集 |
| `baselines.yaml` | 8 个数据集与 3 个传统模型的对比模板 |
| `neural.yaml` | BANKING77 / GoEmotions 上的 3 个神经模型与 3 个种子 |
| `data_conditions.yaml` | 全量、规模、少样本与长尾条件模板 |

后三个是研究运行模板，可能耗时较长，不会在打开项目或界面时自动执行。它们统一新工程接口，不自动等同于历史研究的固定实验协议。

抽样方式：`all`、`scale`（`size` 为样本数或比例）、`fraction`、`kshot`（`k` 为每类最少配额）、`long_tail`（`imbalance_ratio`）。多标签样本可以同时满足多个标签配额，因此最终各标签数量可能超过最低配额。抽样只操作训练集。

可选 `data.validation_fraction` 决定没有官方验证集时，从训练集划出的比例。`data.limit_per_split` 仅用于快速读取测试：它读取文件头部，类别有序的数据可能偏向单一类别，不用于代表性研究配置。

MNLI 使用公开有标签的 matched 验证集作验证、mismatched 验证集作保留测试；这不是官方隐藏测试集，报告和论文应据此标注。

## 神经网络

`model` 支持 `bilstm`、`distilbert`、`modernbert`。常用参数：`epochs`、`patience`、`batch_size`、`max_length`、`learning_rate`、`device`、`seed`。`device: auto` 优先使用可用 CUDA，否则 CPU。

BiLSTM 默认学习率为 0.001，Transformer 默认为 0.00002。验证损失用于早停和检查点选择；多标签共享阈值只从验证预测选择。Transformer 的权重与 tokenizer 从资产目录加载，全程离线。

模型保存后可通过 CLI 重新加载预测。PPT 前端使用原研究检查点，由新包中的兼容加载器读取，不执行归档源码；新流程的模型不会替代原研究演示模型。joblib/torch 模型文件应来自可信来源。

## 输出

```text
artifacts/
├── prepared/<dataset>-seed<seed>-val<fraction>/
├── runs/<run_id>/
│   ├── run.json
│   ├── metrics.json
│   ├── predictions.jsonl
│   └── model/
├── results/<run_id>/diagnostics.json
└── reports/<run_id>/
    ├── report.md
    ├── report-data.json
    ├── metrics.csv
    ├── per-label.csv
    ├── predictions.csv
    ├── errors.csv
    └── *.png
```

报告记录真实样本数、模型、配置、类别顺序和耗时。多标签 AP 只平均同时具有正负样本的标签，并显式保留掩码；F1 使用完整声明的类别空间。

`report --history` 将已有历史研究导出到 `artifacts/reports/historical-research/`，保留原始 428 / 232 / 226 / 42 的统计范围，并复制原报告、图册和表格用于阅读。它不训练或推理。

## 测试与集群

```powershell
python -m unittest discover -s tests -v
python ops/package_source.py --output artifacts/lftc-source.zip
```

测试使用临时小数据和小型神经网络，不重跑历史实验。[ops/README.md](../ops/README.md)说明 Linux、Slurm 和源码打包。Slurm 模板需按实际集群补充账户、分区和 GPU 资源；本机未提交集群任务。
