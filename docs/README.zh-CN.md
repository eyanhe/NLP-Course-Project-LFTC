# LFTC 文本分类研究工程

[English](../README.md) | **简体中文**

重构后的工程统一使用 `src/lftc/` 运行新实验、评估报告和 PPT 交互前端；历史代码与研究结果保留在 `archive/2026-11-08/` 供查阅。

[GitHub 仓库](https://github.com/eyanhe/NLP-Course-Project-LFTC)保存源码、历史报告、图表、PPT、Excel 和结果数据；[Hugging Face 资产仓库](https://huggingface.co/datasets/Eyanhe/LFTC_Dataset_And_Checkpoints/tree/5fd879463414b2059ae05b4e188da58e706456dd)保存原始数据、特征与模型权重。GitHub 保留原有提交历史，历史材料按本次重构的位置归档。

## 直接使用

在 `sentiment-project/` 中执行：

```powershell
# 打开基于 project ppt 原稿的交互式研究展示
.\.venv-research\Scripts\python.exe -m lftc app
```

直接运行 `.\apps\start.ps1`，脚本会优先使用本机 `.venv-research`。

前端沿用 PPT 的英文内容、深蓝/米白配色和三条研究问题，包含总览、性能与成本、鲁棒性、机制、实时预测、方法与证据、结论七个页面。图表读取原研究结果；实时预测比较原来的五个模型，支持三个评论数据集、AG News 和 MultiNLI 文本对。侧栏可下载源 PPT。[前端说明](../apps/README.md)列出了页面与模型来源。

新实验仍通过以下 CLI 运行。已有 `quickstart-local` 是 200 条训练样本的流程示例，不参与前端的原研究展示或模型预测。

```powershell
# 新建一个小规模实验，输出模型、逐样本预测和指标
.\.venv-research\Scripts\python.exe -m lftc run --config configs/experiments/quickstart.yaml --run-id my-first-run

# 用保存的模型预测并生成报告
.\.venv-research\Scripts\python.exe -m lftc predict --run-id my-first-run --text "A thoughtful and enjoyable film."
.\.venv-research\Scripts\python.exe -m lftc report --run-id my-first-run
```

## 阅读入口

- [使用说明](USAGE.md)：实验配置、完整命令、模型与输出文件。
- [工程架构](ARCHITECTURE.md)：分层职责与数据流。
- [历史资料入口](archive/2026-11-08.md)：原研究报告、模型、数据和旧代码。
- [资产注册表](../configs/assets.yaml)：8 套数据集、2 套底座模型的来源和目录。
- [操作工具](../ops/README.md)：集群模板、运行包装器和源码打包。

## 已整理的能力

- **数据与实验**：8 个本地 Parquet 数据集；单标签、多标签和文本对分类；训练集抽样、规模、少样本与长尾条件；数据集/模型/随机种子/条件组合。
- **模型**：TF-IDF + 逻辑回归、TF-IDF + SVM、Word2Vec + SVM、BiLSTM、DistilBERT、ModernBERT。特征只从训练数据学习，Transformer 使用本地模型。
- **评估与分析**：完整类别空间的 F1、AP、混淆矩阵、验证集阈值选择、配对 Bootstrap、McNemar、Holm 校正、文本长度与错误案例诊断。
- **报告**：新实验的 Markdown、JSON、CSV 和统计图；历史研究的数据与原始成品导出。
- **PPT 交互前端**：按原演示的三条研究问题展示真实结果、成本和机制案例，比较五个原研究模型的预测，下载源 PPT 与报告。
- **工程工具**：Windows/Linux 运行脚本、Slurm 模板、排除本地大文件与归档的源码 ZIP 打包器。

## 目录

```text
sentiment-project/
├── pyproject.toml          包声明与可选依赖
├── src/lftc/               唯一活跃 Python 核心包
├── configs/                路径、资产与实验配置
├── apps/                   PPT 交互式 Streamlit 展示及演示素材
├── ops/                    运行、集群和打包工具
├── tests/                  新工程测试
├── docs/                   使用说明、架构和历史资料导航
├── artifacts/              本地资产、新模型、结果和报告
└── archive/2026-11-08/     历史代码、已有实验产物和文档
```

`.venv*`、`.tools/`、`.cache/`、`.superpowers/` 保留原位，由编辑器隐藏；归档与产物排除出默认搜索。项目旁的 PPT、录音文稿和传输包归档保持原位。

## 环境与路径

本机现有 `.venv-research` 已登记 `src/` 路径。新环境可按需安装：

```powershell
python -m pip install -e .           # 基础 CLI 和配置
python -m pip install -e ".[ml]"     # 传统模型、评估和报告
python -m pip install -e ".[train]"  # 神经网络
python -m pip install -e ".[ui]"     # 前端
```

CUDA 环境按集群环境选择匹配的 PyTorch 安装。原始数据和底座模型保留本地，不随源码包分发。

Hugging Face 的固定版本沿用重构前路径。恢复所需资产时，先移除其外层 `artifacts/`，再按下表放入当前工程；不必下载全部资产：

| 资产仓库中的路径 | 当前工程位置 |
| --- | --- |
| `artifacts/assets/distilbert/` | `artifacts/assets/models/distilbert/` |
| `artifacts/research-v3/assets/modernbert/` | `artifacts/assets/models/modernbert/` |
| `artifacts/assets/<数据集>/` | `artifacts/assets/datasets/<数据集>/` |
| `artifacts/research-v3/assets/<数据集>/` | `artifacts/assets/datasets/<数据集>/` |
| `artifacts/models*/`、`artifacts/data*/`、`artifacts/research-v3/` 中其余历史资产 | `archive/2026-11-08/` 下对应的原相对路径 |

例如，原 `artifacts/models/imdb/linear_svm/model.joblib` 应恢复为 `archive/2026-11-08/models/imdb/linear_svm/model.joblib`。只浏览研究图表无需下载权重；实时预测需要对应的历史检查点，新实验需要相应数据集和底座模型。历史发布清单保存在 [归档发布说明](../archive/2026-11-08/docs/publishing/README.md)。

`configs/paths.yaml` 中 `artifacts_dir` 相对工程根目录，其余产物目录相对 `artifacts_dir`。支持 `LFTC_ROOT`、`LFTC_CONFIG_DIR`、`LFTC_ARTIFACTS_DIR` 环境变量覆盖，以及全局 `--root` 参数。

## 当前状态

四个子项目的本地实现与整合均已完成。前端按原 PPT 组织研究展示，新实验保留 CLI 入口，不将示例成绩混入原来的实验数量。原系统不再维护运行入口，一次性调试与绘图脚本留在归档。

发布前完整测试：`python -m unittest discover -s tests -v`，53 项通过。未重跑历史实验、未新建资产锁或执行大文件哈希核验。
