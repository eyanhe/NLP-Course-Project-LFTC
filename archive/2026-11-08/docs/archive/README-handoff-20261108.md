# Text Classification Lab：二分类与多分类

最终主线：**常规任务的性能／延迟取舍、MultiNLI 复杂语义任务、类别不平衡下的衰退**。优先阅读 `report/final-report.md`（英文整合报告）、`report/final-synthesis-zh.md`（中文结论）、`report/imbalance-report.md`（新增实验详细报告）与 `report/final-presentation.pptx`（英文最终展示稿）。历史 V1/V2 和 MultiNLI 详细报告继续保留。

新增受控 AG News 不平衡实验：训练总量固定6,000，World/其他三类比例分别为25/25/25/25、70/10/10/10、85/5/5/5、94/2/2/2。验证与测试沿用原均衡样本，每类250条。比较五类模型的Macro-F1、少数类Recall/F1、MCC和逐类错误，同时计算相对自身均衡基线下降百分点。严重条件下预先指定SVM类别加权与DistilBERT加权交叉熵补救实验。深度模型各三个种子，超参数沿用均衡验证集选择；每个条件仅使用自己的训练文本学习特征。数据、权重和结果在 `data-imbalance/`、`models-imbalance/`、`results-imbalance/`。

```powershell
python imbalance.py all
python make_report_imbalance.py
python -m unittest -v test_project test_extension test_hard test_imbalance
python -m streamlit run app.py
```

英文GUI的实验页为 **Imbalance robustness**。主测试始终均衡；页面另有明确标注的部署先验加权准确率模拟，不能当作另一组真实测试数据。演示预测仍使用均衡训练的检查点。`run_project.ps1 -SkipDownload` 可续跑全部实验与报告，已完成阶段会复用固定结果。旧紧凑压缩包不含大体积权重与原始文本；本次完整交接包包含全部数据、权重和调参候选，见 HANDOFF.md。

MultiNLI 句间语义推断 benchmark：GUI 支持 Natural language inference、Premise／Hypothesis 双输入，以及 matched／mismatched 体裁评估。五种模型均有成对输入处理，另加 hypothesis-only SVM 诊断。现行难任务结果见 `report/hard-benchmark-report.md` 与 `report/hard-summary-zh.md`。

```powershell
python download_assets.py
python hard_benchmark.py all
python make_report_hard.py
python -m unittest -v test_project test_extension test_hard
```

该任务使用6,000训练对、999源验证对，以及各999对的同／跨体裁保留评估集；评估池来自官方validation，不冒充官方隐藏test成绩。通过premise分组避免不同假设共享前提导致跨集合泄漏。原V1/V2的结果与权重保持不变。数据、权重和结果分别保存在 `data-hard/`、`models-hard/`、`results-hard/`。

当前版本为 V2：**全英文 GUI、四个数据集、五种方法、多种评价指标**。在原有 IMDb／Rotten Tomatoes／Yelp 二分类上，新增 AG News 四分类（World、Sports、Business、Sci/Tech）；新增 Word2Vec 均值 + SVM 和 Word2Vec 初始化的 BiLSTM，保留 TF-IDF + LR／SVM 与 DistilBERT。

界面支持 Macro／Weighted／Micro Precision、Recall、F1，Accuracy、MCC、每类支持数和混淆矩阵。每种深度模型三个种子，验证集选参数。原始 V1 测试记录与权重保留，不混入不兼容的新闻／情感标签迁移。

当前完整结果优先阅读 `report/classification-report-v2.md`、`report/results-summary-v2-zh.md` 和 `report/classification-presentation-v2.pptx`。下方原有情感实验说明作为 V1 记录保留。

### V2 复现

```powershell
python -m pip install -r requirements-v2.txt --extra-index-url https://download.pytorch.org/whl/cu118
python download_assets.py
python pipeline.py all
python extension.py all
python make_report.py
python make_report_v2.py
python -m unittest -v test_project test_extension
python -m streamlit run app.py
```

新增代码：`extension.py`（新闻数据、训练、冻结、评估）、`models_extra.py`（词向量特征与 BiLSTM）、`make_report_v2.py`、`test_extension.py`。新增资产记录在 `data-v2/`、`models-v2/`、`results-v2/`。配置为 `config-v2.json`。V2 使用 AG News 新保留的测试样本；新增二分类方法属于查看旧测试结果后的扩展比较，已在报告披露。

深度模型输入长度分别按 word token 与 subword token 计算，不能把两个 256 当作相同单位。当前任务不需要实现课程举例中的所有算法。

## V1 情感实验说明

原始版本比较 TF-IDF + Logistic Regression、TF-IDF + Linear SVM 与微调 DistilBERT，在 IMDb、Rotten Tomatoes 和 Yelp 上的同集表现、直接迁移能力与计算成本。

## 快速使用

在包含本文的目录打开 PowerShell：

```powershell
python -m streamlit run app.py --server.address 127.0.0.1 --browser.gatherUsageStats false
```

如果当前 `python` 不是已配置环境，使用安装了依赖的 Python。当前机器验证的解释器为 `C:\Users\LUZIJ\anaconda3\python.exe`。首次加载来源模型需要数秒；随后应用缓存权重。推理不需要联网，不会上传输入评论。

## 从公开资产完整复现

推荐 Python 3.11，另建环境安装依赖，避免修改已有环境。实际版本锁定在 `requirements-lock.txt`，硬件与 Python 版本在 `environment.json`。这份锁文件记录本次运行环境，而非跨平台通用保证。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cu118
python -u download_assets.py
python -u pipeline.py all
python -u make_report.py
python -m unittest -v test_project
```

下载脚本只读取 Hugging Face 上的公开数据与基础模型，不登录、不发布。首次运行下载约600MB公开资产。`assets.lock.json` 固定仓库 commit、每个文件 SHA256 和下载地址。脚本 `run_project.ps1` 按相同次序执行；加 `-SkipDownload` 可复用现有资产。CUDA 11.8 的 PyTorch 轮子适用于本次已验证环境，其他平台应安装兼容本机的 PyTorch 并记录变更。

分阶段执行：

```powershell
python pipeline.py prepare
python pipeline.py baselines
python pipeline.py smoke
python pipeline.py transformers
python pipeline.py freeze
python pipeline.py evaluate
python pipeline.py benchmark
python make_report.py
```

已有阶段使用其元数据复用结果。`config.json` 或数据清单变化会触发过期结果检查。若要修改实验问题或数据量，请在新的项目副本中重新执行，不覆盖已查看测试集的确认性实验历史。`freeze` 必须在 `evaluate` 之前。

## 文件结构

- `config.json`：预定样本规模和超参数。
- `download_assets.py`：公开数据、模型下载和版本锁定。
- `pipeline.py`：数据准备、模型训练、冻结、评估、配对 bootstrap 与成本测量。
- `app.py`：本地 Streamlit 应用，固定 seed42 演示。
- `test_project.py`：标签／划分隔离、指标、重载与词表检查。
- `make_report.py`：根据实际结果生成报告、科学图表和真人讲稿。
- `assets/`：原始 Parquet 数据与基础模型，供离线训练使用。
- `data/`：抽样文本、清单、去重及截断审计。
- `models/`：6个传统来源模型、9个最终 Transformer 权重及完整元数据。
- `results/`：45个来源—目标—种子评估记录、逐条预测、协议冻结、配对区间、基准计时及错误复核队列。
- `report/`：实验报告、PNG/PDF 图表、讲稿、展示数据和提交材料草稿。
- `.cache/`：可复用调参 checkpoint、冒烟模型和工具缓存，不属于紧凑提交包。
- `run.log`：本次实验运行日志。

## 实验设计

每个数据集目标6,000训练、1,000验证、1,000测试，正负均衡，所有模型使用同一抽样。IMDb/Yelp 验证集从官方训练集抽取，Rotten Tomatoes 保留官方验证来源。测试样本全部来自官方测试 split。

规范化文本的哈希在三个数据源全局去重，完全重复文本优先留在测试集、其次官方验证集、最后训练集；冲突标签移除。TF-IDF 只在训练文本拟合。主指标为 Macro-F1，另含 Accuracy、每类 Precision/Recall/F1 与混淆矩阵。DistilBERT 按来源验证集选择学习率和 epoch，运行三个训练种子。传统固定模型不制造三次重复误差条。

三个语料不是三个完全独立领域：两个来自电影，另一个来自商家与服务。跨语料变化同时包含长度、文体、标注及难度差异。长文本的 DistilBERT 输入仅保留前256个 token，审计报告披露截断率。演示中的 SVM margin 不是概率；其他概率输出也没有额外做校准。

## 真实结果与解释

先看 `report/experiment-report.md` 和 `report/figures/within_dataset.png`。完整逐种子、逐来源、逐目标数字保留在 `results/metrics.csv` 和 `results/evaluation.json`。不能把三数据集的平均 F1 当作论文官方全量 benchmark 分数。

CPU 成本比较对所有模型使用同一组200条评论，4线程、batch1，每条重复5次，包含清理和特征转换、不含权重加载。训练在实际GPU上进行，硬件信息和每轮耗时记录在各模型元数据中。GPU allocated peak 不是总显存；本实现没有测量CPU内存峰值。

## 提交与人工完成事项

本项目提供可运行代码、模型、实际结果、报告、演示与讲解材料。课程最终要求还包括真人录制视频、真实贡献和成员签名、指定 GAI 声明表及个人互评。

`results/error_review.csv` 是最多60条真实模型错误／分歧样本的人工复核队列。成员应填写真实类别、解释和复核人，不得把待填字段说成已经完成。助手的案例解释若另有提供，应标为辅助草稿。

`report/team-reflection-draft.md` 和 `report/gai-disclosure-draft.md` 只是材料草稿。小组需核实和签名，GAI 内容需转入老师提供的正式模板。不得伪造贡献、签名、互评或真人录音。代码中没有作业提交功能。

## 原始资料

- [IMDb](https://huggingface.co/datasets/stanfordnlp/imdb)
- [Rotten Tomatoes](https://huggingface.co/datasets/cornell-movie-review-data/rotten_tomatoes)
- [Yelp Polarity](https://huggingface.co/datasets/fancyzhx/yelp_polarity)
- [DistilBERT 基础模型](https://huggingface.co/distilbert/distilbert-base-uncased)
- [DistilBERT 论文](https://arxiv.org/abs/1910.01108)
- [Hugging Face 分类教程](https://huggingface.co/docs/transformers/tasks/sequence_classification)

课程 PDF 和完整项目计划保留在项目目录之外。完整评分细则与最终提交格式以 syllabus/NTULearn 为准。
