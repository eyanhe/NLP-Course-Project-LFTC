# 项目交接文档

交接日期：2026-11-06。项目：Comparing traditional NLP and deep learning methods for text classification。

## 1. 接手后先做什么

1. 解压 **text-classification-full-handoff.zip**，进入其中的 `sentiment-project` 目录。
2. 阅读本文件、`report/final-synthesis-zh.md` 和 `report/final-report.md`。
3. 安装下述 Python 依赖，再启动英文 GUI。包内已经有训练好的权重、基础模型、原始数据和处理后数据，**演示不需要重新训练或下载模型**。
4. 汇报优先使用 `report/final-presentation.pptx`。此前各版本 PPT 是历史／补充资料。

这次交接的是完整版，包含最终权重、调参候选权重和开发验证资料；此前约10 MB的 `sentiment-project-submission.zip` 是不带权重的紧凑版，不应把它当作本次交接包。

## 2. 已完成的内容

- 常规文本分类：IMDb、Rotten Tomatoes、Yelp 二分类，以及 AG News 四分类。
- 复杂任务：MultiNLI 句对三分类，含 matched／mismatched 两组保留评估和 hypothesis-only 诊断。
- 不平衡实验：AG News 四种训练比例、固定训练总量6,000、固定均衡验证／测试集；严重条件下增加类别加权。
- 五类方法：TF-IDF + LR、TF-IDF + SVM、Word2Vec 均值 + SVM、Word2Vec 初始化的 BiLSTM、DistilBERT。
- Accuracy、Macro／Micro／Weighted Precision／Recall／F1、MCC、逐类指标、混淆矩阵、训练种子波动、部分预先指定比较的 bootstrap 区间。
- 英文 Streamlit GUI、实验报告、图表、12页最终英文 PPT、真人讲解安排。
- 原环境中22项测试已通过。英文GUI三类任务、四种不平衡比例及少数类曲线已检查。打包时另外进行文件清单、SHA256和压缩包完整性检查。

## 3. 启动演示：推荐的 Windows CPU 方式

原环境是 Windows、Python 3.11.4、PyTorch 2.2.1+cu118，训练使用 RTX 4090 Laptop GPU。**GUI推理在CPU上运行，无需独立显卡**。建议使用64位 Python 3.11；不要直接复制别人的 Anaconda 目录。

在 `sentiment-project` 目录打开 PowerShell，依次执行：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.2.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements-share.txt
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501 --browser.gatherUsageStats false
```

在浏览器打开 `http://127.0.0.1:8501/`。关闭终端或按 Ctrl+C 停止服务。使用虚拟环境解释器的完整相对路径，可以避免 PowerShell 激活策略和多个 Python 环境造成的混淆。

`requirements-share.txt` 沿用本项目的依赖版本，PyTorch版本不绑定CUDA后缀；上面的第一步安装CPU轮子。依赖安装需要联网；完成后，包内的模型与数据支持离线演示。本次没有在组员的新电脑上实际安装依赖，其他操作系统／硬件需要先验证兼容性。

已有依赖环境也可直接运行：

```powershell
python -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501 --browser.gatherUsageStats false
```

`start_app.cmd` 使用当前 `python`。新建 `.venv` 的组员应优先使用上面的显式路径命令。

## 4. GUI怎么演示

- 顶部任务：Binary sentiment／Multiclass news／Natural language inference。
- Try a prediction：输入英文文本；NLI需要分别填写 Premise 和 Hypothesis，比较五种模型。演示检查点固定为均衡训练模型，深度模型使用seed42。
- Evaluation & comparison：查看多种评价指标、逐类结果、混淆矩阵和CPU延迟。
- Imbalance robustness：切换25/25/25/25、70/10/10/10、85/5/5/5、94/2/2/2；查看绝对分数、下降百分点、少数类曲线和类别加权结果。
- Methods & limitations：课程方法对应关系、选择规则和限制。

窄窗口下结果表可以横向滚动，也可使用表格右上角的全屏／CSV下载。SVM margin不是概率；深度模型概率未经校准。输入样例用于演示模型行为，不能替代统计评估。

## 5. 最终研究主线与关键结果

### 常规任务：性能与推理成本

IMDb、Yelp上，TF-IDF + LR与DistilBERT的Macro-F1分别相差约0.54和0.90个百分点；RT、AG News差距更大。因此应说“部分常规任务性能接近，差距依赖数据集”。相同评论输入、四CPU线程、batch=1的已有测试中，IMDb模型的SVM median约1.59 ms，DistilBERT约60.48 ms。延迟与硬件、输入长度和实现有关。

### MultiNLI：复杂语义任务

Matched：配对SVM Macro-F1 0.4990，DistilBERT 0.6580；mismatched分别0.5111和0.6721，约16个百分点差距。深度数值是三个种子的均值。评估使用官方validation池的保留样本，不是隐藏test榜单。传统方法也读取完整句对交互特征。DistilBERT使用外部预训练，不能把收益完全归于架构。

### 不平衡：各自相对均衡基线的衰退

训练总量始终6,000，World为多数类；验证和测试各1,000，每类250。严重条件为World 5,640、其他三类各120。

- TF-IDF + LR：Macro-F1 0.1082，下降77.17个百分点；少数类Recall 0.0053。
- TF-IDF + SVM：Macro-F1 0.1000，下降77.76个百分点；少数类Recall 0。
- Word2Vec + SVM：Macro-F1 0.2856，下降56.41个百分点；少数类Recall 0.1373。
- BiLSTM：Macro-F1 0.4424，下降42.54个百分点；少数类Recall 0.2991。
- DistilBERT：Macro-F1 0.7910，下降12.86个百分点；少数类Recall 0.7200。

预先指定的类别加权使严重条件SVM Macro-F1回升至0.4000、DistilBERT回升至0.8193；少数类Recall分别0.2413和0.7684。这里支持本次设置下预训练Transformer更稳健，不能推广为所有深度方法都更稳健。

“94%多数类时只预测多数类就有94%准确率”是另一个部署先验加权数学示例，**不是主测试成绩**。主测试始终均衡，此规则Accuracy为25%、Macro-F1为0.10。

## 6. 文件导航

- `app.py`：英文GUI入口。
- `pipeline.py`：原始情感实验，数据审计、TF-IDF模型、DistilBERT、迁移、延迟。
- `extension.py`、`models_extra.py`：新闻四分类、Word2Vec、BiLSTM及多指标扩展。
- `hard_benchmark.py`、`pair_models.py`：MultiNLI数据分组与句对模型。
- `imbalance.py`：受控不平衡抽样、冻结、训练和评估。
- `config*.json`：不同实验固定配置；`assets.lock.json`：公开资产版本、下载地址和校验值。
- `assets/`：五个数据集原始parquet文件及通用DistilBERT基础模型。
- `data/`、`data-v2/`、`data-hard/`、`data-imbalance/`：实际使用的JSONL数据、样本清单和审计结果。
- `models/`、`models-v2/`、`models-hard/`、`models-imbalance/`：最终模型、tokenizer／词表、Word2Vec特征和元数据。模型目录是相对项目自动定位的，不依赖原电脑用户名。
- `.cache/`：调参候选检查点、smoke模型及运行缓存。本次完整交接保留；GUI主要使用上述最终模型目录。
- `results*/`：逐样本预测、指标CSV／JSON、冻结协议、延迟、区间和复核队列。不平衡40条评估记录包含9个复用均衡参考、27个新不加权模型、4个加权模型；有40,000条预测。
- `report/final-report.md`：最终英文整合报告；`final-synthesis-zh.md`：中文结论；`imbalance-report.md`：新增实验细节。
- `report/final-presentation.pptx`：最终12页英文主讲稿；`recording-plan-final.md`：约9分30秒真人讲解安排。
- `report/submission-checklist.md`：交付状态；`team-reflection-draft.md`、`gai-disclosure-draft.md`：需要真实成员核实的草稿。
- `reference/`：原始任务PDF、最初计划。计划属于历史设计，最终实验以最终报告和配置为准。
- `development-support/`：PPT构建源文件、渲染、验证日志等开发资料。PPT构建脚本依赖原作者使用的Artifact Tool运行时并包含本地路径，不能直接当作通用启动脚本；现成PPT可以直接用PowerPoint／WPS编辑。演示和实验Python入口不依赖该运行时。
- `requirements-lock.txt`、`requirements-v2.txt`、`environment.json`：原环境记录；`requirements-share.txt`：交接安装入口。
- `package_project.py`：旧紧凑包；`package_full.py`：本次完整包；`verify_full_package.py`：解压后逐文件SHA256校验。

## 7. 校验与复现

解压后的 `sentiment-project` 内运行下述命令，核对完整包清单中的每个文件；它只使用Python标准库，读完约11.5 GB内容需要一些时间：

```powershell
python verify_full_package.py
```

这个检查应在修改代码／重新运行实验之前执行。清单位于 `FULL-PACKAGE-MANIFEST.json`；它不对自身求哈希。若只改了代码，校验失败会指出改动文件，不代表权重也损坏。

依赖安装好后检查项目功能：

```powershell
.\.venv\Scripts\python.exe -m unittest -v test_project test_extension test_hard test_imbalance
```

全部实验续跑／重新生成报告：

```powershell
.\run_project.ps1 -SkipDownload
```

该脚本使用当前 `python`；先激活自己的环境，或按脚本中的顺序用 `.venv` 解释器逐条执行。完整包已有资产，首次接手不需要运行下载。已有结果会进行配置／样本哈希检查并复用；这不是强制重新训练。真正修改配置后，请在新项目副本中使用新的结果目录，避免覆盖已冻结实验。CPU可演示；完整训练建议使用兼容的NVIDIA GPU和对应PyTorch，训练耗时不能照搬原机器。

若只生成报告，可运行各 `make_report*.py`。改变数据／配置后不要用旧结果冒充重跑；也不要根据测试集成绩调整超参数。训练SD和测试bootstrap区间回答不同问题。

## 8. 常见问题

- `ModuleNotFoundError`：检查启动服务和安装依赖是否使用同一个Python解释器。
- 权重／数据找不到：确认进入完整解压后的项目根目录，并保留原目录结构；不要只复制 `app.py`。
- 8501端口被占用：改成 `--server.port 8502`，再打开对应地址。
- 页面仍显示旧内容：刷新浏览器；出现“Source file changed”时点击Rerun。
- CPU预测首次较慢：首次会载入模型；后续使用缓存。GUI延迟不是完整实验计时，正式数值看benchmark结果。
- 校验失败：先看具体文件；重新传输／解压损坏文件，或区分组员已经做出的有意修改。
- 旧报告与新报告措辞不同：优先阅读final系列；旧文件保留各阶段真实结果。README中“紧凑包不含权重”的说明仅适用于旧紧凑包。

## 9. 仍需小组完成

- 填写真实姓名、实际贡献、个人反思，由相关成员核实和签署；本次没有编造这些信息。
- 人工复核错误案例队列，记录真实复核人；助手草稿不能声称由两人独立复核。
- 按最终PPT和讲解安排录制真人视频，控制在10分钟内。
- 把真实GAI使用情况转入老师指定模板，核实课程rubric、提交细则和截止时间。
- 由小组自行上传、确认视频可播放，并完成要求的个人互评。

建议先由一位组员完成解压、依赖安装、完整性校验和GUI演示，再安排报告／PPT审阅、案例复核和录制。具体人员由小组自行决定。
