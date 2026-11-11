# PPT 交互式研究展示

`streamlit_app.py` 根据 `project ppt/Traditional NLP, Deep Learning and Pretrained Transformers (v2).pptx` 重做，沿用原稿的英文叙事、深蓝/米白底色和橙、蓝、粉模型配色。原 PPT 保持原位，下载副本放在 `assets/research-presentation.pptx`。

在项目根目录启动：

```powershell
.\apps\start.ps1
# 或指定 Python 和端口
.\apps\start.ps1 -Python ./.venv-research/Scripts/python.exe -Port 8501
# CLI 入口
.\.venv-research\Scripts\python.exe -m lftc app --port 8501
```

打开 `http://127.0.0.1:8501`。Linux/macOS 可运行 `bash apps/start.sh --server.port 8501`。新环境安装 `.[ui]` 可浏览展示，运行全部五种模型的预测还需要 `.[train]` 和本地检查点。

| 页面 | 内容 | PPT |
| --- | --- | --- |
| Overview | “When does pretraining pay off?”、三条 RQ、不同条件下的模型差距 | 1–3、8 |
| Performance & cost | 五模型、多指标、种子差异、条件置信区间、CPU 成本与混淆矩阵 | 8–9 |
| Robustness | 类别失衡、权重补偿、多数类规则、MultiNLI、跨来源迁移 | 10–11 |
| Mechanisms | 预训练与适配、标签支持、词语干预、优化预算和真实案例 | 12、附录 |
| Live prediction | 同一输入比较五个已训练模型；MultiNLI 提供前提和假设输入框 | 7 的交互演示 |
| Methods & evidence | 数据、协议、三类方法、六项指标、原报告下载与资料入口 | 4–7、14–15 |
| Conclusions | 按任务选择模型、证据局限和待验证的级联方案 | 13–14 |

`lftc.reporting.presentation` 读取 `archive/2026-11-08/` 的原始结果数据；`lftc.training.benchmarks` 读取其中 `models/`、`models-v2/`、`models-hard/` 的原检查点。两者不导入归档 Python，不训练或联网下载。神经模型演示使用 seed 42，结果图表按各自原协议汇总；SVM 决策分数与模型概率分别标示。

五个模型为 TF-IDF + LR、TF-IDF + SVM、Word2Vec + SVM、Word2Vec + BiLSTM、DistilBERT。演示任务为 IMDb、Rotten Tomatoes、Yelp、AG News、MultiNLI。200 条样本的 quickstart 不作为演示模型或研究成绩来源。

只复制源码时，页面会说明缺少原研究结果或检查点，PPT 仍可下载。`LFTC_ROOT` 可指定保存原归档的工程根目录。新实验的运行与导出使用 [CLI](../docs/USAGE.md)，打开展示页面不会自动执行实验。
