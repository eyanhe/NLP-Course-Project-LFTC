# Text Classification Lab：文本分类模型的能力边界与失败机制

当前主线：**模型在什么数据条件下表现更好，以及优化、预训练、标注支持和词汇线索如何影响失败。** 承接原常规分类、MultiNLI和AG News不平衡比较，优先阅读下列当前成品。

远程文件分工：[GitHub源码、报告与结果](https://github.com/eyanhe/NLP-Course-Project-LFTC) · [Hugging Face数据与检查点](https://huggingface.co/datasets/Eyanhe/LFTC_Dataset_And_Checkpoints)。大资产的路径、发布状态和恢复方法见[发布说明](docs/publishing/README.md)。

## 最终报告与展示

- [中文综合结论](report/research-v3/final-synthesis-zh.md)：研究主线与六组证据。
- [完整分析报告](report/research-v3/final-report-zh.md)：42个正式Q模型、定量对照与具体错误案例。
- [完整研究图册](report/research-v3/figures/完整研究图册.pdf)：12页研究接续与结果图；[实验逻辑链](report/research-v3/figures/当前研究问题与实验逻辑链.pdf)解释问题与对照关系。
- [研究图表演示PPT](report/research-v3/final-presentation.pptx)、[Excel表格与全部成品](report/research-v3/README.md)。
- [Streamlit交互研究界面](docs/FRONTEND.md)：筛选Q结果、查看机制/案例/图表、下载成品；保留原实时预测演示。

## 当前研究范围

截至2026-11-08，v3完成记录共 **428条**，其中当前研究 **232条**、额外早期选参181条、工程检查15条；另2条中断不计完成。当前研究中 **226条有V指标，42个固定模型完成正式Q评估**。

当前232条由首批184、S1新增16、S3新增8和后续优化新增24组成。V用于选择/验证，D用于探索，Q用于固定模型的正式测试。S2/S4复用已有模型，不增加训练数量。原2859项设计只执行其中184项，不能把全部计划写成已完成。

具体口径见[项目沿革](docs/PROJECT-HISTORY.md)，研究衔接见[交接前后的连续论证](docs/RESEARCH-CONTINUITY.md)。

## 文件结构

| 路径 | 内容 |
| --- | --- |
| `report/research-v3/` | 当前成品：综合结论、完整报告、PDF图册、PPT、Excel及来源清单 |
| `report/`中的原文件 | 原交接课程项目的报告、PPT、录制安排与草稿，保留原路径 |
| `research_v3/` | 当前研究的Python源码包 |
| `research-v3/` | 当前研究的数据、模型、实验矩阵、预测、冻结协议和分析源 |
| `data*/`、`models*/`、`results*/`、`assets/` | 原交接的数据、模型、结果和基础资产 |
| `outputs/` | 原工作簿、生成工件和预览 |
| `development-support/` | 构建器、检查工具与归档恢复脚本 |
| `docs/` | 研究设计、沿革、执行记录、目录与归档清单 |
| `app.py`、`pipeline.py`等 | 原交接GUI和实验入口，沿用原路径 |

[详细布局与体积说明](docs/REPOSITORY-LAYOUT.md) · [研究源码说明](research_v3/README.md) · [实验原始目录](research-v3/README.md) · [设计与执行记录](docs/superpowers/README.md)

## 使用与复现

先看当前v3研究，安装轻量前端依赖即可，不需要下载模型：

```powershell
python -m pip install -r requirements-frontend.txt
python -m streamlit run research_ui/viewer.py --server.address 127.0.0.1 --server.port 8501 --browser.gatherUsageStats false
```

完整[前端使用说明](docs/FRONTEND.md)。原课程GUI、依赖安装与离线演示见[HANDOFF.md](HANDOFF.md)。在已安装完整项目依赖的Python环境中，运行原入口：

```powershell
python -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501 --browser.gatherUsageStats false
```

原GUI新增通向当前研究页的导航，原预测功能继续保留。研究原始预测、模型、冻结范围与生成器保持原路径，详见[分析复现说明](research-v3/reports/overnight/analysis-v2/README.md)和[开发支持](development-support/README.md)。整理时受限沙箱无法启动本机虚拟环境；后续已在沙箱外验证Python 3.11的发布与研究环境可用，并完成前端及原功能检查。

## 归档与空间

18个集群传输tar包（29.51 GiB）移到项目旁的 `sentiment-project-archive/2026-11-08/transfer-packages/`，原目录保留传输脚本与校验说明。[归档清单](docs/repository-archive-manifest.json)记录原/新路径与SHA256，[恢复方法](docs/REPOSITORY-LAYOUT.md)可将它们移回原位置。

本次没有删除模型、数据或预测。移出主目录只减少主目录体积，不等于释放磁盘空间。编辑器默认隐藏本地环境、缓存、工具与相邻归档；数据和模型仍可在文件树中访问。

## 原交接与历史

原交接使用方式见[交接README原文](docs/archive/README-handoff-20261108.md)、[HANDOFF](HANDOFF.md)和[原报告入口](report/README.md)。历史状态JSON、旧图和旧报告保留为当时快照；当前结论以本页链接的报告为准。原打包脚本针对旧布局，不适合作为当前轻量交付命令。
