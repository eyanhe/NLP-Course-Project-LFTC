# Streamlit前端与课程交付

课程任务材料[第4页原文](<../reference/Group project details.pdf>)要求参照syllabus rubric，并举例用Streamlit增强最终产品。本次在保留原实时预测演示的基础上，加入当前v3研究的交互展示；完整syllabus rubric不在本仓库，不能仅凭该例子宣称所有评分项已经完成。

## 只看当前研究：无需权重

在安装了前端依赖的Python 3.11环境中，从项目根目录运行：

```powershell
python -m pip install -r requirements-frontend.txt
python -m streamlit run research_ui/viewer.py --server.address 127.0.0.1 --server.port 8501 --browser.gatherUsageStats false
```

浏览器打开 `http://127.0.0.1:8501`。此入口只读取GitHub包中的分析、图表和成品，不需要先下载52.65 GiB资产，也不会训练、重新测试或加载模型。

当前机器已在研究虚拟环境安装并验证前端依赖，可以运行：

```powershell
.\.venv-research\Scripts\python.exe -m streamlit run research_ui/viewer.py --server.address 127.0.0.1 --server.port 8501 --browser.gatherUsageStats false
```

## 原预测演示与研究页之间切换

完整依赖、原数据与权重的准备方式仍见[原交接说明](../HANDOFF.md)。运行原入口：

```powershell
python -m streamlit run app.py
```

原预测页面顶部新增“Explore current v3 research”链接，侧栏也有“Current Research”；研究页可以返回原预测演示。原二分类、新闻分类、MultiNLI和不平衡功能继续保留。

## 当前研究页包含什么

- 232条当前研究记录、226条V、42个Q模型及全部428条完成记录的口径。
- BANKING77与GoEmotions分别展示primary模型的Macro-F1、Macro-AP及种子波动，不混合不同指标排名。
- 按数据集、阶段和模型筛选正式Q明细，下载筛选后的CSV；空模型选择保留为空。
- 浏览已有机制对照、BANKING77词汇干预案例、12张图册图片。
- 下载两份Markdown、三份PDF、一份PPT和三份Excel。

页面展示保存的真实结果。V用于选配方，D用于探索，Q用于冻结后的测试；种子SD不是测试置信区间。各配方明细不混成一个总体模型排名。37对词汇改写采用Gemini辅助审核，不能当作双真人独立审核。

## 验证

```powershell
python -m unittest discover -s tests -p "test_research_frontend*.py" -v
python -m unittest -v test_project test_extension test_hard test_imbalance
```

12项前端数据/AppTest检查与22项原功能检查已通过。前端检查包含真实控件切换、空选择、图表/案例切换、九项成品下载组件和原页面导航；本地HTTP健康检查返回`ok`。AppTest属于程序化界面检查，不是人工浏览器视觉验收。
