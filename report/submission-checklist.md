# 项目交付状态

## 最终整合：训练类别不平衡实验

- 五类模型、四种训练比例，总量固定6,000；均衡验证／测试各1,000，每类250条。
- 9组均衡参考复用，新增27组不加权与4组类别加权，合计40组评估、40,000条预测。
- 比较Macro Precision/Recall/F1、MCC、少数类Recall/F1、逐类错误及相对自身基线衰退；提供三个深度训练种子的SD和条件bootstrap区间。
- 严重94/2/2/2条件：SVM Macro-F1 0.1000，BiLSTM 0.4424，DistilBERT 0.7910。类别加权后SVM 0.4000、DistilBERT 0.8193。结果限于本次World多数类的受控AG News实验。
- 最终阅读入口：`final-report.md`、`final-synthesis-zh.md`、`imbalance-report.md`；主讲PPT为12页`final-presentation.pptx`；真人录制安排仍见`recording-plan-final.md`。
- 英文GUI新增Imbalance robustness页；主评估与部署先验加权准确率模拟分开。演示预测继续使用均衡检查点。
- 22项单元／数据完整性检查全部通过；GUI任务切换、四种比例及少数类曲线检查通过。历史结果与均衡预测保持一致。
- 样本清单、设计预冻结、最终权重冻结和完整结果在`data-imbalance`／`results-imbalance`。权重在本机`models-imbalance`，紧凑压缩包不包含权重与原始文本。

## 新增 MultiNLI 难任务

- 前提／假设双输入三分类，同体裁与跨体裁两组保留评估。
- 训练／源验证／matched／mismatched为6,000／999／999／999，按premise分组隔离。
- 原五类模型均完成配对适配；另有hypothesis-only SVM诊断。
- 包含Macro/Micro/Weighted P/R/F1、Accuracy、MCC、逐类指标、混淆矩阵和premise-cluster bootstrap。
- 完整报告：`hard-benchmark-report.md`；补充PPT：`hard-benchmark-supplement.pptx`；压缩讲解安排：`recording-plan-final.md`。
- 全部18项单元／完整性检查通过。应用的双输入、三类输出、体裁切换及空输入检查通过。
- 课程抽样评估使用官方validation池，不声称取得官方隐藏测试榜单成绩。成员信息仍暂不处理。

## 当前 V2 更新

- GUI 已改为全英文，支持 Binary sentiment / Multiclass news 切换。
- 新增 AG News 四分类，原有三组二分类样本与权重保留。
- 五种方法：TF-IDF + LR、TF-IDF + SVM、Word2Vec + SVM、Word2Vec + BiLSTM、DistilBERT。
- 多指标：Accuracy、Macro/Micro/Weighted Precision/Recall/F1、MCC、每类支持数与混淆矩阵。
- 当前报告／PPT 优先使用 `classification-report-v2.md` 和 `classification-presentation-v2.pptx`；下方清单记录 V1 的原始交付。
- 成员信息与分工按用户要求暂不处理。

## 已完成

- [x] 三个公开数据集下载、固定仓库版本与文件校验值。
- [x] 658,662条原始样本审计、全局规范化哈希去重、24,000条平衡抽样。
- [x] 36个传统候选拟合、12次DistilBERT训练，保存15个最终模型。
- [x] 在测试前冻结配置、样本清单和最终权重哈希。
- [x] 45组来源—目标—种子评估、45,000条逐样本预测。
- [x] 每类指标、混淆矩阵、三种子变动及配对bootstrap区间。
- [x] 同一CPU、相同评论集合的median/p95推理计时，训练耗时和模型体积记录。
- [x] 60条实际错误／分歧复核队列与12例助手定性分析草稿。
- [x] 本地Streamlit交互应用、结果页面与错误案例展示。
- [x] 五项关键单元／完整性检查通过；应用真实推理、空输入、长文本提示检查通过。
- [x] 完整流程续跑复用检查通过，未重新训练或用测试成绩挑选新模型。
- [x] 中英文结果说明、六组PNG/PDF科学图表、13页可编辑PPT与约9分钟真人讲稿。
- [x] 最终PPT包结构、可编辑图表工作簿、字体声明、布局检查和逐页渲染审阅。

## 仍需真实成员完成

- [ ] 提供实际成员姓名和真实贡献，核实反思草稿并由本人签名。
- [ ] 由真实成员检查错误案例，填写复核人；本项目未声称两人已人工复核。
- [ ] 使用真人声音录制不超过10分钟的视频；现有讲稿不是已录制视频。
- [ ] 把真实GAI使用情况转入老师的指定声明模板并按要求签署。
- [ ] 对照syllabus完整rubric与NTULearn核实提交细则和截止时间。
- [ ] 上传作业，检查播放和文件完整；每人按通知完成个人互评。

## 交付范围

本机项目目录保留原始资产、处理后数据、全部最终模型权重，可直接运行演示。紧凑压缩包包括源码、版本锁、样本清单、结果、报告与PPT，不包括大体积资产、处理后文本和模型权重；在另一台机器解压后需要按README下载和训练。

未在原生PowerPoint中交互检查，也未上传或提交作业。PPT已通过文件级检查与渲染审阅。GPU确定性和CPU内存峰值测量的限制已在报告披露。
