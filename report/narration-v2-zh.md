# V2 真人录制讲稿（目标9分钟）

这份讲稿供真实成员理解和录制。不得使用AI配音；成员身份、贡献和签名仍按用户要求暂不处理。

## 0:00–0:40 问题与课程对应

我们比较传统NLP特征与深度学习文本分类。传统方法包括TF-IDF加SVM、Word2Vec均值特征加SVM，深度方法包括Word2Vec初始化的BiLSTM和微调DistilBERT。逻辑回归保留作额外基线。任务同时覆盖二分类和四分类，不要求把课程举例中的所有算法都实现。

## 0:40–1:30 数据与划分

原有IMDb、Rotten Tomatoes、Yelp用于正负情感二分类；新增AG News预测World、Sports、Business、Sci/Tech。每个语料使用6,000训练、1,000验证、1,000测试，类别均衡。我们保留官方测试来源、固定抽样和去重。新闻和情感是不同任务，不能把两者分数差异全部解释为类别数变化。

## 1:30–2:30 方法与调参

TF-IDF保留统计n-gram特征；Word2Vec学习100维局部词向量，其均值丢失词序。BiLSTM保留序列信息，用两个方向各64个隐藏单元；DistilBERT使用外部预训练上下文。Word2Vec和词表仅用训练文本；验证Macro-F1选择参数和epoch，深度方法各三个种子。BiLSTM最多256个单词token，DistilBERT最多256个子词token，二者不是相同长度单位。

## 2:30–3:20 评价指标

Precision表示预测为某类的样本有多少正确；Recall表示该类真实样本找回了多少；F1平衡二者。Macro对类别等权，Weighted按真实支持数加权，Micro合并所有决策。由于测试集均衡，Macro和Weighted相同；单标签分类的Micro-P/R/F1等于Accuracy。不能把这些数学上的重复当作多份独立证据。混淆矩阵还展示错到了哪一类。

## 3:20–4:40 实际结果与“哪个更好”

展示V2多指标表。AG News中，DistilBERT平均Macro-F1为0.9195，TF-IDF+SVM为0.8776，BiLSTM为0.8678，Word2Vec+SVM为0.8497；逻辑回归为0.8800。以报告保存的完整数字为准。DistilBERT本次分数最高，但有外部预训练优势。BiLSTM并未普遍胜过TF-IDF；这说明需要实际比较，不能凭深度模型名称判断优劣。

如果应用特别怕误报，就看相应类别的Precision；如果怕漏掉相关文本，就看Recall；若需平衡两者，看F1。CPU延迟和资源预算也影响选择。现有配对区间只验证旧二分类中预先指定的SVM和DistilBERT比较，不能扩展成所有模型的显著性结论。

## 4:40–5:30 多分类错误与成本

展示AG News四乘四混淆矩阵。行是真值，列是预测，找到图中最大的非对角格，并用报告里的实际计数解释。类间混淆可以提示需要检查的文本，但不能证明模型内部原因。展示同CPU计时，说明采用相同200条评论作为共同输入，这不是新闻业务实际延迟基准。

## 5:30–6:15 Python代码

展示extension.py里的metric_set：统一计算多分类指标。展示models_extra.py的Word2Vec均值、BiLSTM打包序列与分类头。展示freeze发生在evaluate之前，以及results-v2保存的逐条预测与每类指标。旧二分类测试已在此前查看过，因此新增二分类模型属于扩展比较，报告已披露。

## 6:15–8:15 英文GUI演示

切换Binary sentiment，输入含转折的英文影评，比较五个模型。说明SVM margin不是概率。

切换Multiclass news，输入一条体育新闻，查看四分类标签；再输入一条商业或科技新闻。打开Evaluation & comparison，切换Macro／Weighted／Micro，查看Precision、Recall、F1、Accuracy。选中BiLSTM和DistilBERT，比较每类P/R/F1和混淆矩阵；切换行归一化。说明总体高分仍可能掩盖某一类的低Recall。

## 8:15–9:00 结论与限制

本次DistilBERT取得最高平均分，TF-IDF提供强而快速的传统基线，BiLSTM没有在当前有限训练预算下超越TF-IDF。结论仅限抽样任务；外部预训练、截断单位、局部词向量质量和旧测试集复用都限制推断。展示参考文献，由成员真人结束讲解。
