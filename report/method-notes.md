# 方法讲解备忘

本文件解释项目代码实际使用的方法。实验数字以正式报告为准。

## TF-IDF

TF-IDF 将文本转为稀疏词特征。项目的 `sublinear_tf=True` 使用 1+log(tf)；默认平滑 IDF 为 log((1+n)/(1+df))+1，再做默认 L2 归一化。词表与 IDF 只能从当前来源的训练文本拟合，其他 split 使用已保存的转换器。

unigram 表示单词，bigram 额外表达相邻词组合，例如 `not good`。这仍是统计特征，不完整建模句法、反讽或长距离关系。`max_features=30000` 控制词表上限，C 调节分类器正则化；C 越大通常表示更弱的正则化。项目通过验证集选择配置，不预设 bigram 必定更好。

Logistic Regression 用线性分数经 logistic 函数输出类别概率。Linear SVM 用间隔目标学习分界，其 decision_function 输出 margin，不是经过校准的概率。两者共享稀疏表示方案，但最优超参数可能不同；因此主实验比较完整 pipeline，不把分数差异归结为分类器的纯因果效应。

## DistilBERT

本次基础模型配置是6个 Transformer 层、768维隐藏表示。预训练来自外部语料；本项目只微调公开基础权重，不自行蒸馏或预训练。

自注意力使用 Q、K、V 表示，核心为 softmax(QK^T/sqrt(d_k))V。多个注意力头在上下文中更新 token 表示。分类模型取第一个位置的隐藏向量，经线性层、ReLU、dropout 和二分类头生成两个 logits，交叉熵监督训练。

分类头随机初始化，所以首次加载出现部分分类头权重未从基础模型初始化的提示是预期行为。必须经过训练后才能把它当作情感分类模型。

DistilBERT 的动态 padding 随当前 batch 补齐，输入只保留前256个 token；这不等于256个英文单词。TF-IDF 则接收整条清理后的文本。比较结果同时反映外部预训练、表示、训练算法和长度预算，不能称为完全隔离变量的架构实验。

## 指标与统计

Precision=TP/(TP+FP)，Recall=TP/(TP+FN)，F1=2PR/(P+R)。本项目对正负两类各算 F1，再取平均得到 Macro-F1。Accuracy 是所有样本中预测正确的比例。混淆矩阵行是真值，列是预测，顺序均为 negative、positive。

三个训练种子的标准差描述固定样本下的训练变动。配对 bootstrap 使用同一批抽样索引同时重采样 SVM 和 DistilBERT seed42 的预测，反映条件于两个固定模型的测试样本变动。它不合并训练随机性，也不保证类别独立于同电影或同作者的聚类结构。

跨集评估直接使用来源模型，不在目标标签上调参。来源与目标难度不同，所以迁移矩阵可描述性能变化，无法单独证明领域变化造成多少损失。

## 文档

- [TF-IDF 官方参数](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
- [LinearSVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html)
- [DistilBERT 官方模型说明](https://huggingface.co/docs/transformers/model_doc/distilbert)
- [基础模型卡](https://huggingface.co/distilbert/distilbert-base-uncased)
- [原论文](https://arxiv.org/abs/1910.01108)
