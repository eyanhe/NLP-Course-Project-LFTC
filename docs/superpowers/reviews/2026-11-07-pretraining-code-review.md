# 训练前准备独立实现审查与处理

审查者：fresh-context `pretraining_final_review`；模式只读，未训练模型。无Git元数据，采用`.superpowers/sdd/2026-11-07-text-classification-research-plan/review-package.json`中的源码SHA检查点。

审查范围：正式计划Task1–6，以及本阶段所需的GPU前向加载检查与独立数据验收。原始结论为0 Critical、1 Important、1 Minor。审查确认分区隔离、多标签语义、K-shot预算、M2/M3/M5结构配置、物理复用和未训练状态基本符合协议。

## Important：已修复并验证

`matrix.py`原先未将全局训练配置纳入物理运行与解析运行身份。只将BiLSTM max_epochs从10改20，设计与run ID仍可能相同，后续断点恢复有错误复用风险。

新增`test_changed_training_budget_changes_design_and_resolved_identity`，先观察设计hash仍相同导致失败，再将完整公共协议hash和固定有效训练策略加入身份；解析超参包含模型默认设置。回归验证配置改变使设计/物理/解析身份改变，普通BiLSTM保留变更后的max_epochs与有效batch，同配置逻辑别名继续复用。全部研究48项及原有22项测试通过，日志和实际退出码见`research-v3/test-evidence.json`。

设计清单与design.lock必须在此修复后重新生成，最终是否通过以`research-v3/pretraining-verification.json`为准。

## Minor：延期

`preparation.py`在准备阶段检查空AP mask时只匹配`multi_label`，正式注册表使用`multilabel`。当前Go验证集全部28列均有正负支持，实际产物不受影响；独立最终验收支持两种别名并阻止空mask。因此本次保留为适配器统一工作中的小项，不把它描述为已修复。

## 审查未判断内容及执行裁定

- Task7–14的学习器、预测恢复和统计软件：按补充规则§8保留到后续；训练命令继续关闭。
- 反向传播、训练峰值内存、E0拟合、锚点搜索、真实D试点及模型表现：均未执行，不能报告通过。
- 当时仍在生成的最终矩阵、锁和独立验收：由主执行者在进程完成后核对实际文件与退出码，不能从审查意见推断通过。
- 审查者未重跑全量MinHash或全部测试：主执行者提供实际全数据准备与48+22项最终测试证据；审查者另做了不拟合的运行身份反例复现。

按executing-plans流程只进行这一轮独立审查与一次重要修复。修复由RED→GREEN和完整回归验证，不再派发重复审查。
