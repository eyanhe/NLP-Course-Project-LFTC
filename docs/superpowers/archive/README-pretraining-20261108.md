# 研究设计与执行入口

当前正式设计版本v1.0，协议tc-explain-v3-20261107。

- [完整研究设计](specs/2026-11-07-text-classification-research-design.md)：问题、选型、数据划分、实验矩阵、失败机制、人审、统计和边界。
- [执行计划](plans/2026-11-07-text-classification-research-plan.md)：14项任务的文件、接口、测试、依赖和G0—G5验收。
- [机器协议](specs/2026-11-07-research-protocol.json)：固定revision、种子和条件。
- [需求追踪](requirements-traceability.csv)：18项用户/研究/工程要求到任务和证据的映射。
- [独立审查](reviews/2026-11-07-research-design-review.md)：两轮九项问题的处理及第三轮结论。
- [来源核查](reviews/2026-11-07-source-verification.json)：实际读取的公开元数据，未下载文件的SHA不预填。
- [交付核验](reviews/2026-11-07-delivery-verification.json)：设计文件核验及哈希，不代表实验测试通过。

设计与计划已交付，随后已实施Task1–6的训练前准备：独立Python3.11环境、三套新增固定数据、七套数据完整划分和公共输入、真实容量下的结构矩阵，以及GPU前向检查。当前验收状态以[训练前机器验收](../../research-v3/pretraining-verification.json)为准；[操作入口](../../research-v3/PRETRAINING.md)列出工件、命令与后续边界。

- [实施补充规则](specs/2026-11-07-preparation-clarifications.md)：二分类尾部、Go覆盖、任务别名、运行复用、当前硬件与本阶段范围。
- [实现独立审查](reviews/2026-11-07-pretraining-code-review.md)：一项重要问题的RED→GREEN修复及一个延期小项。
- [训练前完成记录](reviews/2026-11-07-pretraining-completion.md)：实际数量、逐项证据和未执行事项。

模型训练、锚点选择、真实D试点和人审仍未开始。report下两份讨论稿保留为历史，实施以本目录为准。Superpowers的15项技能已安装到Codex用户技能目录；训练前实施使用了executing-plans、TDD、systematic-debugging、独立审查和verification-before-completion流程。

