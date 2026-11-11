# D 发现阶段：指南修订材料

Yelp、GoEmotions、BANKING77、Yahoo各100个独立D组，已按固定种子7301均匀抽组并在组内均匀取一条。原始来源文件和SHA在各frame.json中记录，全部仍为unreviewed。

各目录的frame.json提供完整规范化文本、标签、组、抽样框、纳入概率和空标注模板。证据span使用展示文本的Python Unicode字符偏移；原始文本仍保留在已冻结的D来源文件中。

两位实际复核者分别复制reviewer-A.blank.jsonl、reviewer-B.blank.jsonl并独立填写自己的真实稳定reviewer_id。A/B是文件占位标识。保留两份原始标注、分歧和裁决；没有判断的字段继续not_checked、unreviewed或unknown。

第一轮依据文本、来源标签与标签定义核实信息充分性和现象，提交前隐藏模型身份和输出。之后可在D上讨论匿名输出并修订正式指南；指南在A揭示模型输出前冻结。现象本身不能证明失败原因，原因结论须链接实际对照证据。

三项各50对的D干预试点材料在[discovery-review](../discovery-review/README.md)，具有其固定资格、顺序、双人有效性核实和编辑冻结要求。人员安排沿用“先生成材料，人员稍后安排”。
