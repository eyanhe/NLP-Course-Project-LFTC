# D试点人工复核材料

三个候选池已经按固定顺序生成，均来自D；每项最终需要50个经真实双人复核的有效配对。当前尚无任何已完成的人类复核或原因结论。

- Bank_M1_SVM/candidates.jsonl：937个自动满足长度/训练线索条件的组。
- Yelp_M4_DistilBERT/candidates.jsonl：61个320–2048内容token的组。
- Yahoo_M4_DistilBERT/candidates.jsonl：434个320–2048内容token的组。

这些数量只表示自动筛选通过，单一意图、连续关键证据、自然且标签保持的改写/对照仍须真实人员判断。不要查看模型身份、预测或是否改正错误后选择样本。

两位复核者各自使用稳定reviewer_id。原始candidates.jsonl保留，编辑副本保存为reviewed-candidates.jsonl；保留被拒绝样本及具体原因，不删掉它们。按candidate_order顺序审阅，用前50个独立复核合格且组不重复的基例；不足50则报告真实不足，不能复制或放宽资格。

M1：pair_template.original_text是模型实际可见文本，source_full_text_normalized保留完整来源。填写intervention_text和control_text；两者分别替换1–3个线索内容词/同数非线索内容词，英文word编辑距离相同，长度变化≤2，保持动作、对象、状态、否定与槽位。cue_words只来自固定6k训练集，不按模型权重或Q错误选择。两份review_templates分别填写实际复核ID、盲法、标签保持及控制自然性；不得由助手代填true。

M4：在source_full_text_normalized标记连续关键evidence_span=[start_char,end_char]，采用Python Unicode字符的左闭右开位置，证据须在前254内容token之外。工具据此建立254-token证据窗口与最近合法stride32控制窗口。两人复核全文标签、证据与控制，另记录控制是否仍有其他线索；不能根据模型对窗口的预测决定保留。

当前reviews=[]，所有复核字段为null，eligibility_status=awaiting_human_review。完成真实记录后才可校验/冻结编辑并用固定锚点checkpoint进行D试点预测。锚点选模仍仅使用T/V；候选池里没有模型预测。

样本量按真实50对中正确/错误不一致数的95% Wilson上界和预定公式计算，400–1200。人工指南、试点记录与n最终进入selection.lock，之后才可开始长批次。选模完成前primary checkpoint仍用已声明的selection_node_reference占位，不伪造已选模型。

用户当前安排：先生成复核材料，人员稍后安排。全部候选继续保持awaiting_human_review；此安排不表示复核完成。

两位复核者须分别保存自己的原始标注副本，例如reviewer-A.original.jsonl与reviewer-B.original.jsonl。先独立提交，再制作合并副本；分歧、裁决、拒绝理由与原始版本同时保留。reviewer_id填写实际稳定标识，不预填虚构人员。

后续导入接口已具备：research_v3.pilots.freeze_pilot_edits(root,pilot_id,reviewed_pairs)校验真实记录并冻结前50个有效独立组；evaluate_pilot(root,pilot_id)用指定T/V锚点评分，merge_pilot_result(root,pilot_id)显式合并已校验结果。导入的是pair_template经人工完成后的对象列表，不是尚未填写的候选包装记录。当前未对真实候选调用这些接口。

M1操作已按独立审查补充：[复核操作补充](../m1-review-clarification.md)。仅接受等word数的1–3词替换，不接受插入/删除或词序移动；两人分别填写intervention_content_words_replaced与control_content_words_replaced。它们在未复核前均为null；旧候选文件保留原样，在独立复核副本中添加字段。内容词由真实人员判断，不用自动词性猜测代替。这一保守子集在任何真实D响应前固定，减少可接受改写空间的代价会披露。

固定顺序要求：提交前50个有效独立组时，前面的每个候选都必须已有明确的人工决定。未复核的候选不能省略后跳到后面的50例。拒绝或无法编辑的基例请保留原始复核记录，并填写rejection_reason；至少由实际盲法复核者记录该决定。只有50个有效配对需要全部双人有效性核实。工具会把所有提交的原始reviewed_pairs与拒绝原因一起冻结，不能用空白模板冒充已拒绝案例。

完整标签映射见[label-map.md](label-map.md)，编号直接对应已冻结prepared_manifest的来源标签名称；这是参考映射，不是额外释义或人审结论。存在标签歧义时明确记录，不能为了凑够50例强行裁决。
