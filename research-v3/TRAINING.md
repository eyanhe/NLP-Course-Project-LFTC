# 当前执行：一晚精简计划，原全矩阵延期

用户限定新加坡时间2026-11-07 18:30至2026-11-08 12:00，优先主要结论、减少工程量。按[一晚训练计划](../docs/superpowers/plans/2026-11-07-overnight-research-plan.md)及`training/overnight/run-plan.json`执行：P0新增32项，P1/P2可选25项，合计最多57项；复用选中60个检查点。其余原矩阵2,742项延期，不再启动旧`parallel_training`的全量入口。

最新实况：P0新增32项全部完成并通过保存重载。P1作业277766/277767/277768已提交。独立看门狗`training/overnight/watchdog.py`在本机隐藏后台每60秒监测并自动接续P2；实时状态见[实时进度](reports/overnight/live-progress.md)。GPU由Slurm运行；监测和阶段接续要求本机及VPN保持在线。本机仍不执行正式训练。

用户随后要求从原2,859项再挑高价值批次。已新增选中84个不重复请求：77项新训练、7项已有结果复用，见[补充计划](../docs/superpowers/plans/2026-11-07-overnight-extension.md)和`training/overnight/extension-run-plan.json`。当前合计选中201项（67复用、134新训练上限），原矩阵2,658项延期。补充CPU准备作业277786在执行；看门狗会在原P0/P1/P2完整结束且准备完成后自动提交补充GPU队列，截止时间不变。此前段落的2,742项延期是补充选定前的历史数量。

09:00停止启动新fit，10:00训练硬截止，保留最后两小时汇总。人审未完成时只交付V/D探索性结果，不生成正式Q预测。本机训练仍禁止。学校集群SSH已成功认证，尚未上传或提交作业；下列迁移记录保留为历史。

## 历史迁移记录：本机训练已停止

用户指令：停止本机训练，改用学校计算集群。

本机 worker PID7244 已终止，exec session69490 已结束。`training/authorization.json` 的 `training_enabled` 为 false；不得在本机自动重启训练。

停止时队列计数为94。连同尚未轮到的已有基线，共131个正式计划检查点已完成、通过重载核验并重新核对全部文件SHA256；另保留15个E0工程检查点。最后一个未完成任务保留为 interrupted，不计入完成。

目标为 `yiyan002@10.97.216.128`。VPN已连通；SSH返回NTU EEE/ROSE Cluster02登录信息。ED25519主机指纹与官方指南一致：`SHA256:AhYlWQBTsN/4GAzYTVTiZzrVhPhcurFMu1sBMglqvdM`。

用户重新提供的登录信息在 keyboard-interactive 和 password 两种认证中均被服务器拒绝，各尝试一次；已停止进一步自动尝试，等待用户手动确认当前凭据或改密提示。密码没有写入项目、命令或日志。此前获准检索的本项目Codex记录中未找到可确认的实际密码；Gemini全局目录检查被自动审批拒绝，未扩大检索。

集群官方技能：https://github.com/NTUEEECluster/docs/blob/main/skill.md 。训练与安装使用Slurm；登录节点只进行轻量操作。数据、环境与缓存放SSD项目目录，GPU批作业最长3天。必须先查询实际GPU、账号配额和存储，再生成集群专用执行授权。

基础迁移包 `cluster-base.tar` 为27,796,007,936字节，含1,348个文件；全部文件大小和SHA256已从包内逐项核验。整包SHA256为 `461b481ba52361ebc784cd94347d94b23e08c8404aaed417727f243331c80362`。基础包保持不变，最新执行说明和控制脚本通过小型覆盖包同步。

尚未上传或提交集群作业。迁移仍需完成Windows资产绝对路径适配、Linux环境与依赖核对、并行缓存协调、传输与模型重载验证。正式Q预测保持关闭，真实人审仍须在正式测试前完成。

原定范围共2,859项，尚未缩减：E1=657、E2=783、E3=513、E4=54、E5=60、M2=252、M3=420、M5=120。

以下保留历史执行记录，以上述最新用户指令为准。

---

# Historical execution record (superseded)

用户已批准主训练先跑。全部2,859个计划物理请求进入training/fixed-main-run-manifest.json，固定参数、不再搜索。队列优先小规模条件，复用同身份完整权重。执行：`.venv-research/Scripts/python.exe -m research_v3.parallel_training`。

当前worker PID46792，创建时间1794035865.7400453，exec session48889；唯一实时入口为training/progress.json。不要重复启动。首次启动的调度变量错误已修复，首批实际训练已完成，旧错误日志保留为历史。

三份D试点所需固定基线均已就绪；人工复核仍待人员安排，不再阻塞主训练。正式Q预测继续关闭。顺序修订详见training/training-order-amendment.md。全训练范围核验与全研究完成分开记录。

---
以下为先前阶段记录，执行顺序以上述最新用户修订为准。
# 当前政策：固定超参数，停止搜索

用户最新指示已覆盖原六候选搜索流程。39个参数组合已经固定，36份相同默认配置的既有检查点可复用。旧anchors入口已撤销执行许可，禁止重新启动网格搜索。

参数与修订说明见training/fixed-hyperparameters.json及training/fixed-hyperparameter-amendment.md。当前只补所需的Yahoo DistilBERT试点基线，一套参数；其余训练按实际研究条件执行。

当前执行命令：`.venv-research/Scripts/python.exe -m research_v3.fixed_policy --fit-needed-pilots`。进度仍读training/progress.json。人员安排沿用先准备材料、稍后安排真实复核。

---
以下保留先前执行记录；其中六候选搜索顺序已由上述用户修订覆盖。
# 当前训练执行入口

用户已授权：开始按计划完成训练。实验协议、数据资产和公共输入仍沿用已验收版本；执行许可独立保存于training/authorization.json。

状态以training/progress.json、相应runs/<run_id>/status.json及真实进程为准。complete仅在训练、文件校验和保存前/重载分数比较通过后写入。全局E0结论为training/engineering-smoke.json；锚点选择为selections.json。

执行顺序：15项真实E0工程拟合与D评估 -> 39组共234个锚点候选 -> 三个真实D试点 -> selection.lock -> 全部主/机制训练 -> evaluation.freeze -> 首次Q模型预测。没有真实试点不能开启长批次。

PowerShell命令：

```powershell
$researchPython = (Resolve-Path '.venv-research/Scripts/python.exe').Path
& $researchPython -m research_v3 train --phase engineering --authorization research-v3/training/authorization.json
& $researchPython -m research_v3 train --phase anchors --authorization research-v3/training/authorization.json
```

不要同时重复启动同一阶段。worker和run均有包含PID及进程创建时间的独占锁；同一request的完整结果经SHA核验可复用。代码/超参/输入身份改变后不能复用旧任务。

当前学习器只使用VisibleInput。E0评估来自D，Q预测保持关闭。运行记录包含训练内词表、完整目标维度、分数类型、特征/拟合成本、实际采样的CPU RSS及Torch allocated/reserved峰值。微批减半保持有效batch，OOM后从同一初始种子重启该条件并保留失败尝试。

人工试点/双人复核不预填，未知原因保留unknown；材料与人员安排独立于模型训练。先前PRETRAINING.md及历史验收只描述训练启动前的状态，当前执行进度读取本入口和training目录。

已实现的后续入口独立于正在运行的锚点程序，须按真实依赖执行：

```powershell
& $researchPython -m research_v3.workflow finalize-selection
& $researchPython -m research_v3.workflow train-batch --authorization research-v3/training/authorization.json
& $researchPython -m research_v3.workflow freeze-evaluation --authorization research-v3/training/authorization.json
```

finalize-selection先核实39组全部六候选及其V胜者、三个真实50例D试点和冻结编辑，再生成全量run_manifest/contrasts与selection.lock。train-batch复用同身份锚点并逐一执行全部唯一检查点；strict条件不加载V/D。freeze-evaluation要求所有主/机制检查点与阈值齐全；上述命令均未产生Q预测。当前人员安排为先准备材料、稍后安排复核，因此真实试点尚未完成，不能提前运行正式长批次。


