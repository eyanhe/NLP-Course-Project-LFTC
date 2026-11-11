# 训练前准备入口

当前阶段：仅完成训练前配置、数据与输入准备、结构矩阵、加载及验证。最终状态读取`pretraining-verification.json`，不能仅凭本说明或旧日志判断成功。

使用项目独立Python3.11环境：

```powershell
$researchPython = (Resolve-Path '.venv-research/Scripts/python.exe').Path
$env:PYTHONUTF8 = '1'
& $researchPython -m research_v3 verify --scope pretraining --config configs/research-v3.json
```

本会话的Windows沙箱看不到用户安装的Python基础解释器，代理运行需要获得环境访问；在本机PowerShell使用上面的路径。旧Python3.14仍存在，没有删除或复用旧环境。

主要工件：`environment.json`和`requirements-research-lock.txt`记录环境；`assets/assets.lock.json`固定31个源文件；`data/prepared_manifest.json`记录七套数据及T/V/D/Q/quarantine；`inputs/*.parquet`是所有主模型的公共可见文本；`matrix/design_manifest.json`和`matrix/conditions/*.ids.gz`记录真实训练ID与运行图；`design.lock.json`冻结源码、配置、数据和输入；`forward_checks.json`记录合成前向检查；`test-evidence.json`连接实际测试日志。`audit/`存失败审计格式及双人复核规则。

必要的重建命令：

```powershell
& $researchPython -m research_v3 preflight --config configs/research-v3.json
& $researchPython -m research_v3 prepare --offline --config configs/research-v3.json
& $researchPython -m research_v3 smoke --forward-only --config configs/research-v3.json
& $researchPython -m research_v3 matrix --stage design --config configs/research-v3.json
& $researchPython -m research_v3 verify --scope pretraining --config configs/research-v3.json
```

配置和源文件改变后需要重新核查并生成相应锁；旧锁不得被描述为仍有效。不要并行启动两个prepare或matrix写同一输出树。

训练、训练smoke和官方Q评测当前均返回blocked。下一阶段先实现并验收Task7–10所需学习器/训练器与三锁流程，再执行E0工程拟合、锚点选择和三个真实D试点。其后才有selection.lock；全研究必需训练完成后才有evaluation.freeze和首次Q模型预测。不能通过改一个开关跳过这些步骤。

CPU/GPU配置不代表完整训练smoke已通过。GPU推理检查只说明固定权重可加载、输出维度与数值正常、没有梯度或权重更新。当前RTX4060 8GB与旧报告的4090 Laptop不同，后续训练实际资源另记。

正式方案和执行计划：`../docs/superpowers/README.md`。实施边界与决定：`../docs/superpowers/specs/2026-11-07-preparation-clarifications.md`。独立审查与延期小项：`../docs/superpowers/reviews/2026-11-07-pretraining-code-review.md`。
