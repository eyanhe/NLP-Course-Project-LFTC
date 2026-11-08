# Convergence and Scoped Test Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 完成两个固定数据条件的验证集收敛选参、训练随机性复跑及独立冻结后的正式 Q 测试。

**Architecture:** 新增独立 convergence 训练入口、停止器和 scoped_evaluation 流程，复用已有数据、特征、度量和认证后的模型接口；历史学习器、原始请求及全局测试锁保持原身份。由一个有领取记录的队列分配给两个 CPU worker，冻结后统一推理。

**Tech Stack:** Python / PyTorch / Transformers / sklearn / Slurm / JSON / Parquet；结果工作簿使用项目可用的 spreadsheet 工具。

**Spec:** [实施设计](../specs/2026-11-08-convergence-and-scoped-test-design.md)。机器可读准备材料：`research-v3/training/convergence-v1/proposal.json`。该文件不是已完成的训练清单或正式测试冻结。

## Global Constraints

- 本轮条件固定为 BANKING77 N=1000、GoEmotions N=6000，data_seed=2026。
- 训练随机性检查固定 training_seed={42,43,44}，不更换训练样本。
- 固定 min_delta=0.0001，BANKING77 监控 V Macro-F1，GoEmotions 监控冻结 V 标签 mask 上的 Macro-AP。
- BiLSTM：lr={0.0003,0.001,0.003}、min_epochs=30、patience=10、max_epochs=100、effective_batch=64、constant LR。
- DistilBERT：lr={0.00002,0.00005,0.0001}、min_epochs=5、patience=5、max_epochs=30、effective_batch=16、warmup_updates=ceil(N/16)，此后 constant LR。
- SVM C={0.1,1,10}；只允许对通过请求与全部文件哈希校验的完全相同产物复用。
- 两 worker 各 partition=cluster02、account=msc、qos=msc、cpus-per-task=16、mem=48G；模型线程=8、micro_batch=4。
- 2026-11-08 09:00 SGT 后不启动新训练，10:00 SGT 停止训练。
- 旧学习器、已有 checkpoint、原 2859 请求和原全局测试门禁不改写；新行为绑定独立代码身份。
- 原目录无 .git，也无可用 git 命令；采用新增文件隔离和文件 SHA256 台账，不创建提交。

## Review Focus

- 最低轮数前验证指标处于低位、随后才改善：不能提前终止。
- 连续小幅提升累计超过 min_delta：重置 patience，同时最佳 checkpoint 始终保留原始指标最大值。
- 修改最大轮数：不能改变此前 LR 轨迹；恢复训练必须恢复优化器、随机状态和早停计数。
- 上限/截止/异常退出：不能当作 patience 收敛，也不能被静默跳过以释放 Q。
- 冻结后模型/阈值/文件变化或缓存不匹配：必须在 Q 加载前拒绝；三个训练种子全部测试，不能择种子。

### Task 1: 版本化训练与可靠停止

**Files:** Create `research_v3/convergence_training.py`; test `tests/research/test_convergence_training.py`.

**Interfaces:**
- `StopState.update(epoch: int, value: float) -> bool`：持久化原始最佳值/轮次、显著改善参考值、无改善次数，返回是否满足 patience。
- `lr_multiplier(update: int, warmup_updates: int) -> float`：恒定或固定长度 warmup 后恒定；不接收 max_epochs。
- `convergence_code_hash() -> str`：绑定新训练文件及所有实际调用的旧学习器依赖。
- `fit_convergence(entry: dict, data: PreparedDataset, inputs: dict, assets: dict, output: Path, deadline: datetime) -> dict`：输出兼容认证重载接口的 bundle/status，并附完整停止与调度元数据。

- [ ] 先写失败测试：最低轮数后才开始累计 patience；迟发改善和累计微小改善；完全同分取更早 checkpoint；NaN 拒绝；改变 max_epochs 不改变已执行更新的 LR。
- [ ] Run `.venv-research/Scripts/python.exe -m pytest tests/research/test_convergence_training.py -q`，确认新增能力缺失导致失败。
- [ ] 实现独立神经循环，复用旧 `_condition/_features/_batch/_logits/_neural_scores/_checkpoint` 的确定输入接口；不 monkeypatch 旧学习器。原 RunSpec 不变，新请求 code_hash 绑定新代码及依赖。
- [ ] 实现训练状态保存/恢复；用微型模型比较中断恢复与连续训练的更新序列、LR、patience 和最佳模型预测。验证最大分数对应 checkpoint 的磁盘重载。
- [ ] status 将 stop_reason、最后轮次、选中轮次分别记录；deadline/max_epochs 标记未通过收敛检查。通过微型真实训练和已有训练/特征/指标兼容检查后记录文件 SHA。

### Task 2: 具体请求、CPU 队列和 V 选参

**Files:** Create `research_v3/convergence.py`; test `tests/research/test_convergence.py`; create runtime files under `research-v3/training/convergence-v1/` and `research-v3/reports/overnight/convergence-v1/`.

**Interfaces:**
- `prepare(root: Path, proposal: dict) -> dict`：从明确来源构造 12 神经网格、6 SVM 候选和 8 待选配方复跑槽位，绑定不可变 run_id、条件与代码；候选复跑阶段分两次冻结，不能给尚未选择的 LR 伪造已解析 RunSpec。
- `select_recipes(candidate_statuses: list[dict]) -> dict`：只使用 seed=42、全部候选终态及指定 V 主指标；1e-12 同分取较小 lr/C。
- `claim_next(root: Path, worker_id: str, now: datetime) -> dict | None` / `worker(root: Path, manifest: Path) -> None`：原子领取、互斥、时限、Slurm 计算节点约束与终态记录。

- [ ] 先写失败测试：源对象及 T/V 角色哈希不变；恰好 12+6 候选；43/44 继承已选 LR；SVM 不复制种子；候选缺失/运行失败/触及上限拒绝自动选择；并发不能双领；09:00 后拒绝新训练。
- [ ] 运行新测试确认失败，再实现清单、记录和确定性选参。SVM 用相同源特征规则拟合，验证求解器没有触及迭代上限及收敛警告。
- [ ] 使用独立资源授权与执行台账；两 worker 原子领取预计较长任务，GPU 队列不纳入本轮。运行单个微型 smoke，核验身份、完整曲线和模型重载。
- [ ] 提交前读取当前官方使用规则、squeue/账户限制、CPU 状态，sbatch --test-only；真实提交保存作业号与脚本哈希，防重复提交。
- [ ] 观察首个真实进度及产物。只读同步状态，所有候选完成后冻结配方，产生 8 个额外种子 RunSpec，记录选参材料哈希后执行。
- [ ] 若任一必要候选或复跑未完成，输出具体缺口，测试步骤保持未开启。

### Task 3: 独立冻结与正式测试

**Files:** Create `research_v3/scoped_evaluation.py`; test `tests/research/test_scoped_evaluation.py`; runtime files `research-v3/training/convergence-v1/scoped-evaluation.freeze.json`, `research-v3/reports/overnight/convergence-v1/evaluation-started.json`, `predictions/`.

**Interfaces:**
- `freeze_scope(root: Path, selected: dict, references: list[dict]) -> dict`：认证 14 主 checkpoint 与预先指定的 4 历史参考模型，先冻结主次对比、数据及阈值身份，不加载 Q。
- `verify_scope(root: Path, scope: Path) -> dict`：完整验证源与模型哈希、候选终态及选择规则，返回已认证冻结内容。
- `predict_scope(root: Path, scope: Path) -> dict`：先 verify_scope，写测试开启日志，再按冻结数据完整加载 Q，CPU 分批推理、保存 float64 分数和逐样本身份，回读重算指标。

- [ ] 先写失败测试：冻结前 Q loader 调用次数为 0；模型、选择证据、代码、阈值或 Q 源字节变化均在 loader 前拒绝；非完整 Q、重复 sample_id、group/gold 错配、缓存身份变化拒绝。
- [ ] 用隔离 fixture 验证 18 个物理 checkpoint、三个神经种子和主/辅助对比；训练种子只用于报告离散程度，不择优。
- [ ] 实现 scoped freeze，清楚记录旧全局实验未完成、本轮 E1 范围以及 Gemini 复核与本测试的独立关系；不改写旧 freeze.py/evaluate.py 门禁。
- [ ] 复用 `evaluate.validate_prediction_rows/metric_bundle` 与经过认证的 CPU scorer；Q loader 仅在本步骤调用。主阈值保持 logit/margin=0，单标签 argmax；可选全局 V 阈值先冻结并辅助报告。
- [ ] 在 Slurm 节点执行正式测试；每模型逐样本对齐，保存开启时间、冻结哈希、Q 源哈希、运行时间和产物 SHA。新旧优化差值以整体配方比较命名。

### Task 4: 可读台账、图表和审阅

**Files:** Create `research-v3/reports/overnight/convergence-v1/build-results.py`; test fixtures under `tests/research/`; generate experiment ledger, Excel and static scientific figures in the same results directory.

- [ ] 用小型 fixture 检查模型、条件、种子和预测严格对齐；训练种子均值/SD 与配对样本区间分别输出，单模型 SVM 不伪造三种子。
- [ ] 生成实验 Excel：每行条件、阶段、lr/C、种子、状态、stop_reason、最佳/最后 epoch、V/Q、run_id、来源和文件链接。绘图时采用每任务独立尺度与标签，图注明确比较条件和主指标。
- [ ] 输出四类图：四面板收敛曲线、两个任务的最终三模型比较、四个新旧优化配方差值、逐类表现与错误案例；沿用原 D 机制证据，不从 Q 分数直接推出机制。
- [ ] 运行新测试及相关已有训练、阈值、评估、统计测试；仅在新增变更/失败时追加检查。真实绘图与 Excel 需要打开或渲染查看，核验表格和图的数值与预测文件一致。
- [ ] 按 requesting-code-review 技能安排独立审阅，检查隔离、数据泄漏、停止状态、选参、冻结和交付。修复发现后记录验证与产物哈希。

## 本轮规划完成核验

- [x] 核对原训练逻辑、固定预算曲线与全部 24 个补充训练终态。
- [x] 给出具体 grid、停止/调度规则、额外种子、SVM 对照和受限测试范围。
- [x] 从真实日志估算计算量，保留既定截止约束。
- [x] 仅规划与读取现有证据；未提交本方案作业，未生成正式测试预测。

执行实现任务尚未开始。先前用户的方向性同意记录在实施设计中；具体数值是本次给出的建议，不伪装为用户逐项指定。
