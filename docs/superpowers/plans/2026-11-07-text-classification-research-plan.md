# 文本分类能力边界与失败机制 Implementation Plan

> 2026-11-07用户修订：今晚执行范围已缩减，改按[一晚训练计划](2026-11-07-overnight-research-plan.md)执行。原全矩阵保留为后续研究，不再要求12点交付前完成2,859项。

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 接入三套固定版本数据，建立受监督预算约束的分层评测、失败审计及机制对照，产出可重建的研究证据。

**Architecture:** 新建独立research_v3包和research-v3输出树，复用现有算法思路及models_extra.BiLSTM类，保持旧实验文件可核对。设计锁、选择锁、评估锁逐级约束数据与模型访问。主评测、人工审计、干预分析通过稳定sample_id和group_id连接。

**Tech Stack:** Windows PowerShell、Python3.11、NumPy/Pandas/PyArrow、scikit-learn、Gensim、PyTorch、Transformers、unittest。新环境初始候选保留requirements-share的固定版本，Torch候选2.6.0，额外datasketch用于MinHash、psutil用于内存采样；G0通过后生成完整精确lock，G2后禁止无记录升级。

**Spec:** [正式研究设计](../specs/2026-11-07-text-classification-research-design.md)，[机器协议](../specs/2026-11-07-research-protocol.json)。

**实施状态更新：** 用户随后授权完成训练前配置。Task1–6及必要前向/验证已执行，实际状态、证据和明确未执行项见[训练前完成记录](../reviews/2026-11-07-pretraining-completion.md)。以下复选框保留原始执行计划的历史状态，不作为当前进程或验收状态。旧Python3.14在本机仍存在，先前不可启动是沙箱可见性问题，不能据此认定被删除。

## Global Constraints

- 协议标识：`tc-explain-v3-20261107`；新增数据集恰为go_emotions、banking77、yahoo_answers_topics。
- 五主模型：tfidf_lr、tfidf_svm、w2v_svm、bilstm、distilbert。
- partition seed2026；data seeds2026/2027/2028；training seeds42/43/44；head permutations9101/9102/9103；statistics seed7301。
- 主输入为同一前254内容subword字符片段，含特殊token上限256；原文保留。
- 单标签主终点Macro-F1，多标签主终点Macro-AP；不跨任务平均。
- E2不得访问目标V/D标签或锚点选择结果；固定默认值及epoch。
- GoEmotions保留28维多标签，允许常量标签列，margin不冒充概率。
- 新代码research_v3/；全部新运行输出research-v3/；旧models*/results*/data*不覆盖。
- 三个锁分别为design.lock、selection.lock、evaluation.freeze；新Q预测只在最后一锁核验通过后产生。
- 人工身份、双人复核和干预验证均须真实完成；未知原因保留。
- 本文件描述未来执行，勾选项初始均未完成。已安装Superpowers不意味着研究环境或研究代码已完成。

## Review Focus

1. 相同/近重复文本跨T/V/D/Q，或标签冲突仅从测试移除而污染训练：由Task3隔离测试覆盖。
2. 每类1条、空词表、全正/全负多标签列及实际无法实现的长尾配额：由Task5、Task7退化与可行性测试覆盖。
3. 严格少样本偷偷读取锚点选择配置、验证标签或更大池词表：由Task8访问拦截测试覆盖。
4. 配置、输入或权重变化后续跑仍使用旧预测，或单标签score误解释为多标签概率：由Task9/Task10篡改测试覆盖。
5. 将多个seed或同一基例的改写当独立样本，将当前尾部与自然稀有类混用：由Task11/Task13的分母和配对统计测试覆盖。

## 0. 执行方式、先决条件与阶段依赖

本轮交付为设计和执行计划，不安装研究依赖、不训练模型。实际实施前审阅完整设计及本计划。推荐后续按单任务实现与独立审查推进；若采用本会话直接实现，同样保留全部阶段门。

当前可见交接目录未发现.git，git命令不可用；不假定能commit，也不在11GB交接目录自动git add全部。若后续进入合法Git工作副本，每个Task仅提交所列源文件/测试/配置；否则使用源码SHA与变更日志作为阶段检查点，不能声称已提交Git。

阶段顺序：
`P0环境与契约 → P1数据/输入 → P2矩阵与锚点选择 → P3冻结及主评测 → P4失败/机制 → P5报告与复现`。
各任务依赖在表中指定；只有门满足才进入后续批次。自动工程工作由执行者完成，人审由用户安排的真实成员完成，不预填姓名。

| Task | 交付 | 依赖 | 对应门 |
| --- | --- | --- | --- |
| 1 | 可运行环境、配置和CLI契约 | 无 | G0前置 |
| 2 | 固定资产与统一标签数据 | 1 | G1 |
| 3 | 划分、精确/近重复组与隔离 | 2 | G1 |
| 4 | 统一可见输入和审计 | 3 | G1 |
| 5 | 嵌套规模、K-shot与分布抽样 | 3、4 | G2a |
| 6 | 结构矩阵、依赖和运行身份 | 5 | G2a |
| 7 | 五模型、指标与多标签后备 | 4 | G0、G3 |
| 8 | 训练器、严格少样本、选择结果 | 6、7 | G2b |
| 9 | 三锁、断点与固定评测 | 8 | G3 |
| 10 | 现代参照实现与锚点选择 | 8、9的实现 | G2b前置 |
| 11 | 失败抽样/人审工具实现；真实人审在P4 | 3、9的实现 | G4 |
| 12 | 机制/干预实现与D试点；真实重训在P3 | 8、11的实现 | G2b前置、G4 |
| 13 | 统计实现，真实计算在P4 | 9、12的实现 | G4 |
| 14 | 证据驱动报告与总验收 | 10、11、12、13 | G5 |

下列Task首先交付经过fixture检验的软件能力，其中Task8、10、12只提前执行明确许可的锚点选择与D试点。真实长批次由§2的集中阶段执行，不能因Task编号靠后而在Q预测之后才搜索模型或重训。

P2先形成结构清单与design.lock，允许锚点训练和D试点；Task8、10锚点选择与Task12试点全部完成后解析所有实际run、样本量和contrasts，形成selection.lock，再开始长批次。M2/M3/M5和ModernBERT其余训练全部先于evaluation.freeze；新的Q只在该全研究锁通过后预测。三个锁不按结果滚动追加。缺试点只能继续软件工作和已有锚点，不能绕过selection.lock启动长批次。

## 1. 文件地图与公共接口

新增文件职责如下；旧pipeline.py、extension.py、hard_benchmark.py、imbalance.py、app.py不修改。

| 文件 | 职责 |
| --- | --- |
| research_v3/types.py、config.py、runtime.py、__main__.py | 类型、配置校验、环境证据、CLI |
| research_v3/assets.py、datasets.py | 固定版本下载、SHA与字段/标签映射 |
| research_v3/partitions.py、inputs.py | 分组隔离与可见输入 |
| research_v3/sampling.py、matrix.py | 条件构建、可行性、运行图 |
| research_v3/features.py、models.py、metrics.py | 训练内特征、五模型、多标签与指标 |
| research_v3/train.py、storage.py、freeze.py、evaluate.py | 训练、原子保存、锁与预测 |
| research_v3/audit.py、interventions.py、diagnostics.py | 人审、改写/窗口、机制重训 |
| research_v3/statistics.py、report.py | 配对估计与证据报告 |
| configs/research-v3.json、requirements-research.in | 从机器协议映射配置与研究依赖候选 |
| tests/research/test_*.py | 小型人工fixture上的有意义单元/集成测试 |

类型契约：
- `Protocol = dict[str, Any]`。
- `Example(dataset_id: str, sample_id: str, text: str, labels: tuple[int, ...], official_split: str, group_id: str, fields: dict[str, str])`。
- `PreparedDataset(dataset_id: str, task: str, label_names: tuple[str, ...], partitions: dict[str, tuple[Example, ...]], manifest_hash: str)`，partitions键为train/select/discovery/test/quarantine。
- `VisibleInput(sample_id: str, text: str, start_char: int, end_char: int, content_tokens: int, full_tokens: int, text_hash: str)`。
- `Condition(condition_id: str, experiment_id: str, dataset_id: str, train_ids: tuple[str, ...], data_seed: int, metadata: dict[str, Any])`。
- `RunSpec(run_id: str, condition_id: str, model_id: str, training_seed: int | None, hparams: dict[str, Any], input_hash: str, asset_hash: str, code_hash: str, strict_no_val: bool)`。
- `PredictionRow(run_id: str, sample_id: str, group_id: str, labels: tuple[int, ...], prediction: tuple[int, ...], scores: tuple[float, ...], score_kind: str, input_hash: str)`。
- 审计与干预记录使用以schema_version=1标记的JSONL字典；字段严格采用Spec§8、Task11/12定义，校验函数拒绝未声明状态与缺失ID。

常用验证命令在新环境建立后执行：
`$researchPython = (Resolve-Path '.venv-research/Scripts/python.exe').Path`。
每个Task局部测试通过后，运行`& $researchPython -m unittest discover -s tests/research -p 'test_*.py' -v`；触及公共训练/加载时另运行原有`test_project test_extension test_hard test_imbalance`。记录所有失败，不将局部通过说成全部通过。

## Task 1: 环境、配置与CLI

**Files:** Create research_v3/{__init__,types,config,runtime,__main__}.py、configs/research-v3.json、requirements-research.in、tests/research/test_runtime.py、test_config.py。

**Interfaces:** `load_protocol(path: Path) -> Protocol`；`validate_protocol(p: Protocol) -> None`；`probe_environment(output: Path) -> dict[str, Any]`；`main(argv: list[str] | None = None) -> int`。

- [ ] 先核实解释器。旧.venv的Python3.14在初始沙箱不可见，不复用它；本机实际存在的解释器应通过环境访问核实。检查`py -3.11 --version`，缺失则从可用官方Python安装渠道安装3.11并记录绝对路径；无法获得时G0为blocked，仍可做不依赖Python的文档工作。
- [ ] 使用已核实3.11解释器执行`-m venv .venv-research`。安装候选依赖，GPU wheel按驱动支持选择并记录index；CPU可完成小smoke，主GPU批次需单独实测。输出pip freeze、Python路径/版本、Torch CUDA、设备和驱动信息。
- [ ] 写失败测试：配置包含第四个新增数据集应抛ValueError；output_root指向旧results应拒绝；probe遇到不可用GPU保存原因，不能标GPU ready。
- [ ] 运行`& $researchPython -m unittest discover -s tests/research -p 'test_config.py' -v`，确认因接口未实现失败。
- [ ] 实现上述接口及全部CLI子命令的参数解析。未实现子命令返回非零not_implemented，不产生complete状态。
- [ ] 重跑局部与项目套件，运行`& $researchPython -m research_v3 preflight --config configs/research-v3.json --out research-v3/environment.json`。通过实际单标签/多标签smoke前环境状态为provisional。
- [ ] 保存锁定依赖候选、环境证据及变更检查点；不宣称原22项已在新环境通过，除非实际输出证实。

## Task 2: 固定资产与三套数据适配

**Files:** Create research_v3/assets.py、datasets.py、tests/research/test_assets.py、test_datasets.py。

**Interfaces:** `resolve_assets(p: Protocol, root: Path, offline: bool) -> dict[str, Any]`；`load_examples(dataset_id: str, lock: dict[str, Any]) -> tuple[Example, ...]`；`label_names(dataset_id: str) -> tuple[str, ...]`。

- [ ] 写失败测试：revision不是固定SHA、下载字节与SHA不符时拒绝；Go的[6,9,27]保留三个标签；Bank字符串/整数标签映射相同；Yahoo空question_content仍保留title和answer。
- [ ] 运行test_assets与test_datasets，确认缺接口失败。
- [ ] 只下载协议固定parquet文件及数据卡；现有四源与DistilBERT复用assets.lock所指文件但重新验SHA。ModernBERT解析一次revision并固定，再下载safetensors/tokenizer/config；该锁生成后禁止使用main。
- [ ] 读取官方字段并构造Example；ID包含split，标签映射来自实际features和原始categories，不凭字母顺序猜。
- [ ] 保存research-v3/assets/assets.lock.json、来源/许可元数据和官方计数。计数偏离卡片需审计说明；哈希只记录实读文件。
- [ ] 重跑测试和离线读取fixture；关闭网络仍能读取已完整缓存文件，缺文件报明确错误。保存检查点。

## Task 3: 合法划分、重复组与隔离

**Files:** Create research_v3/partitions.py、tests/research/test_partitions.py。

**Interfaces:** `prepare_partitions(examples: tuple[Example, ...], p: Protocol) -> dict[str, PreparedDataset]`；`build_duplicate_groups(examples: tuple[Example, ...], seed: int) -> dict[str, str]`。

- [ ] 写失败测试：相同文本在train/test只留test用途；完全同文冲突标签从T/V/D/Q全部进入quarantine；近重复链A—B—C形成同组；近重复不同标签不自动被当冲突；跨任务标签不比较冲突。
- [ ] 对人工fixture断言Bank各类的V/D配额各为max(1,floor(0.1*n_c))；Yahoo各类V=1000/D=500；Go与旧validation按60%分配。规模不足的fixture使用显式fixture配额，生产规则不被fixture改写。
- [ ] 先运行test_partitions确认失败，再实现确定性排序、去重连通分量、最高用途归属和计数。
- [ ] 保存T/V/D/Q/quarantine的ID与group清单、排除原因、近重复审计参数和manifest_hash。生成所有分区交集报告。
- [ ] 验证相同输入与种子重建出相同ID；改变输入字节会改变manifest_hash；各用途group交集为零。重跑套件，完成G1的划分部分。

## Task 4: 统一输入与可见证据

**Files:** Create research_v3/inputs.py、tests/research/test_inputs.py。

**Interfaces:** `make_visible_input(example: Example, tokenizer: Any, limit: int = 254) -> VisibleInput`；`materialize_inputs(data: PreparedDataset, tokenizer: Any) -> dict[str, VisibleInput]`；`audit_inputs(data: PreparedDataset, tokenizer: Any) -> dict[str, Any]`。

- [ ] 写失败测试：否定、标点和emoji不被清理掉；长文本片段字符范围与token offset一致；五模型获得同一text_hash；BiLSTM不二次截断；空白或空字段按协议处理。
- [ ] 运行test_inputs确认失败；使用fast tokenizer的offset映射实现统一片段，保留原文/字段与完整token数，不把word和subword计数混为一谈。
- [ ] 保存每条输入的字符范围、full/visible token数、截断与零文本标志。任何模型额外截断产生error而非悄悄继续。
- [ ] 以真实少量缓存文本核对截断边界、中文/组合字符及超长单词；只做输入检查，不生成Q模型预测。重跑套件并保存检查点。

## Task 5: 规模、严格K-shot与分布条件

**Files:** Create research_v3/sampling.py、tests/research/test_sampling.py。

**Interfaces:** `sample_scale(data: PreparedDataset, n: int | str, seed: int) -> Condition`；`sample_kshot(data: PreparedDataset, k: int, seed: int) -> Condition`；`allocate_long_tail(capacities: tuple[int, ...], n: int, rho: int, order: tuple[int, ...]) -> tuple[int, ...]`；`shared_tail_budget(capacities: tuple[int, ...], orders: tuple[tuple[int, ...], ...], ratios: tuple[int, ...], cap: int = 6000) -> int`；`sample_multilabel(data: PreparedDataset, n: int, alpha: float, seed: int) -> Condition`。

- [ ] 写失败测试：N子集嵌套；K-shot每类恰K且无重复；不足K返回infeasible而非删类；全量条件不因data seed产生伪重复。
- [ ] 写容量反例：某类容量很小，ρ=50请求不能超采；同一dataset所有ρ/排列必须使用同一N0；整数配额和=N，每类≥1，余数同分按label_id。
- [ ] 写多标签反例：同一文本多标签不得按互斥类拆成重复行；α共用U键，覆盖集固定；α=0与E1同N同seed的ID完全一致。
- [ ] 运行test_sampling确认失败，再实现Spec§6.1/6.2算法。输出请求和实际分布、共现差、自然稀有组与当前条件尾部。
- [ ] 检查未来数据池哈希不一致时旧Condition拒绝加载，无法满足覆盖时保存失败原因。重跑套件并保存检查点。

## Task 6: 结构矩阵、运行身份与预算清单

**Files:** Create research_v3/matrix.py、storage.py、tests/research/test_matrix.py、test_storage.py。

**Interfaces:** `build_design_matrix(data: dict[str, PreparedDataset], p: Protocol) -> dict[str, Any]`；`resolve_runs(design: dict[str, Any], selections: dict[str, Any], assets: dict[str, Any], code_hash: str) -> tuple[RunSpec, ...]`；`atomic_write_json(path: Path, payload: dict[str, Any], root: Path) -> None`。

- [ ] 写失败测试：恰三套新增、E1七套、E2六套、E3三套、E4一套、E5四代表；主模型五个；SVM无training_seed三重复制。
- [ ] 断言共享实际训练ID/输入/超参数的run只生成一个物理任务，保留逻辑别名；hparam选择节点未解析时禁止输出resolved状态。
- [ ] 断言输出路径逃出research-v3、与旧目录重合、写入中断留下半个JSON时均被防护。
- [ ] 运行测试确认失败；实现结构矩阵、任务依赖图、canonical hash及原子保存。生成design_manifest、各阶段拟合数、预计输出字节的实测外推和未满足条件列表。
- [ ] 命令：`& $researchPython -m research_v3 matrix --stage design --config configs/research-v3.json --root research-v3`；输出不含Q预测，验证合法条件总数可由清单重算。
- [ ] 形成design.lock并记录G2a，等待Task8选择与Task12的D试点解析selection.lock。

## Task 7: 特征、五模型、多标签与指标

**Files:** Create research_v3/features.py、models.py、metrics.py、tests/research/test_features.py、test_models.py、test_metrics.py；Reuse models_extra.BiLSTM class only。

**Interfaces:** `fit_features(model_id: str, train_inputs: tuple[VisibleInput, ...], hp: dict[str, Any]) -> Any`；`build_model(model_id: str, task: str, label_count: int, assets: dict[str, Any], hp: dict[str, Any], features: Any = None) -> Any`；`metric_bundle(labels: Any, scores: Any, task: str, score_kind: str, threshold: float | None = None, label_mask: Any = None) -> dict[str, Any]`。所有学习器只消费VisibleInput.text和获准训练标签，不能从Example.text读取全文；构造BiLSTM时features提供训练内词表与embedding matrix。

- [ ] 写失败测试：验证独有词不进入词表；零已知词映射可预测；全正/全负列保持28维；退化词表走有标记的训练先验后备，不读取V。
- [ ] 小型fixture固定预期：单标签完美预测F1=1；多标签用已知排序验证AP；相同SVM分数按margin0阈值，不当probability0.5；V/Q无正例或无负例label为NA并分别固定AP mask，V mask为空阻止选择；Q完整28列Macro-F1另算。
- [ ] 保持VisibleInput相同但更换原文不可见后段时，词表、输入batch和主预测不变，避免包装器又读取全文。
- [ ] 运行局部测试确认失败；实现TF-IDF、共享训练内Word2Vec、单标签/逐标签分类器、BiLSTM及DistilBERT的目标和预测契约。
- [ ] 使用256行工程训练与独立重载，比较保存前后scores/预测；神经确定性容差明确写入测试，不只比较标签。
- [ ] 重跑新套件和原四模块测试。失败逐项记录；不改旧模型路径或旧函数行为来掩盖新代码问题。保存检查点。

## Task 8: 训练、严格监督预算与选择结果

**Files:** Create research_v3/train.py、tests/research/test_training.py、test_label_budget.py。

**Interfaces:** `fit_run(run: RunSpec, data: PreparedDataset, inputs: dict[str, VisibleInput], assets: dict[str, Any], output: Path) -> dict[str, Any]`；`select_anchor(design: dict[str, Any], data: dict[str, PreparedDataset], inputs: dict[str, dict[str, VisibleInput]], assets: dict[str, Any]) -> dict[str, Any]`。

- [ ] 写失败测试：在strict_no_val=True时将V/D访问器替换为抛异常的sentinel，训练、epoch选择、阈值和词表拟合仍须成功；hparams不允许selected_by锚点引用。
- [ ] 验证E2只读取C×K唯一标注；固定epoch为10/5；严格实验不因V分数提前停止。其他实验按六候选、seed42、固定tie-break选参。
- [ ] 验证有效batch不因micro-batch减半改变；训练不中途更改输入上限；非收敛、NaN、OOM保留failed原因。
- [ ] 运行测试确认失败，再实现训练/选择与成本计量；显式seed分开控制权重初始化、batch顺序和数据抽样。
- [ ] 运行E0的15组合smoke，通过后完成G0。再运行锚点选择与D预测；不得读取Q预测。
- [ ] 保存selections.json、搜索总成本、每epoch历史、最佳checkpoint；与Task12的D试点合并，调用resolve_runs生成无未解析项的run_manifest和contrasts.json，形成selection.lock（G2b）。

## Task 9: 三阶段冻结、恢复与固定评测

**Files:** Create research_v3/freeze.py、evaluate.py；Extend storage.py；Create tests/research/test_freeze.py、test_evaluation.py。

**Interfaces:** `freeze_stage(stage: str, root: Path, inputs: dict[str, Any]) -> dict[str, Any]`；`verify_freeze(stage: str, root: Path) -> None`；`predict_run(run: RunSpec, examples: tuple[Example, ...], inputs: dict[str, VisibleInput], root: Path) -> tuple[PredictionRow, ...]`。

- [ ] 写失败测试：无evaluation.freeze拒绝Q；任一checkpoint、协议、输入、code_hash改变即拒绝；运行中断后只恢复匹配身份，不重用其他条件文件。
- [ ] 断言每个样本恰一行/run，scores长度与类别数一致；多标签预测允许空集合但金标非空；未知score_kind拒绝。
- [ ] 运行测试确认失败，实现三个锁、运行独占锁和原子状态写入；重复complete任务校验后复用，stale锁不能直接当任务已完成。
- [ ] 使用fixture验证E1—E6的训练依赖：ModernBERT和M2/M3/M5未完成时，evaluation.freeze拒绝；D试点pending时selection.lock拒绝。此Task不启动真实Q，真实批次统一在§2执行。
- [ ] 从保存PredictionRow重算所有指标，必须与记录一致；失败/不可行/待运行分别统计。原有Q标探索，新Q首次评估时间写入审计。
- [ ] 重跑局部和项目套件；验证可生成G3完整性报告，实际G3通过需等待§2真实批次，缺失任务不以均值填补。

## Task 10: ModernBERT参照与成本证据

**Files:** Extend models.py、train.py；Create tests/research/test_modern_reference.py。

**Interfaces:** 沿用build_model/fit_run/PredictionRow；新增`resource_profile(run_id: str, root: Path) -> dict[str, Any]`于runtime.py。

- [ ] 写失败测试：E5只能四代表任务和两档；相同输入hash；不允许ModernBERT静默再次截断；缺模型revision阻止运行。
- [ ] 运行测试确认失败，使用AutoModelForSequenceClassification，优先可验证的eager实现并关闭不适用的自动compile；实际版本能力以G0 smoke证据为准。
- [ ] 在design.lock之后运行ModernBERT的6k锚点六候选选择，仅使用T/V；其他档与重复种子纳入selection.lock后再统一训练。成本模块分别记录搜索、训练、特征准备、推理时间及硬件，不把收益说成纯架构收益。
- [ ] 检验CPU内存仅在实际采样psutil后报告，GPU allocated与总显存分开。保存E5证据并重跑套件。

## Task 11: 独立失败审计与条件轨迹

**Files:** Create research_v3/audit.py、tests/research/test_audit.py；Create research-v3/analysis/annotation-guide.md as runtime artifact。

**Interfaces:** `draw_audit(data: PreparedDataset, purpose: str, n: int, seed: int, excluded_groups: set[str]) -> tuple[dict[str, Any], ...]`；`validate_reviews(records: tuple[dict[str, Any], ...]) -> dict[str, Any]`；`track_failures(predictions: tuple[PredictionRow, ...], phenomenon_rows: tuple[dict[str, Any], ...]) -> tuple[dict[str, Any], ...]`。

- [ ] 写失败测试：A从Q随机抽200组并每组均匀取1条而非只抽错误；大小不同的组按π_i=(200/G)/m_g生成权重；D发现100与A隔离；模型身份在第一轮人审导出中隐藏。
- [ ] 断言P(error|phenomenon)与P(phenomenon|error)使用不同正确分母；多标签漏检/误报不被压成单一accuracy；同phenomenon可重叠。
- [ ] 验证空reviewer不得标double_reviewed；分歧须保留两份标注再裁决；单人状态不被自动升级。
- [ ] 运行测试确认失败，实现抽样与schema校验，使用D形成指南并在A揭示预测前冻结。已有60/24队列仅作历史发现，不并入A估计比例。
- [ ] 实现并以fixture核对800条正式审计记录的导出；真实抽样/复核在§2 P4执行，成员独立核实，工程执行者不伪造人审。
- [ ] 生成持续/纠正/新增/不稳定错误轨迹；从A中选解释案例时保留完整抽样框与正确对照。重跑套件。

## Task 12: 干预数据、原因对照与试点

**Files:** Create research_v3/interventions.py、diagnostics.py、tests/research/test_interventions.py、test_mechanisms.py。

**Interfaces:** `build_mechanism_runs(data: dict[str, PreparedDataset], p: Protocol, main_runs: tuple[RunSpec, ...]) -> tuple[RunSpec, ...]`；`validate_pair(pair: dict[str, Any]) -> None`；`plan_pair_count(discordant: int, pilot_n: int, family_size: int = 3) -> int`；`run_intervention(pair: dict[str, Any], run: RunSpec, root: Path) -> dict[str, Any]`。

- [ ] 写失败测试：M2四臂head state和CLS池化相同，freeze仅改变encoder梯度；M3 SVM各格sample_weight总量均为N0，削减组不引入新标签数据，当前尾部随排列改变。
- [ ] M1测试同编辑距离、1—3词、长度差≤2、必须双份有效标签核实；M4证据窗口包含完整span、控制不含span且选择最近合法起点，固定seed42不能按正确性换checkpoint。
- [ ] 功效测试：n被限制400—1200；pilot50与确认组无交集；资格不足时标underpowered，不能复制或放宽条件。
- [ ] 运行测试确认失败，严格实现Spec§7/8.3。人审内容由真实记录进入，不由程序假定自然语言改写有效。
- [ ] 在D完成三项各50的试点，只据预定公式确定n、可行资格和配对响应；进入selection.lock。没有真实试点则全研究selection.lock为pending，只继续软件工作/已声明锚点，不启动长批次。
- [ ] 实现并以fixture验证确认基例抽样、编辑记录和固定checkpoint诊断。M2/M3/M5的真实重训在P3、evaluation.freeze之前执行；Q改写/窗口预测仅在锁后P4执行。全部版本、原样本与干预记录保存。
- [ ] 保存每个假设支持/反对/不足以判断的原始证据，辅助分析不覆盖主要失败结果。重跑套件，交付G4机制部分。

## Task 13: 明确估计量、同时区间与结论规则

**Files:** Create research_v3/statistics.py、tests/research/test_statistics.py。

**Interfaces:** `estimate_contrasts(predictions: tuple[PredictionRow, ...], contrast_manifest: dict[str, Any]) -> dict[str, Any]`；`bootstrap_family(predictions: tuple[PredictionRow, ...], contrast_manifest: dict[str, Any], replicates: int = 2000, seed: int = 7301) -> dict[str, Any]`；`classify_effect(estimate: float, lower: float, upper: float, seed_deltas: tuple[float, ...], margin: float = 0.01) -> str`。

- [ ] 写失败测试：平均逐run F1不等于对平均概率算F1时，必须使用前者；SVM不制造三个独立训练样本；共享Q group和同基例改写使用同一重采样权重。
- [ ] 给定bootstrap误差矩阵断言critical为每行最大绝对误差的95%分位；所有contrast共享该critical，不能混用点态区间下优势结论。
- [ ] 断言区间含0判uncertain，整体落±0.01才判practically_close；正/负优势要求同时带越过相应margin且data seed方向一致；反转还须interaction同时带不含0。
- [ ] 断言E3尾部取当前rank底部、自然同频组为NA；支持数缺失标签的AP mask在所有模型一致，bootstrap退化率按Spec记录。
- [ ] 运行测试确认失败；实现Spec§9全部估计量、全局group重采样、F1—F6问题族与M2/M3/M5辅助族。原始与Holm敏感性p值另存，主要分类只用同时带。
- [ ] 小fixture用完全相同预测验证Δ及区间为0；用人工可算的不同预测验证配对差与误差带。实际分析只读取保存预测，不重新调参。重跑套件。

## Task 14: 报告、证据地图与总验收

**Files:** Create research_v3/report.py、tests/research/test_report.py；Create runtime outputs research-v3/analysis/{research-report-zh.md,research-report-en.md,evidence-map.csv,completion.json}。

**Interfaces:** `build_report(root: Path, output: Path) -> dict[str, Any]`；`verify_research(root: Path, required_scope: str) -> dict[str, Any]`。

- [ ] 写失败测试：模型指标没有PredictionRow、原因结论没有干预/审计记录、human_review未完成、全量run pending时，不能输出scope=full_research complete。
- [ ] 断言报告不生成不同任务的混合总分、不把辅助诊断写成主确认结论、不把旧测试重新标unseen；未知原因显示unresolved。
- [ ] 运行测试确认失败，再实现数据量/分布/学习曲线、各类失败、纠正与新增错误、干预、资源成本、限制和相关工作对照的报告。
- [ ] 运行全部新测试、原四模块测试、离线重载与指标重建；执行`& $researchPython -m research_v3 verify --root research-v3 --scope full_research`。
- [ ] 保存验收JSON：每个G0—G5的证据路径、SHA、运行状态、缺口和真实复核状态。只有全部必需证据满足才标研究完成；否则报告实际已完成scope。
- [ ] 生成归档清单，排除不允许再分发的原始语料与无关全量缓存，保留下载锁、配置、源码、预测与必要模型。若发布，另外执行用户实际授权的发布流程。

## 2. 阶段执行清单与验收

| 阶段 | 必须产物 | 验收动作 | 后续可启动 |
| --- | --- | --- | --- |
| P0 | 解释器、依赖lock、环境、CLI与测试 | 实际运行256行单/多标签保存重载 | 数据处理 |
| P1 | 三新增资产、全部T/V/D/Q、group、输入hash | 版本、计数、分区group交集、可见内容核对 | 结构矩阵 |
| P2a | design_manifest、design.lock | 条件与搜索依赖可完整枚举 | 锚点与D试点 |
| P2b | selections、实际run_manifest、n、contrasts、selection.lock | 无未解析任务与假定结果，拟合数/预算实际列明 | 长批次训练 |
| P3 | E1—E5及M2/M3/M5全部checkpoint、evaluation.freeze、Q预测 | 全部T/V训练完成，再核验全研究锁，最后首次Q预测 | 确认审计/干预预测 |
| P4 | 人审、干预、机制重训、同时区间 | 盲法/控制有效、配对、未解决原因保留 | 研究解释 |
| P5 | 报告、证据地图、复现及completion.json | G0—G5逐项通过 | 对外结论 |

执行顺序以依赖和验收为准，不给未经实测的工期承诺。数据或环境缺口不得转化为默认通过；计算不足可以保留pending并交付阶段性成果，但不能缩小后仍称完整研究。

真实集中执行动作依次为：①完成Task1—7及E0；②实现Task8—14并通过fixture测试；③生成design.lock；④运行五主模型及ModernBERT锚点选择、D预测和三项各50试点，冻结指南与n；⑤解析全run_manifest/contrasts并生成selection.lock；⑥训练所有E1—E5及M2/M3/M5运行，包括辅助优化敏感性；⑦生成并核验evaluation.freeze；⑧首次Q主评测；⑨进行A盲法复核及Q剩余组干预，冻结有效编辑后预测；⑩统计、报告和G5验收。已完成的锚点按身份复用。任何步骤缺证据均不跳过，也不把软件实现完成当作研究执行完成。

## 3. 本计划的自审项目

- [x] Spec每节都能指向Task与验收物，包含许可元数据、旧测试已见和监督预算，见requirements-traceability.csv。
- [x] 已核对所定义的类型、文件路径和函数签名；VisibleInput显式进入特征、训练和预测接口。
- [x] 每步有可核查结果；执行任务均保持未勾选，自审勾选不代表研究已执行。
- [x] 五类Review Focus均有所属Task中明确规定的反例测试；实际编写/运行属于未来任务。
- [x] 计划固定接口与实验决定，未提前编写全部程序实现。
- [x] 独立审查两轮提出的九项关键问题已落实；第三轮结论nothing important left，见reviews审查记录。

