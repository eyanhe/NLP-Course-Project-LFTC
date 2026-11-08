# 仓库布局与体积说明

整理日期：2026-11-08，新加坡时间。按原交接项目“源码入口、报告、数据、模型、结果、开发支持”的方式组织；保留科研文件的路径和内容。

## 日常阅读

从[当前成品目录](../report/research-v3/README.md)进入：中文综合结论、完整报告、三份PDF、一份PPT和三份Excel共九项。研究的原始分析来源仍在 `research-v3/reports/`，工作簿构建源与预览仍在 `outputs/` 和 `development-support/`。

```text
sentiment-project/
├── README.md                  阅读入口
├── HANDOFF.md                 原交接记录，保留原文
├── report/
│   ├── research-v3/           当前成品：报告、figures/、tables/、PPT
│   └── 原报告和PPT             原课程交接成果，沿用原路径
├── research_v3/               当前研究Python源码
├── research-v3/
│   ├── data/、inputs/、assets/ 原数据、划分与公共输入
│   ├── runs/、features/       原模型和特征
│   ├── reports/               逐样本预测、冻结证据、分析源和历史图
│   └── training/、matrix/     实验设计、训练协议与执行快照
├── data*/、models*/、results*/ 原交接数据、权重和结果
├── outputs/                   原工作簿及工件生成目录
├── development-support/       报告构建、检查及归档恢复工具
├── docs/                      研究沿革、接续、设计和整理清单
└── tests/                     原测试
```

## 大体积传输包归档

18个 `.tar` 传输包共 **31,684,085,248字节（29.51 GiB）**，已移到项目旁的目录，内部保持原相对路径：

```text
6405 自然语言处理 课程作业/
├── sentiment-project/         日常工作目录
└── sentiment-project-archive/
    └── 2026-11-08/transfer-packages/
        ├── manifest.json
        └── research-v3/.../*.tar
```

[移动清单](repository-archive-manifest.json)记录每个包的原路径、新路径、大小和SHA256。原目录内的小型传输脚本、协议与 `.sha256` 文件继续保留。历史文档中的传输包路径是当时路径，再次执行相关上传/解包流程前，应先恢复。

在项目根目录运行，只验证归档：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\development-support\repository\restore-transfer-packages.ps1 -VerifyOnly
```

恢复到原位置：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\development-support\repository\restore-transfer-packages.ps1
```

恢复脚本在移动前检查全部目标和SHA256，拒绝越出工作区及覆盖不同内容；已有同哈希文件会跳过。恢复使用移动，不额外复制一套大包。已恢复的包也能用 `-VerifyOnly` 校验。

这是**移出主目录的归档**，没有删除文件，也没有释放同等数量的磁盘空间。模型和数据仍是主目录的大部分体积；其中v3模型约39.20 GiB。进一步减少磁盘占用需要另外确定哪些模型或历史资产可以丢弃。

核验时主目录由 **89.17 GiB降至59.66 GiB，减少约33.09%**。少量新增成品副本和整理清单计入整理后的大小。

## 环境、搜索与打包

- `.venv*`、`.cache/`、`.tools/`、`.superpowers/`和Python缓存保留，编辑器默认隐藏这些目录；原数据与模型在文件树中仍可访问。
- 编辑器搜索及文件监视排除环境、权重、原始数据和归档，避免扫描大量生成文件。
- `.gitignore`补齐v3大工件、环境及传输包的规则。忽略仅影响未来Git收录，不删除磁盘文件。当前目录没有 `.git`，本次没有创建仓库或提交。
- 原 `package_full.py` 和 `package_project.py`针对旧交接布局编写，不按新 `.gitignore`筛选；不要把它们当作当前轻量交付命令。只分享当前成品可复制 `report/research-v3/`；复现还需要模型、数据及原证据。
- 整理前的Python虚拟环境已无法找到基础解释器，本次没有修复或移动环境；因此没有重新运行Python功能测试。

## 核验记录

本次基线与结果在 `.superpowers/repository-tidy-20261108/`：`before.json`保存科研文件大小/修改时间及证据SHA，`verification.json`保存整理后的核验结果，另保留更新前的入口文件。九项成品的来源在[成品清单](../report/research-v3/deliverables-manifest.json)。

321份受保护文件的SHA256、3,656份科研文件的大小与修改时间、九项成品的来源与副本SHA，以及24份入口文档的258个本地链接均已核验。恢复脚本已对18个实际归档包执行只读SHA校验；另用小文件验证只读模式、实际恢复、重复运行，以及拒绝不同内容覆盖和两种路径越界。

模型计数保持原口径：全部v3完成记录428，当前研究232，有V指标226，正式Q模型42；2条中断不计完成。归档与报告副本均不新增训练数量。

[回到首页](../README.md)
