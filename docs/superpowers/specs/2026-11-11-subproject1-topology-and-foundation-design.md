# 子项目 1 设计：本地整理与 `src/lftc/` 工程基础

**日期**：2026-11-11  
**所属阶段**：整体重构的子项目 1 / 4  
**实施依据**：原 Clean-Core 分层架构，以及 2026-11-11 用户补充要求。

> 后续子项目 2—4 已完成本地实现与整合。本文件保留子项目 1 的设计范围；当前整体实现见 [工程架构](../../ARCHITECTURE.md) 与 [使用说明](../../USAGE.md)。

## 1. 目标与范围

将多代实验脚本、结果、调试代码和历史文档从活跃工程中收起，建立统一的 `src/lftc/` 工程基础。

用户已明确：

- GitHub 整理属于整体范围，先完成本地，用户审阅后再提交。本次不初始化 Git、不提交、不推送。
- 旧系统保留用于查阅旧数据、代码和结果，无需继续运行。
- 直接迁移，遇到实际问题修复后继续。不增加映射文档、迁移状态机或回滚协议。
- 不新建资产 lock，不计算哈希，不建立额外清单核验系统；旧 lock 仅随旧材料保留。
- 文档链接随新位置更新，不要求历史文档原文冻结。
- 原始数据、底座模型和历史产物保留。环境、工具与项目外资料不移动。

整体按四个有序子项目推进：

| 子项目 | 内容 | 当前状态 |
| --- | --- | --- |
| 1 | 本地拓扑整理、基础包、配置与最小 CLI | 已完成 |
| 2 | 数据、特征、模型、训练和实验引擎 | 待实施 |
| 3 | 评估、机制分析、统计与报告 | 待实施 |
| 4 | 统一前端、集群与构建工具 | 待实施 |

本子项目不重写训练、评估和前端逻辑，不重训模型，不维护旧系统运行兼容。

## 2. 目录结构

```text
sentiment-project/
├── pyproject.toml
├── README.md
├── src/lftc/
├── configs/
│   ├── paths.yaml
│   ├── assets.yaml
│   └── experiments/
├── apps/
├── ops/
├── tests/unit/
├── docs/
│   ├── archive/2026-11-08.md
│   └── superpowers/
├── artifacts/
│   ├── assets/
│   │   ├── datasets/
│   │   └── models/
│   ├── prepared/
│   ├── runs/
│   ├── results/
│   └── reports/
└── archive/2026-11-08/
```

归档日期沿用原设计的历史研究阶段标记，实际整理日期为 2026-11-11。

原始数据和底座模型集中到 `artifacts/assets/`。旧根目录脚本、配置、前端、依赖清单进入归档中的 `legacy-root/`；`research_v3/`、移出原始资产后的 `research-v3/`、旧测试、各代数据/模型/结果、报告和开发辅助材料进入归档，保留各自目录名称。

旧 `configs/research-v3.json` 同步归档。旧文档进入归档，`docs/superpowers/` 留在原位，相关 Markdown 链接按新位置更新。[历史资料入口](../../archive/2026-11-08.md)提供查阅导航。

旧实验缓存进入归档，`.cache/matplotlib/` 保留。`.venv*`、`.tools/`、`.vscode/`、`.superpowers/` 保留原位。项目旁的 PPT、录音文稿和 `sentiment-project-archive/` 不动。

## 3. 包架构

新代码统一放入 `src/lftc/`。高层依赖低层，禁止循环导入，禁止导入归档和旧模块。

| 层 | 模块 | 本阶段职责 |
| --- | --- | --- |
| 0 | `domain` | 共享数据类型、任务类型和异常，仅依赖标准库 |
| 1 | `governance` | 公共路径处理，依赖 `domain` 和标准库 |
| 1 | `config` | 路径与资产注册表，依赖 `domain`、`governance` 和 PyYAML |
| 2 | `data`、`features`、`models`、`evaluation` | 仅保留职责说明 |
| 3 | `training`、`reporting` | 仅保留职责说明 |
| 4 | `experiments`、`cli` | 工作流槽位与基础命令行 |

### 3.1 基础类型与接口

- `TaskType`：`single_label`、`multilabel`、`pair_classification`。
- `WorkspacePaths`：`root`、`config_dir`、`artifacts_dir`、`assets_dir`、`prepared_dir`、`runs_dir`、`results_dir`、`reports_dir`。
- `DatasetAsset`：`id`、`directory`、`repo`、`revision`、`files`、`task`、`label_count`、可选 `label_names`。
- `ModelAsset`：`id`、`directory`、`repo`、`revision`、`files`。
- `AssetsCatalog`：数据集与模型元数据集合。
- 异常：`LftcError`、`ConfigError`、`PathBoundaryError`。
- 公共函数：`bounded_path(candidate, root)`、`resolve_project_root(start=None)`、`load_paths_config(root=None)`、`load_assets_config(config_path)`。

本阶段不建立资产锁、哈希函数或全量资产核验接口。

### 3.2 包声明与依赖

`pyproject.toml` 使用 setuptools 与 `src/` 布局，包名 `lftc`，Python 要求 `>=3.11`，基础运行依赖仅 `PyYAML>=6.0`。`ml`、`train`、`ui`、`dev` 可选组沿用原设计，具体版本以该文件为准。

控制台入口为 `lftc = "lftc.cli:main"`，也支持 `python -m lftc`。本机现有研究环境用相对路径 `lftc-local.pth` 登记 `src/`，不联网安装依赖。新环境使用 `python -m pip install -e .`。

## 4. 配置与命令行

### 4.1 路径

```yaml
schema_version: 1
artifacts_dir: artifacts
assets_dir: assets
prepared_dir: prepared
runs_dir: runs
results_dir: results
reports_dir: reports
```

`artifacts_dir` 相对工程根目录，其余产物子目录相对 `artifacts_dir`。

- `LFTC_ROOT` 指定工程根目录；否则从当前目录向上查找 `project.name = "lftc"` 的 `pyproject.toml`。
- `LFTC_CONFIG_DIR` 指定配置目录，该目录中的 `paths.yaml` 与默认 `assets.yaml` 配套使用。
- `LFTC_ARTIFACTS_DIR` 指定产物根目录，所有产物子目录随之切换。
- 环境变量覆盖支持绝对路径或相对工程根目录的路径；YAML 采用相对路径。
- CLI 的全局参数 `--root` 可显式指定工程根目录。

### 4.2 资产注册表

`configs/assets.yaml` 保存既有来源与目录元数据，不包含哈希清单。

- 数据集：`imdb`（2 类）、`rotten_tomatoes`（2 类）、`yelp`（2 类）、`ag_news`（4 类）、`mnli`（3 类）、`banking77`（77 类）、`go_emotions`（28 类）、`yahoo_answers_topics`（10 类）。
- MNLI 为 `pair_classification`，GoEmotions 为 `multilabel`，其余为 `single_label`。
- 底座模型：`distilbert` 和 `modernbert`。
- `directory` 相对 `artifacts/assets/`，`files` 相对各资产目录；repo、revision 和已有类别名称沿用原始元数据。

### 4.3 最小 CLI

- `python -m lftc env`：输出工程路径、Python 版本和可选依赖可用性。
- `python -m lftc assets list [--config ...]`：输出资产注册表，不加载 parquet 或权重，不计算哈希，不联网。
- 成功时输出 JSON、返回 0。配置或路径错误时向 stderr 输出 JSON、返回 1。

## 5. 本地交付

仅运行新工程的轻量功能测试和两个基础 CLI 命令，不执行旧测试、全量哈希、链接扫描或科研重训。

```powershell
.\.venv-research\Scripts\python.exe -m unittest discover -s tests -v
.\.venv-research\Scripts\python.exe -m lftc env
.\.venv-research\Scripts\python.exe -m lftc assets list
```

当前 README 和归档索引提供新工程与历史资料入口。`archive/`、`artifacts/`、本地环境、缓存和工具由 `.gitignore` 隔离，编辑器搜索与监视排除归档和大文件。

先交付本地供用户查看，用户审阅后再进行 GitHub 整理与提交。
