# 本地工程整理 Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task-by-task. 用户已要求直接执行，完成后留在本地，等待用户审阅再处理 GitHub。

**Goal:** 完成子项目 1 的本地目录整理和轻量工程基础。

**Architecture:** 沿用 `src/lftc/` 分层架构。旧系统仅作数据和历史参考，不再保持运行兼容；原始数据和底座模型集中在 `artifacts/assets/`。

**Tech Stack:** Python 3.11、PyYAML、setuptools；沿用本地虚拟环境。

**Spec:** [子项目 1 设计](../specs/2026-11-11-subproject1-topology-and-foundation-design.md)，以 2026-11-11 用户补充要求为准。

## Global Constraints

- 先整理本地，用户看过后再提交 GitHub；本次不初始化 Git、不提交、不推送。
- 旧内容保留；旧代码无需继续运行。
- 不新增资产 lock、哈希核验、迁移状态机、回滚协议或清单校验系统。
- 直接迁移已有文件，遇到实际问题就修正；不再产出处置映射文档。
- 更新 Markdown 链接和阅读入口，不要求历史文档原文冻结。
- 环境、工具和工作区外的文件保持原位。
- 新增功能仅限基础包、路径和资产配置、最小 CLI；实验引擎和前端留待后续子项目。

## Review Focus

- 文件移动限定在当前 `sentiment-project/` 中，不覆盖已有目标。
- `LFTC_ARTIFACTS_DIR` 覆盖后子目录跟随它；`LFTC_CONFIG_DIR` 决定配置目录。
- 底座模型与数据集使用各自的元数据字段。
- CLI 可在现有环境使用；不强制安装 ML 或前端依赖。
- 只做轻量功能检查，不读取大文件计算哈希，不运行历史测试。

## Task 1: 整理目录与阅读入口

**Files:** 旧根目录内容、`archive/2026-11-08/`、`artifacts/assets/`、`configs/assets.yaml`、旧 Markdown 链接。

**Interfaces:** 生成数据集与底座模型注册表，路径相对 `artifacts/assets/`。

- [x] 将原始资产直接迁至新目录。
- [x] 将旧源码、配置、结果、报告、调试及构建工具直接归档。
- [x] 更新既有 Markdown 相对链接。

## Task 2: 建立轻量包

**Files:** `pyproject.toml`、`src/lftc/`、`configs/paths.yaml`、`tests/`。

**Interfaces:** `resolve_project_root()`、`load_paths_config()`、`load_assets_config()`、`python -m lftc env`、`python -m lftc assets list`。

- [x] 为路径覆盖与 CLI 行为编写少量测试，确认缺失功能时失败。
- [x] 实现 domain、governance、config 和 CLI，建立其余架构槽位。
- [x] 在现有 Python 环境运行新测试与 CLI；不安装新依赖。

## Task 3: 整理文档与交付

**Files:** `README.md`、`docs/archive/2026-11-08.md`、当前设计、`.gitignore`、`.vscode/settings.json`。

**Interfaces:** 主 README 指向新工程、历史研究报告和归档索引。

- [x] 将用户补充要求写入当前设计，移除已否定的复杂校验设计。
- [x] 更新项目入口、归档索引、忽略与编辑器设置。
- [x] 查看整理后的顶层结构，报告本地结果并等待用户审阅。

## 执行结果

- 本地归档完成；旧演示服务停用后，开发工具和日志也已迁入归档。
- 8 套数据集与 2 套底座模型集中到 `artifacts/assets/`；仅整理目录和既有元数据，未执行哈希核验。
- 新包、配置、CLI、入口文档和忽略规则已建立；现有研究环境使用相对路径 `.pth` 登记源码。
- `python -m unittest discover -s tests -v`：8 项通过；`env` 与 `assets list` 正常。
- 未初始化 Git、提交或推送，等待用户查看本地。
