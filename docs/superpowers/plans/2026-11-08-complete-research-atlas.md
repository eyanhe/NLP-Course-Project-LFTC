# 完整研究图册实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 把现有研究证据补成11张可独立阅读、可追溯的主题图和统一PDF。

**Architecture:** 独立图册生成器读取既有分析、计时与训练状态，不覆盖原始证据。每页记录数据和来源哈希，生成PNG/PDF与阅读指南，再更新导航。

**Tech Stack:** 已有Python、NumPy、Matplotlib、Poppler。

**Spec:** `docs/superpowers/specs/2026-11-08-complete-research-atlas-design.md`

## Global Constraints
- 仅处理已有数据，不训练、不推理、不SSH，不改冻结输入。
- 保护原252文件和198分析输入；不改原analysis-data.json。
- 11页为11个研究主题，不把格式副本或渲染预览算作新图。
- 实测成本不包含完整超参数搜索；计时、种子及证据集口径明确。
- 无可用Git，增量文件与哈希记录作为可回溯证据。

## Review Focus
- 旧固定预算V与新优化Q不可拼为同一学习曲线。
- GoEmotions AP与完整标签集合对错不能混为一项指标。
- 宏观改善与尾类退化同时呈现，冻结表示不作架构总排名。
- 训练时间重建和共享CPU噪声不能伪装成精确硬件基准。
- 字体、标签、案例文本和页脚不得重叠或超出版面。

## Tasks
- [x] 编写图册生成器：数据读取/指纹、11主题图、逐页图解和源数据。
- [x] 生成并核对11页与11张PNG，数值与来源校验、逐页渲染检查。
- [x] 更新仓库和研究图表入口，检查链接与保护哈希，完成只读审查。
