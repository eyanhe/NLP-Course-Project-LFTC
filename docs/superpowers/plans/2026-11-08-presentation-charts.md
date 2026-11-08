# PPT展示图实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 将现有研究图重画为投影可读、可编辑的18页PPT图表素材。

**Architecture:** 规范生成器在development-support/research-v3；输出至outputs/research-ppt-20261108，构建/核验在.build，交付在files，单图在images。使用现有Artifact Tool，229源SHA先验校验，原结果路径不动。

**Spec:** `docs/superpowers/specs/2026-11-08-presentation-charts-design.md`

## Tasks
- [ ] 实现原生图表版式，先检查代表页，再生成18页。
- [ ] 逐页检查与修正，PPTX结构/图表数据/字号/可编辑性验证，导出高清PNG。
- [ ] 补阅读索引和当前入口；核验来源、保护文件及链接，完成独立只读审查。

## Review focus
- 中文字体/图例/数据标签投影可读，无碰撞和裁切。
- 分组、量纲、历史/新模型与V/D/Q来源准确。
- S1预算含调度变化，M2是V机制，S4为AI审核子集。
- 全部主题保留，拆页不是新增实验。
- 成本显示实际时长，批均摊不称单请求延迟。
