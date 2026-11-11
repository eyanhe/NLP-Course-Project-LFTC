# 开发与复现支持

日常阅读从[当前研究成品](../report/research-v3/README.md)或[仓库首页](../legacy-root/README.md)进入。本目录保存构建、检查和维护工具。

| 路径 | 用途 |
| --- | --- |
| [research-v3/](research-v3/README.md) | 当前Excel、PPT构建及原整理核验 |
| [repository/restore-transfer-packages.ps1](repository/restore-transfer-packages.ps1) | 校验或恢复移到相邻归档目录的18个tar包 |
| `build_deck*.mjs`、`render_final_deck.mjs`、`deck-*/` | 原课程交接PPT的构建器、候选及预览 |
| `check_*.py`、`check_*.mjs`、`pdf-review/` | 原GUI、PPT和页面的开发检查 |

传输包的目录、大小、校验值与恢复命令见[仓库布局](../docs/REPOSITORY-LAYOUT.md)。本次文件整理不改变模型或训练代码，原构建器的运行时要求继续沿用各自说明。
