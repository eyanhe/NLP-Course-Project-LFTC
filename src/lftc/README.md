# 核心包

新功能统一放入 `lftc`，不从 `archive/` 导入旧代码。

| 层 | 模块 | 职责 |
| --- | --- | --- |
| 0 | `domain` | 共享数据类型、任务类型和异常 |
| 1 | `governance` | 公共路径处理 |
| 1 | `config` | 路径和资产注册表；可依赖 `governance` |
| 2 | `data`、`features`、`models`、`evaluation` | 数据、表示、学习器和评估 |
| 3 | `training`、`reporting` | 训练与报告组织 |
| 4 | `experiments`、`cli` | 工作流和命令行入口 |

依赖从高层指向低层，禁止循环导入。各层已接通，具体职责和数据流见 [工程架构](../../docs/ARCHITECTURE.md)，命令和配置见 [使用说明](../../docs/USAGE.md)。
