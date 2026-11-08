# 按原交接格式整理：核验记录

2026-11-08（SGT）。本次为文件组织与文档整理；由实施者核验，未声称独立双人审查。

- 主目录89.17 → 59.66 GiB，减少约33.09%。18个tar包共29.51 GiB移到相邻归档；未删除文件，未释放同等磁盘空间。
- `report/research-v3/`集中九项成品。两份Markdown仅重定位相对链接；七项二进制工件与原文件SHA一致；来源路径与SHA记入成品清单。
- 321份证据SHA核验通过，覆盖原252份保护清单与当前分析/图册/逻辑图来源。3,656份科研文件大小与修改时间未变；这部分是元数据核验，不等同于对所有大权重重新计算SHA。
- 24份文档的258个本地链接有效。训练计数仍为当前232、V 226、Q 42；全部v3完成记录428。
- 每个传输包均做移动前后SHA校验。用户侧恢复脚本另以 `-VerifyOnly`校验18个实际包并返回成功。
- 小文件恢复检查通过：只读不移动、恢复后内容哈希一致、重复运行跳过；不同内容覆盖、目标路径越界、来源路径越界均拒绝且不移动源文件。小文件检查结果在 `.superpowers/repository-tidy-20261108/restore-checks/result.json`。
- Windows执行策略仅对核验进程使用 `-ExecutionPolicy Bypass`，没有更改系统执行策略。现有虚拟环境无法找到基础Python，是整理前状态；没有运行Python功能测试，也没有修改Python研究实现。

完整机器记录：`.superpowers/repository-tidy-20261108/before.json`与`verification.json`。移动及成品清单分别位于`docs/repository-archive-manifest.json`和`report/research-v3/deliverables-manifest.json`。
