# 实验配置

这里保存统一实验配方：`quickstart.yaml` 提供 200 条训练样本的入门实验，`baselines.yaml`、`neural.yaml`、`data_conditions.yaml` 提供研究运行模板。

先用 `python -m lftc matrix --config <配置文件>` 查看组合，再用 `run --config` 执行。研究模板可能耗时较长；它们不会自动执行，也不自动等同于原研究的固定协议。

参数见 [使用说明](../../docs/USAGE.md)。旧研究配置仍可从 [research-v3.json](../../archive/2026-11-08/legacy-root/configs/research-v3.json) 查阅。
