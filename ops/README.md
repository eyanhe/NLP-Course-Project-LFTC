# 日常运行与源码交付

仅保留日常实验入口、Slurm 模板和源码打包器。请从项目根目录运行，先安装 `python -m pip install -e ".[ml]"`；神经模型使用相应训练环境。

```powershell
./ops/run.ps1 -Config configs/experiments/quickstart.yaml -Python ./.venv-research/Scripts/python.exe
python -m lftc report --run-id <run_id>
python -m lftc report --history
python ops/package_source.py --output artifacts/delivery/lftc-source.zip
```

```bash
bash ops/run.sh configs/experiments/quickstart.yaml
LFTC_PYTHON=/path/to/venv/bin/python bash ops/run.sh my-config.yaml
sbatch ops/slurm.sbatch configs/experiments/quickstart.yaml
```

`run.ps1` 的 `-ArtifactsDir` 或环境变量 `LFTC_ARTIFACTS_DIR` 可指定独立输出目录；`LFTC_CONFIG_DIR` 指定路径/资源配置目录。Slurm 模板继承这些环境变量，以提交目录作为默认项目根目录。请按集群要求补充 account、partition、GPU 资源，并提前准备离线数据和模型。脚本不自动下载、重训或扫描归档。

源码 ZIP 包含 `src/`、`configs/`、`apps/`、`ops/`、`docs/`、`tests/`、`README.md`、`pyproject.toml` 和 `.gitignore`；保留 `docs/archive/` 中的导航索引，排除根目录历史归档、实验产物、Python 环境、工具下载、node_modules 与缓存。不会跟随符号链接。交付包中的历史设计文档可能引用仅本地保存的归档；需要历史资料或模型时请独立传递，不包含在源码包中。
