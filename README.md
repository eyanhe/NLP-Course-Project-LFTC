# LFTC Text Classification Research

**English** | [简体中文](docs/README.zh-CN.md)

The refactored project uses `src/lftc/` for new experiments, evaluation reports, and the interactive presentation app. Historical code and research results are preserved in `archive/2026-11-08/`.

The [GitHub repository](https://github.com/eyanhe/NLP-Course-Project-LFTC) contains source code, historical reports, figures, presentations, Excel workbooks, and result data. The [Hugging Face asset repository](https://huggingface.co/datasets/Eyanhe/LFTC_Dataset_And_Checkpoints/tree/5fd879463414b2059ae05b4e188da58e706456dd) stores original datasets, features, and model weights. Existing Git history is preserved, with historical materials moved to their refactored archive locations.

## Quick start

Run from the project root:

```powershell
# Open the interactive app based on the original course presentation
.\.venv-research\Scripts\python.exe -m lftc app
```

You can also run `.\apps\start.ps1`, which prefers the local `.venv-research` environment.

The app follows the presentation's English narrative, navy and cream palette, and three research questions. Its seven pages cover overview, performance and cost, robustness, mechanisms, live prediction, methods and evidence, and conclusions. Charts read the original research results. Live prediction compares the five original models across three review datasets, AG News, and MultiNLI sentence pairs. The original presentation is available to download from the sidebar. See the [app guide](apps/README.md) for page details and model sources.

Use the CLI below for new experiments. The existing `quickstart-local` run demonstrates the workflow with 200 training examples; it supplies neither research results nor prediction models to the presentation app.

```powershell
# Run a small experiment and save the model, per-example predictions, and metrics
.\.venv-research\Scripts\python.exe -m lftc run --config configs/experiments/quickstart.yaml --run-id my-first-run

# Predict with the saved model and generate a report
.\.venv-research\Scripts\python.exe -m lftc predict --run-id my-first-run --text "A thoughtful and enjoyable film."
.\.venv-research\Scripts\python.exe -m lftc report --run-id my-first-run
```

## Documentation

- [Usage](docs/USAGE.md): experiment configuration, commands, models, and output files.
- [Architecture](docs/ARCHITECTURE.md): layers, responsibilities, and data flow.
- [Historical materials](docs/archive/2026-11-08.md): original reports, models, data, and legacy code.
- [Asset registry](configs/assets.yaml): sources and directories for eight datasets and two base models.
- [Operations](ops/README.md): cluster templates, run wrappers, and source packaging.

## Capabilities

- **Data and experiments:** eight local Parquet datasets; single-label, multilabel, and sentence-pair classification; training sample selection, dataset size, few-shot and long-tail conditions; dataset/model/seed/condition combinations.
- **Models:** TF-IDF + logistic regression, TF-IDF + SVM, Word2Vec + SVM, BiLSTM, DistilBERT, and ModernBERT. Features are learned from training data only, and Transformers use local models.
- **Evaluation and analysis:** F1 over the full label space, AP, confusion matrices, validation-based threshold selection, paired bootstrap, McNemar tests, Holm correction, text-length analysis, and error diagnosis.
- **Reports:** Markdown, JSON, CSV, and statistical plots for new experiments; data and original deliverable exports for historical research.
- **Interactive presentation:** original results, costs, and mechanism examples organized around the three research questions; prediction comparisons across five original models; presentation and report downloads.
- **Engineering tools:** Windows/Linux run scripts, a Slurm template, and a source ZIP packager that excludes local large files and runtime archives.

## Repository layout

```text
sentiment-project/
├── pyproject.toml          Package metadata and optional dependencies
├── src/lftc/               Active Python core package
├── configs/                Paths, assets, and experiment configurations
├── apps/                   Streamlit presentation app and bundled materials
├── ops/                    Run scripts, cluster tools, and source packaging
├── tests/                  Tests for the refactored project
├── docs/                   Usage, architecture, and historical navigation
├── artifacts/              Local assets, new models, results, and reports
└── archive/2026-11-08/     Historical code, experiment outputs, and documents
```

Local `.venv*`, `.tools/`, `.cache/`, and `.superpowers/` directories stay in place and are hidden by the editor. Archives and generated artifacts are excluded from default searches. Presentations, transcripts, and archived transfer packages alongside the project remain in their original locations.

## Environment and paths

The existing local `.venv-research` environment already includes `src/` on its Python path. For a new environment, install the extras you need:

```powershell
python -m pip install -e .           # Base CLI and configuration
python -m pip install -e ".[ml]"     # Traditional models, evaluation, and reports
python -m pip install -e ".[train]"  # Neural networks
python -m pip install -e ".[ui]"     # Presentation app
```

For CUDA, choose a PyTorch build compatible with the cluster environment. Original datasets and base model weights are stored locally and are excluded from the source package.

The pinned Hugging Face revision retains the paths used before the refactor. Restore only the assets you need, using the mapping below to place files in the current project:

| Path in the asset repository | Current project location |
| --- | --- |
| `artifacts/assets/distilbert/` | `artifacts/assets/models/distilbert/` |
| `artifacts/research-v3/assets/modernbert/` | `artifacts/assets/models/modernbert/` |
| `artifacts/assets/<dataset>/` | `artifacts/assets/datasets/<dataset>/` |
| `artifacts/research-v3/assets/<dataset>/` | `artifacts/assets/datasets/<dataset>/` |
| Other historical assets under `artifacts/models*/`, `artifacts/data*/`, or `artifacts/research-v3/` | The corresponding original relative path under `archive/2026-11-08/`, after removing the outer `artifacts/` prefix |

For example, restore `artifacts/models/imdb/linear_svm/model.joblib` to `archive/2026-11-08/models/imdb/linear_svm/model.joblib`. Browsing research charts does not require model weights. Live prediction needs the corresponding historical checkpoints, while new experiments need their datasets and base models. The historical publication manifest is available in the [archived publishing guide](archive/2026-11-08/docs/publishing/README.md).

In `configs/paths.yaml`, `artifacts_dir` is relative to the project root, and the other artifact directories are relative to `artifacts_dir`. Paths can be overridden through `LFTC_ROOT`, `LFTC_CONFIG_DIR`, and `LFTC_ARTIFACTS_DIR`, or the global `--root` option.

## Project status

Local implementation and integration of all four subprojects are complete. The app presents the original research following the course slides, while new experiments remain available through the CLI. Example runs are kept separate from the original experiment counts. Legacy application entry points are no longer maintained; one-off debugging and plotting scripts are preserved in the archive.

The full test suite passed before publication: `python -m unittest discover -s tests -v`, 53 tests passed. Historical experiments were not rerun, and no new asset lock or large-file hash verification was performed.
