# Complete local restructure implementation plan

> Use executing-plans for integration and dispatching-parallel-agents for the three independent implementation areas. The user has requested continuous execution; no further design approval is required.

**Goal:** Complete subprojects 2–4 locally, retaining useful experiment capabilities in the layered package and leaving GitHub untouched for user review.

**Architecture:** Keep source under `src/lftc/`, configuration under `configs/`, one Streamlit application under `apps/`, and maintained operational tools under `ops/`. Reuse relevant algorithms from archived code without importing archived modules. Historical data and completed reports remain readable as data. Do not restore one-off drivers or rigid historical execution gates.

**Tech stack:** Python 3.11, existing numpy/sklearn/scipy/pandas/pyarrow/joblib/PyTorch/Transformers/Streamlit environments.

**Spec:** [Approved topology and layered architecture](../specs/2026-11-11-subproject1-topology-and-foundation-design.md), extended by the user's instruction to finish all remaining subprojects.

## Constraints

- Local work only; no Git initialization, commits, remote writes or publishing.
- Preserve archived materials; no full retraining, bulk artifact hashing, asset-lock generation or migration state machinery.
- Support single-label, multilabel and paired-text tasks across all eight registered datasets.
- Preserve TF-IDF LR/SVM, Word2Vec SVM, BiLSTM, DistilBERT and ModernBERT model families.
- Optional ML/UI libraries are loaded only by commands that use them.
- New outputs go to configured `artifacts/` directories; historical results are never overwritten.
- Use small behavior tests, one tiny local end-to-end run, and a Streamlit smoke check. No expensive test repetition without a failure or new change.

## Shared contracts

`lftc.domain.Example(sample_id, text, labels, split, text_pair=None, dataset_id='')` stores labels as a tuple of integer indices.

`PreparedDataset(dataset_id, task, label_count, label_names, train, validation, test)` uses tuples of Example and a TaskType task.

Engine public APIs:

- `data.load_dataset(asset, assets_dir, *, limit_per_split=None, validation_fraction=0.15, seed=42) -> PreparedDataset`
- `data.save_prepared(dataset, directory)` / `data.load_prepared(directory)`
- `data.sample_training(examples, *, mode='all', size=None, k=None, imbalance_ratio=10.0, seed=42) -> tuple[Example,...]`
- `training.train_model(train, validation, task, label_count, model_id, options=None, model_dir=None) -> Predictor`
- Predictor exposes `model_id`, `task`, `label_count`, `score_kind`, `predict_scores(examples, batch_size=32)` and `predict(examples, threshold=None)`.
- `training.save_model(predictor, directory)` / `training.load_model(directory)`

Evaluation/report APIs:

- `evaluation.metric_bundle(labels, scores, task, score_kind, threshold=None, label_mask=None) -> dict`
- `evaluation.select_threshold(labels, scores, score_kind='probability', candidates=None) -> float`
- `evaluation.bootstrap_difference(...)`, `evaluation.mcnemar_test(...)`, `evaluation.holm_adjust(...)`
- `evaluation.diagnose_predictions(examples, scores, task, label_count, score_kind, threshold=None) -> dict`
- `reporting.load_history(root) -> dict`, containing counts, q_records, primary and available mechanism/case/figure/download data.
- `reporting.list_runs(paths) -> list[dict]`
- `reporting.build_report(run_dir, output_dir=None) -> dict` of generated file paths.
- `reporting.build_history_report(root, output_dir) -> dict` exports the existing completed research, clearly labeled as historical.

Run layout: `artifacts/runs/<run_id>/run.json`, `metrics.json`, `predictions.jsonl`, `model/`; reports go under `artifacts/reports/<run_id>/`.

`run.json` fields: schema_version, run_id, dataset, task, label_count, label_names, model, seed, training_rows, validation_rows, test_rows, config, metrics, threshold, timing, created_at.

Prediction rows: sample_id, split, text, text_pair, labels, prediction (list of label indices), scores, score_kind.

## Tasks and ownership

- [x] Engine: implement `data/`, `features/`, `models/`, `training/` and focused tests. Pair inputs retain both texts. Features fit training rows only. Handle absent labels and empty lexical features. Transformer operations remain offline; no large training run.
- [x] Evaluation and reporting: implement `evaluation/`, `reporting/` and focused tests. Use full declared label space, correct score semantics, validation-only threshold selection, paired statistics, diagnostics, CSV/Markdown/JSON/figure exports and historical research readers.
- [x] Frontend and tools: implement one `apps/streamlit_app.py`, launchers, source packaging and a usable Slurm template under `ops/`. Read current runs and historical results, allow prediction with new saved models, provide downloads, handle an empty fresh checkout.
- [x] Integration: root agent owns shared domain types, experiment configuration loading, `experiments/`, CLI, example configs, documentation and integration/architecture tests. Provide prepare, matrix, train/run, evaluate, predict, diagnose, report, history, runs and app commands.
- [x] Validate and document: run all new tests, a tiny real TF-IDF experiment on local data, historical report export and Streamlit smoke checks; update README and progress, leave all changes local.

## Review focus

Class order, paired inputs, multilabel score thresholds, unseen labels and train-only feature fitting; output directory overrides; serialization and reload; historical and newly produced results must remain distinguishable.

## Progress

- Initial state: subproject 1 complete; no Git repository; existing Python 3.11.9 research environment available.

## Implementation and validation results

- Engine modules implemented without legacy imports; all eight registered datasets loaded using bounded smoke reads. Pair fields and full declared label dimensions are preserved.
- Tiny BiLSTM, DistilBERT and ModernBERT train/save/reload checks use synthetic local model configurations. Device auto selection and CUDA-to-CPU reload are covered. No historical checkpoints were retrained.
- Actual local example `quickstart-local` uses a random sample of 200 Rotten Tomatoes training rows and the full 1,066-row validation/test partitions. The earlier physical-head smoke sample was kept under ignored `.superpowers/refactor-smoke/`, outside active results, because a class-ordered split was unrepresentative.
- Prepare, 24-recipe baseline matrix, predict, evaluate, diagnose, paired compare, runs, history, new-run report and historical export commands all executed successfully.
- JSONL readers preserve Unicode U+0085/U+2028 inside JSON strings; text and score CSV roundtrips are covered.
- Consolidated command: `.venv-research/Scripts/python.exe -B -m unittest discover -s tests -v` → **42 tests passed**.
- Six Streamlit pages, empty-checkout behavior and actual saved paired-model prediction are exercised by AppTest. Local server is available at http://127.0.0.1:8501; health endpoint returns `ok`.
- Source delivery is available under `artifacts/delivery/lftc-source.zip`, including tests and the archive navigation index while excluding runtime archives/assets/environments.
- Slurm/Linux templates are provided; no cluster job was submitted from this Windows machine. No Git initialization, commit, push, bulk hashing or asset-lock generation was performed.

## Final review and fixes

- Fresh read-only review found the Word2Vec dependency missing from the traditional-model extra; moved gensim to `ml` so a clean traditional installation supports all three families.
- Fixed prepared-output collisions between different validation fractions, and sorted the recent-run list by creation time. Both behaviors now have regression coverage.
- Included historical Excel workbooks in report/download exports; all three original workbooks are available.
- Final consolidated suite after fixes: **42 tests passed** in 10.395 seconds. Streamlit health returns `ok`.
- User-defined exclusions remain unchanged: historical protocol parity/execution, full scientific retraining, hash/lock/manifest work, migration machinery, Git writes and actual cluster submission are outside this local refactor delivery.
