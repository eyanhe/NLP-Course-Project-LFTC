"""Read the original presentation's preserved evidence without running experiments.

Deep-model summaries average the saved seeds. Bootstrap dictionaries retain their
separate fixed-model scope; they are never substituted for training uncertainty.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
MODELS = ['logistic_regression', 'linear_svm', 'word2vec_svm', 'bilstm', 'distilbert']
MODEL_NAMES = {
    'logistic_regression': 'TF-IDF + LR', 'linear_svm': 'TF-IDF + SVM',
    'word2vec_svm': 'Word2Vec + SVM', 'bilstm': 'Word2Vec + BiLSTM',
    'distilbert': 'DistilBERT', 'tfidf_svm': 'TF-IDF + SVM',
}
DATASETS = {'imdb': 'IMDb', 'rotten_tomatoes': 'Rotten Tomatoes',
            'yelp': 'Yelp', 'ag_news': 'AG News'}
COLORS = {'logistic_regression': '#B4471A', 'linear_svm': '#D18A62',
          'word2vec_svm': '#A9ADB1', 'bilstm': '#C9638C', 'distilbert': '#2A78D6'}
METRICS = {'macro_f1': 'Macro-F1', 'accuracy': 'Accuracy',
           'macro_precision': 'Macro precision', 'macro_recall': 'Macro recall',
           'mcc': 'MCC', 'minority_recall': 'Minority recall'}
CONDITIONS = {'balanced': '25%', 'mild': '70%', 'moderate': '85%', 'severe': '94%'}

_SCORE_COLUMNS = ['accuracy', 'macro_precision', 'macro_recall', 'macro_f1', 'mcc']
_IMBALANCE_COLUMNS = [
    *_SCORE_COLUMNS, 'minority_recall', 'minority_f1', 'worst_class_recall',
    'simulated_prior_weighted_accuracy',
]
_COST_COLUMNS = {'source': 'object', 'model': 'object', 'device': 'object',
                 'batch_size': 'int64', 'threads': 'int64',
                 'median_ms': 'float64', 'p95_ms': 'float64'}


def _archive_root(root: Path) -> Path:
    archive = root / 'archive/2026-11-08'
    if archive.is_dir():
        return archive
    # Accept a direct preserved-evidence directory, as the original reader did.
    if any((root / name).is_dir() for name in ('results-v2', 'results-hard', 'results')):
        return root
    return archive


def _json(path: Path) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding='utf-8-sig'))


def _empty_frame(schema: dict) -> pd.DataFrame:
    return pd.DataFrame({name: pd.Series(dtype=dtype) for name, dtype in schema.items()})


def _csv(path: Path, schema: dict) -> pd.DataFrame:
    return pd.read_csv(path) if path.is_file() else _empty_frame(schema)


def _deduplicate_costs(rows: list[dict]) -> list[dict]:
    keys = ('source', 'model', 'device', 'batch_size', 'threads')
    # The v2 export repeats v1 measurements. Retain its last saved record per
    # configuration, without collapsing different measurement configurations.
    unique = {}
    for row in rows:
        unique[tuple(row.get(key) for key in keys)] = row
    return list(unique.values())


def load_presentation_data(root: str | Path | None = None) -> dict:
    """Read preserved presentation data from a project or direct archive root.

    Missing files yield empty containers with known table columns. Invalid files
    still raise their normal read errors so corrupt evidence is not hidden.
    ``available`` means original evaluation rows are present, not quickstart runs.
    """
    archive = _archive_root(Path(root) if root is not None else ROOT)
    evaluation = (
        _json(archive / 'results-v2/evaluation.json').get('rows', [])
        + _json(archive / 'results-hard/evaluation.json').get('rows', [])
    )
    benchmark = _deduplicate_costs(
        _json(archive / 'results/benchmark.json').get('rows', [])
        + _json(archive / 'results-v2/benchmark.json').get('rows', [])
    )
    transfer_schema = {'source': 'object', 'target': 'object', 'model': 'object',
                       'seed': 'float64', 'test_n': 'int64',
                       'macro_f1': 'float64', 'accuracy': 'float64'}
    imbalance_schema = {'condition': 'object', 'model': 'object',
                        **dict.fromkeys(_IMBALANCE_COLUMNS, 'float64'),
                        **dict.fromkeys(['macro_f1_training_sd', 'macro_f1_drop_pp',
                                         'minority_recall_drop_pp',
                                         'macro_f1_drop_training_sd_pp'], 'float64')}
    mitigation_schema = {'model': 'object',
                         **dict.fromkeys(_IMBALANCE_COLUMNS, 'float64'),
                         'macro_f1_gain_pp': 'float64', 'minority_recall_gain_pp': 'float64'}
    return {
        'evaluation': evaluation,
        'benchmark': benchmark,
        'bootstrap': _json(archive / 'results/bootstrap.json'),
        'nli_bootstrap': _json(archive / 'results-hard/bootstrap.json'),
        'transfer': _csv(archive / 'results/metrics.csv', transfer_schema),
        'imbalance': _csv(archive / 'results-imbalance/comparison.csv', imbalance_schema),
        'mitigation': _csv(archive / 'results-imbalance/mitigation.csv', mitigation_schema),
        'research': _json(archive / 'research-v3/reports/overnight/analysis-v2/analysis-data.json'),
        'archive_root': archive,
        'available': bool(evaluation),
    }


load_evidence = load_presentation_data


def model_summary(data: dict, dataset: str) -> pd.DataFrame:
    """Summarize within-dataset runs in the original five-model display order."""
    source = 'mnli' if dataset.startswith('mnli_') else dataset
    rows = [row for row in data.get('evaluation', [])
            if row.get('target') == dataset and row.get('source') == source
            and row.get('model') in MODELS]
    if not rows:
        frame = _empty_frame(dict.fromkeys(_SCORE_COLUMNS + ['runs', 'seed_sd'], 'float64'))
        return frame.reindex(pd.Index(MODELS, name='model')).reset_index()
    frame = pd.DataFrame(rows)
    grouped = frame.groupby('model')
    summary = grouped[_SCORE_COLUMNS].mean().reindex(MODELS)
    summary['runs'] = grouped.size()
    summary['seed_sd'] = grouped.macro_f1.std()
    return summary.reset_index()


def cost_summary(data: dict, dataset: str | None = None) -> pd.DataFrame:
    """Select saved CPU, batch-one timings, preserving thread configurations."""
    rows = [row for row in data.get('benchmark', [])
            if row.get('model') in MODELS and row.get('device') == 'cpu'
            and row.get('batch_size') == 1
            and (dataset is None or row.get('source') == dataset)]
    return pd.DataFrame(rows) if rows else _empty_frame(_COST_COLUMNS)


def always_world(world_prior: float) -> dict:
    """Analytical AG News majority rule under a specified deployment prior."""
    if not 0 < world_prior <= 1:
        raise ValueError('The World prior must be in (0, 1].')
    return {'accuracy': world_prior,
            'macro_f1': (2 * world_prior / (1 + world_prior)) / 4,
            'minority_recall': 0.0}


def evidence_downloads(root: str | Path | None = None) -> list[tuple[str, Path]]:
    """List existing source-deck and report files for presentation downloads."""
    root = Path(root) if root is not None else ROOT
    archive = _archive_root(root)
    files = [
        ('Source presentation', root / 'apps/assets/research-presentation.pptx'),
        ('Original benchmark report', archive / 'report/final-report.md'),
        ('Mechanism extension report', archive / 'report/research-v3/final-report-zh.md'),
        ('Current research synthesis', archive / 'report/research-v3/final-synthesis-zh.md'),
    ]
    return [(label, path) for label, path in files if path.is_file()]
