"""Export current runs and read completed historical research as data."""
from __future__ import annotations

import csv
from datetime import datetime
import json
from pathlib import Path
import shutil


def _read_json(path, default=None):
    path = Path(path)
    return json.loads(path.read_text(encoding='utf-8-sig')) if path.is_file() else default


def _write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return str(path.resolve())


def _csv(path, records):
    records = list(records)
    columns = list(dict.fromkeys(key for row in records for key in row))
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in records:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict))
                             else value for key, value in row.items()})
    return str(path.resolve())


def _table(records, columns):
    def cell(value):
        if value is None:
            return ''
        if isinstance(value, float):
            return f'{value:.6g}'
        return str(value).replace('|', '\\|').replace('\n', ' ')
    return '\n'.join(['| ' + ' | '.join(columns) + ' |',
                      '| ' + ' | '.join('---' for _ in columns) + ' |',
                      *['| ' + ' | '.join(cell(row.get(key)) for key in columns) + ' |' for row in records]])


def load_history(root):
    """Read known archived analysis/inventory and deliverables, without inference."""
    root = Path(root)
    archive = root / 'archive/2026-11-08'
    overnight = archive / 'research-v3/reports/overnight'
    analysis_path = overnight / 'analysis-v2/analysis-data.json'
    inventory_path = overnight / 'convergence-v1/experiment-inventory-current.json'
    analysis = _read_json(analysis_path, {})
    inventory = _read_json(inventory_path, {})
    count_keys = ('complete_total', 'current_research_complete', 'current_research_with_V', 'formal_test_complete')
    mechanisms = {key: analysis.get(key, []) for key in
                  ('contrasts', 'm2', 'm2_summary', 'scale', 'imbalance', 'trajectories', 's4')}
    deliverable_dirs = [root / 'report/research-v3', archive / 'report/research-v3']
    downloads, figures = [], []
    extensions = {'.md', '.csv', '.json', '.pdf', '.pptx', '.png', '.svg', '.xlsx', '.xls', '.docx'}
    for directory in deliverable_dirs:
        if not directory.is_dir():
            continue
        for path in sorted(directory.rglob('*')):
            if path.is_file() and path.suffix.lower() in extensions:
                downloads.append(str(path.resolve()))
                if path.suffix.lower() in {'.png', '.svg'}:
                    figures.append(str(path.resolve()))
    # The completed analysis has additional figures outside the final deliverables.
    for path in sorted((overnight / 'analysis-v2').glob('*.png')):
        figures.append(str(path.resolve()))
    for path in (analysis_path, inventory_path):
        if path.is_file():
            downloads.append(str(path.resolve()))
    result = {'historical': True, 'available': bool(analysis or inventory or downloads),
              'counts': {key: inventory[key] for key in count_keys if key in inventory},
              'q_records': analysis.get('q_records', []), 'primary': analysis.get('primary', []),
              'mechanisms': mechanisms, 'cases': analysis.get('cases', []),
              'figures': list(dict.fromkeys(figures)), 'downloads': list(dict.fromkeys(downloads)),
              'generated_at': analysis.get('generated_at_sgt'),
              'verification': analysis.get('verification', {}),
              'definitions': inventory.get('definitions', {}),
              'sources': {'analysis': str(analysis_path.resolve()) if analysis_path.is_file() else None,
                          'inventory': str(inventory_path.resolve()) if inventory_path.is_file() else None}}
    result.update(mechanisms)
    return result


def list_runs(paths):
    """List only new run directories; historical records remain a separate view."""
    directory = Path(paths.runs_dir if hasattr(paths, 'runs_dir') else paths)
    result = []
    for path in sorted(directory.glob('*/run.json')):
        try:
            metadata = _read_json(path, {})
        except (OSError, ValueError):
            continue
        if not isinstance(metadata, dict):
            continue
        row = dict(metadata)
        row.setdefault('run_id', path.parent.name)
        row['historical'] = False
        row['run_dir'] = str(path.parent.resolve())
        row['metrics'] = _read_json(path.parent / 'metrics.json', row.get('metrics', {}))
        result.append(row)
    def created_at(row):
        try:
            return datetime.fromisoformat(row.get('created_at', '')).timestamp()
        except (TypeError, ValueError, OverflowError, OSError):
            return float('-inf')
    return sorted(result, key=lambda row: (created_at(row), str(row['run_id'])), reverse=True)


def _plot_metrics(metrics, predictions, directory, label_names):
    # Report commands opt into matplotlib; importing the base package stays light.
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    files = {}
    confusion = metrics.get('confusion_matrix')
    if confusion:
        count = len(confusion)
        fig, ax = plt.subplots(figsize=(min(14, max(5, count * .45)), min(12, max(4, count * .4))))
        image = ax.imshow(confusion, cmap='Blues')
        ax.set(xlabel='Predicted label', ylabel='True label', title='Current run: confusion matrix')
        if count <= 30:
            ticks = list(range(count))
            names = label_names if len(label_names) == count else ticks
            ax.set_xticks(ticks, names, rotation=70, ha='right')
            ax.set_yticks(ticks, names)
        fig.colorbar(image, ax=ax, label='Rows')
        fig.tight_layout()
        path = directory / 'confusion-matrix.png'
        fig.savefig(path, dpi=150)
        plt.close(fig)
        files['confusion_figure'] = str(path.resolve())
    per_label = metrics.get('per_label', [])
    if per_label:
        fig, ax = plt.subplots(figsize=(min(16, max(6, len(per_label) * .25)), 4))
        ids = [row.get('label_id', i) for i, row in enumerate(per_label)]
        ax.bar(ids, [row.get('f1', 0) for row in per_label])
        ax.set(xlabel='Label index', ylabel='F1', ylim=(0, 1), title='Current run: per-label F1')
        fig.tight_layout()
        path = directory / 'per-label-f1.png'
        fig.savefig(path, dpi=150)
        plt.close(fig)
        files['per_label_figure'] = str(path.resolve())
    if predictions:
        lengths = [len((row.get('text', '') + ' ' + (row.get('text_pair') or '')).split()) for row in predictions]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(lengths, bins=min(30, max(1, len(set(lengths)))))
        ax.set(xlabel='Whitespace words (both texts)', ylabel='Rows', title='Current run: evaluated text lengths')
        fig.tight_layout()
        path = directory / 'text-lengths.png'
        fig.savefig(path, dpi=150)
        plt.close(fig)
        files['length_figure'] = str(path.resolve())
    return files


def build_report(run_dir, output_dir=None):
    """Export saved current-run metadata, metrics, predictions and figures."""
    run_dir = Path(run_dir)
    metadata = _read_json(run_dir / 'run.json')
    if not isinstance(metadata, dict):
        raise ValueError(f'No valid run.json in {run_dir}')
    metrics = _read_json(run_dir / 'metrics.json', metadata.get('metrics', {}))
    predictions_path = run_dir / 'predictions.jsonl'
    predictions = []
    if predictions_path.is_file():
        with predictions_path.open(encoding='utf-8-sig') as stream:
            predictions = [json.loads(line) for line in stream if line.strip()]
    run_id = metadata.get('run_id', run_dir.name)
    directory = Path(output_dir) if output_dir is not None else run_dir.parent.parent / 'reports' / str(run_id)
    directory.mkdir(parents=True, exist_ok=True)
    if directory.resolve() == run_dir.resolve():
        raise ValueError('Report output must differ from the saved run directory')
    scalar_metrics = [{'metric': key, 'value': value} for key, value in metrics.items()
                      if isinstance(value, (int, float, str)) or value is None]
    errors = [row for row in predictions if sorted(row.get('labels', [])) != sorted(row.get('prediction', []))]
    files = {'data': _write_json(directory / 'report-data.json',
                               {'historical': False, 'run': metadata, 'metrics': metrics,
                                'prediction_rows': len(predictions), 'error_rows': len(errors)}),
             'metrics_csv': _csv(directory / 'metrics.csv', scalar_metrics),
             'per_label_csv': _csv(directory / 'per-label.csv', metrics.get('per_label', [])),
             'predictions_csv': _csv(directory / 'predictions.csv', predictions),
             'errors_csv': _csv(directory / 'errors.csv', errors)}
    files.update(_plot_metrics(metrics, predictions, directory, metadata.get('label_names', [])))
    text = [f'# Current run: {run_id}', '', 'This report describes a newly produced local run.', '',
            _table([metadata], ['dataset', 'model', 'task', 'seed', 'training_rows', 'validation_rows', 'test_rows']),
            '', '## Saved metrics', '', _table(scalar_metrics, ['metric', 'value']), '',
            f'Prediction rows: {len(predictions)}. Exact-label errors: {len(errors)}.', '',
            '## Exports', '']
    text.extend(f'- [{Path(path).name}]({Path(path).name})' for path in files.values())
    for key, path in files.items():
        if key.endswith('_figure'):
            text.extend(['', f'![{key.replace("_", " ")}]({Path(path).name})'])
    report = directory / 'report.md'
    report.write_text('\n'.join(text) + '\n', encoding='utf-8')
    files['report'] = str(report.resolve())
    files['markdown'] = files['report']
    return files


def build_history_report(root, output_dir):
    """Export completed historical analysis and original deliverables, without training."""
    history = load_history(root)
    directory = Path(output_dir)
    # Refuse writes inside historical sources even if the caller supplies that path.
    archive = (Path(root) / 'archive').resolve()
    if directory.resolve().is_relative_to(archive):
        raise ValueError('Historical report exports must be outside archive/')
    directory.mkdir(parents=True, exist_ok=True)
    files = {'data': _write_json(directory / 'history-data.json', history),
             'q_records_csv': _csv(directory / 'historical-q-records.csv', history['q_records']),
             'primary_csv': _csv(directory / 'historical-primary.csv', history['primary'])}
    for key, rows in history['mechanisms'].items():
        if rows:
            files[f'{key}_csv'] = _csv(directory / f'historical-{key}.csv', rows)
    copied = []
    root = Path(root).resolve()
    for value in list(dict.fromkeys(history['downloads'] + history['figures'])):
        source = Path(value)
        relative = source.relative_to(root)
        target = directory / 'sources' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.resolve() == source.resolve():
            raise ValueError('Export output cannot overwrite a historical source')
        shutil.copy2(source, target)
        copied.append(target.relative_to(directory).as_posix())
    count_names = {'complete_total': 'Completed physical runs',
                   'current_research_complete': 'Current research completed (historical inventory)',
                   'current_research_with_V': 'Current research with V (historical inventory)',
                   'formal_test_complete': 'Formal Q test completed'}
    text = ['# Completed historical research', '',
            'Historical results are read from the preserved 2026-11-08 research. '
            'They are separate from newly produced local runs. No training or inference was performed.', '',
            '## Original inventory counts', '']
    if history['counts']:
        text.append(_table([{'count': count_names[key], 'value': value} for key, value in history['counts'].items()],
                           ['count', 'value']))
    else:
        text.append('Historical inventory is unavailable in this checkout.')
    text.extend(['', '## Historical primary comparisons', '',
                 _table(history['primary'], ['dataset', 'model', 'n', 'metric', 'mean', 'sd']), '',
                 '## Exported data', ''])
    text.extend(f'- [{Path(path).name}]({Path(path).name})' for path in files.values())
    if copied:
        text.extend(['', '## Preserved original deliverables and sources', ''])
        text.extend(f'- [{Path(path).name}]({path})' for path in copied)
    report = directory / 'historical-report.md'
    report.write_text('\n'.join(text) + '\n', encoding='utf-8')
    files['report'] = str(report.resolve())
    files['markdown'] = files['report']
    return files


__all__ = ['load_history', 'list_runs', 'build_report', 'build_history_report']
