import csv
import json
from pathlib import Path
import tempfile
import unittest

from lftc.reporting import build_history_report, build_report, list_runs, load_history


class ReportingTests(unittest.TestCase):
    def test_newest_runs_are_listed_first_regardless_of_directory_name(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for run_id, created in [('a-old', '2026-11-09T12:00:00+00:00'),
                                    ('z-new', '2026-11-11T12:00:00+00:00')]:
                path = root / run_id
                path.mkdir()
                (path / 'run.json').write_text(json.dumps({'run_id': run_id, 'created_at': created}))
            self.assertEqual([row['run_id'] for row in list_runs(root)], ['z-new', 'a-old'])

    def test_empty_history_and_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(load_history(directory)['q_records'], [])
            self.assertEqual(list_runs(Path(directory)), [])

    def test_history_preserves_inventory_counts_and_exports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / 'archive/2026-11-08/research-v3/reports/overnight'
            (base / 'analysis-v2').mkdir(parents=True)
            (base / 'convergence-v1').mkdir()
            (base / 'analysis-v2/analysis-data.json').write_text(json.dumps({
                'q_records': [{'run_id': 'historical', 'score': .8}],
                'primary': [{'dataset': 'tiny', 'model': 'svm', 'mean': .8}], 'cases': []}))
            counts = {'complete_total': 428, 'current_research_complete': 232,
                      'current_research_with_V': 226, 'formal_test_complete': 42}
            (base / 'convergence-v1/experiment-inventory-current.json').write_text(json.dumps(counts))
            tables = root / 'archive/2026-11-08/report/research-v3/tables'
            tables.mkdir(parents=True)
            (tables / 'results.xlsx').write_bytes(b'opaque-workbook-data')
            result = load_history(root)
            self.assertEqual(result['counts'], counts)
            self.assertIn('results.xlsx', [Path(path).name for path in result['downloads']])
            files = build_history_report(root, root / 'export')
            self.assertTrue(Path(files['report']).exists())
            self.assertEqual(files['markdown'], files['report'])
            self.assertIn('428', Path(files['report']).read_text(encoding='utf-8'))
            self.assertEqual(json.loads(Path(files['data']).read_text(encoding='utf-8'))['counts'], counts)

    def test_current_report_uses_saved_metrics_and_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / 'artifacts/runs/tiny'
            run_dir.mkdir(parents=True)
            (run_dir / 'run.json').write_text(json.dumps({'run_id': 'tiny', 'task': 'single_label',
                'dataset': 'tiny', 'model': 'svm', 'label_count': 2, 'label_names': ['no', 'yes']}))
            (run_dir / 'metrics.json').write_text(json.dumps({'macro_f1': .5,
                'confusion_matrix': [[1, 0], [1, 0]], 'per_label': [{'label_id': 0, 'f1': 2/3}]}))
            (run_dir / 'predictions.jsonl').write_text(json.dumps({'sample_id': 'x', 'labels': [1],
                'prediction': [0], 'text': 'test', 'scores': [.8, .2], 'score_kind': 'probability'}) + '\n')
            files = build_report(run_dir)
            self.assertTrue(Path(files['report']).exists())
            self.assertEqual(files['markdown'], files['report'])
            self.assertTrue(Path(files['confusion_figure']).exists())
            self.assertIn('x', Path(files['predictions_csv']).read_text(encoding='utf-8'))
            self.assertEqual(list_runs(Path(directory) / 'artifacts/runs')[0]['run_id'], 'tiny')

    def test_prediction_jsonl_preserves_unicode_line_separators_in_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / 'run'
            run_dir.mkdir()
            (run_dir / 'run.json').write_text(json.dumps({'run_id': 'unicode'}), encoding='utf-8')
            text = 'first\u0085second\u2028third, "quoted"\nlast'
            rows = [{'sample_id': str(i), 'text': text, 'text_pair': text,
                     'labels': [0], 'prediction': [1], 'scores': [.1, .9]} for i in range(2)]
            (run_dir / 'predictions.jsonl').write_text(
                '\n'.join(json.dumps(row, ensure_ascii=False) for row in rows) + '\n', encoding='utf-8')
            files = build_report(run_dir, Path(directory) / 'report')
            with Path(files['predictions_csv']).open(encoding='utf-8-sig', newline='') as stream:
                exported = list(csv.DictReader(stream))
            self.assertEqual(len(exported), 2)
            self.assertEqual(exported[0]['text'], text)
            self.assertEqual(exported[0]['text_pair'], text)
            self.assertEqual(json.loads(exported[0]['scores']), [.1, .9])
            self.assertEqual(json.loads(Path(files['data']).read_text(encoding='utf-8'))['prediction_rows'], 2)


if __name__ == '__main__':
    unittest.main()
