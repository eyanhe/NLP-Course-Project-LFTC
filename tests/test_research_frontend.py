"""Read-only frontend behavior; no model loading or experiment execution."""
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ResearchFrontendDataTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('research_frontend'), 'Current research frontend data layer is missing')
        return importlib.import_module('research_frontend')

    def test_exact_frozen_q_scope_is_available_without_weights(self):
        data = self.api().load_research(ROOT)
        self.assertEqual(len(data['q_records']), 42)
        self.assertEqual(len({row['run_id'] for row in data['q_records']}), 42)
        self.assertEqual(data['counts'], {'complete': 428, 'current': 232, 'with_v': 226, 'q': 42})
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'research-v3/reports/overnight/analysis-v2'
            target.mkdir(parents=True)
            (target / 'analysis-data.json').write_text(json.dumps(data), encoding='utf-8')
            inventory = Path(directory) / 'research-v3/reports/overnight/convergence-v1'
            inventory.mkdir(parents=True)
            (inventory / 'experiment-inventory-current.json').write_text(json.dumps({'complete_total':428,'current_research_complete':232,'current_research_with_V':226,'formal_test_complete':42}), encoding='utf-8')
            self.assertEqual(len(self.api().load_research(Path(directory))['q_records']), 42)

    def test_dataset_filter_keeps_goemotions_metric_separate(self):
        api = self.api()
        rows = api.select_q(api.load_research(ROOT)['q_records'], 'go_emotions')
        self.assertTrue(rows)
        self.assertEqual({row['dataset'] for row in rows}, {'go_emotions'})
        self.assertEqual({row['metric'] for row in rows}, {'macro_ap'})

    def test_stage_and_model_selection_excludes_other_recipes(self):
        api = self.api()
        rows = api.select_q(api.load_research(ROOT)['q_records'], 'banking77', stage='优化与历史参考', models=['tfidf_svm'])
        self.assertEqual({row['model'] for row in rows}, {'tfidf_svm'})
        self.assertEqual({row['stage'] for row in rows}, {'优化与历史参考'})
        self.assertIn('e8ebbc666e71553305dbaf6fd1c229730e7b1a5c468e17d6a286b3191a8ee695', {row['run_id'] for row in rows})

    def test_empty_model_selection_stays_empty(self):
        api = self.api()
        self.assertEqual(api.select_q(api.load_research(ROOT)['q_records'], 'banking77', models=[]), [])

    def test_missing_downloads_are_reported_without_reading_models(self):
        api = self.api()
        with tempfile.TemporaryDirectory() as directory:
            missing = api.download_inventory(Path(directory))
            self.assertEqual(len(missing), 9)
            self.assertFalse(any(item['available'] for item in missing))
        actual = api.download_inventory(ROOT)
        self.assertTrue(all(item['available'] and item['bytes'] > 0 for item in actual))

    def test_browsing_does_not_change_analysis_evidence(self):
        api = self.api()
        source = ROOT / 'research-v3/reports/overnight/analysis-v2/analysis-data.json'
        before = hashlib.sha256(source.read_bytes()).hexdigest()
        data = api.load_research(ROOT)
        api.select_q(data['q_records'], 'banking77', models=['bilstm'])
        api.download_inventory(ROOT)
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), before)


if __name__ == '__main__':
    unittest.main()
