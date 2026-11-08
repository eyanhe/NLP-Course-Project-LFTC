"""Catch unsafe output routing, source drift and hidden supervision changes."""
import copy
import importlib
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
SPEC = ROOT / 'docs/superpowers/specs/2026-11-07-research-protocol.json'


class ConfigTests(unittest.TestCase):
    def api(self):
        try:
            return importlib.import_module('research_v3.config')
        except ModuleNotFoundError:
            self.fail('research_v3.config is not implemented')

    def protocol(self):
        result = json.loads(SPEC.read_text(encoding='utf-8-sig'))
        result['output_root'] = 'research-v3'
        result['execution'] = {'stage': 'pretraining', 'training_enabled': False,
                               'test_predictions_enabled': False}
        return result

    def test_fourth_new_dataset_is_rejected(self):
        p = self.protocol()
        p['scope']['new_datasets'].append('extra')
        with self.assertRaises(ValueError):
            self.api().validate_protocol(p)

    def test_legacy_results_cannot_be_an_output(self):
        p = self.protocol()
        p['output_root'] = 'results'
        with self.assertRaises(ValueError):
            self.api().validate_protocol(p)

    def test_path_escape_cannot_be_an_output(self):
        p = self.protocol()
        p['output_root'] = '../research-v3'
        with self.assertRaises(ValueError):
            self.api().validate_protocol(p)

    def test_unpinned_source_is_rejected(self):
        p = self.protocol()
        p['sources']['banking77']['revision'] = 'main'
        with self.assertRaises(ValueError):
            self.api().validate_protocol(p)

    def test_strict_fewshot_cannot_read_validation(self):
        p = self.protocol()
        p['partitions']['strict_fewshot_target_validation_access'] = True
        with self.assertRaises(ValueError):
            self.api().validate_protocol(p)

    def test_training_is_blocked_before_user_enables_it(self):
        with self.assertRaises(PermissionError):
            self.api().require_training_enabled(self.protocol())

    def test_test_predictions_are_blocked_before_evaluation_freeze(self):
        p = self.protocol()
        p['execution']['training_enabled'] = True
        with self.assertRaises(PermissionError):
            self.api().require_test_enabled(p)

    def test_protocol_file_loads_without_mutating_original(self):
        api = self.api()
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / 'config.json'
            p = self.protocol()
            target.write_text(json.dumps(p), encoding='utf-8')
            loaded = api.load_protocol(target)
            self.assertEqual(loaded['protocol_id'], 'tc-explain-v3-20261107')
            self.assertEqual(json.loads(target.read_text())['execution']['training_enabled'], False)


if __name__ == '__main__':
    unittest.main()
