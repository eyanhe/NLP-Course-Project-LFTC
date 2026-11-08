"""Catch path escapes and replacement that corrupts an existing artifact."""
import importlib
import json
from pathlib import Path
import tempfile
import unittest


class StorageTests(unittest.TestCase):
    def api(self):
        try:
            return importlib.import_module('research_v3.storage')
        except ModuleNotFoundError:
            self.fail('research_v3.storage is not implemented')

    def test_output_cannot_escape_root(self):
        api = self.api()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / 'outputs'
            with self.assertRaises(ValueError):
                api.atomic_write_json(root / '..' / 'escape.json', {'bad': True}, root)
            self.assertFalse((Path(temp) / 'escape.json').exists())

    def test_complete_replacement_is_valid_json(self):
        api = self.api()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            target = root / 'sub' / 'data.json'
            api.atomic_write_json(target, {'before': 1}, root)
            api.atomic_write_json(target, {'after': '中文'}, root)
            self.assertEqual(json.loads(target.read_text(encoding='utf-8')), {'after': '中文'})
            self.assertEqual(list(target.parent.glob('*.tmp')), [])


if __name__ == '__main__':
    unittest.main()
