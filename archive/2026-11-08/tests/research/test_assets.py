"""Catch corrupted cache reuse and missing assets silently treated as ready."""
import hashlib
import importlib
from pathlib import Path
import tempfile
import unittest


class AssetTests(unittest.TestCase):
    def api(self):
        try:
            return importlib.import_module('research_v3.assets')
        except ModuleNotFoundError:
            self.fail('research_v3.assets is not implemented')

    def test_corrupt_local_asset_is_rejected_offline(self):
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / 'weights'
            target.write_bytes(b'changed')
            spec = {'url': 'https://example.invalid/unused', 'expected_sha256': hashlib.sha256(b'original').hexdigest()}
            with self.assertRaises(ValueError):
                self.api().ensure_file(spec, target, offline=True)
            self.assertEqual(target.read_bytes(), b'changed')

    def test_verified_cache_is_reused_without_network(self):
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / 'asset'
            target.write_bytes(b'abc')
            spec = {'url': 'https://example.invalid/unused', 'expected_sha256':
                    'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad'}
            record = self.api().ensure_file(spec, target, offline=True)
            self.assertEqual(record['bytes'], 3)
            self.assertTrue(record['cache_reused'])

    def test_missing_cache_does_not_turn_into_ready(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(FileNotFoundError):
                self.api().ensure_file({'url': 'https://example.invalid/x'}, Path(temp) / 'missing', offline=True)


if __name__ == '__main__':
    unittest.main()
