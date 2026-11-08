"""Catch false readiness and missing interpreter/dependency evidence."""
import importlib
import json
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest


class RuntimeTests(unittest.TestCase):
    def api(self):
        try:
            return importlib.import_module('research_v3.runtime')
        except ModuleNotFoundError:
            self.fail('research_v3.runtime is not implemented')

    def test_probe_records_real_interpreter_and_does_not_claim_training(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'environment.json'
            result = self.api().probe_environment(path)
            saved = json.loads(path.read_text(encoding='utf-8'))
            self.assertTrue(Path(saved['python_executable']).is_file())
            self.assertEqual(saved['python_version'].split('.')[:2], ['3', '11'])
            self.assertFalse(saved['actual_model_training_run'])
            self.assertIn('packages', saved)
            self.assertEqual(result['python_executable'], saved['python_executable'])

    def test_native_dependencies_load_after_the_research_entrypoint(self):
        # Regression: loading Torch before Arrow24 on this Windows wheel set
        # terminates the interpreter with 0xC0000005, outside Python exceptions.
        code = ('import research_v3; '
                'from transformers import AutoTokenizer; '
                't=AutoTokenizer.from_pretrained("assets/distilbert",local_files_only=True); '
                'assert len(t("a real input")["input_ids"]) > 2; '
                'import datasets; import pyarrow.dataset; print("loaded")')
        result = subprocess.run([sys.executable, '-X', 'faulthandler', '-c', code],
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])
        self.assertIn('loaded', result.stdout)


if __name__ == '__main__':
    unittest.main()
