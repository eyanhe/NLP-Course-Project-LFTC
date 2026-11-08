import copy
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from research_v3 import overnight_training as overnight


class OvernightControlsTest(unittest.TestCase):
    def test_diagnostics_cuda_contract_requires_gpu_even_for_fixed_svm(self):
        self.assertTrue(overnight.gpu_job_required({'model_id':'fixed_tfidf_svm','engine':'diagnostics'}))
        self.assertFalse(overnight.gpu_job_required({'model_id':'tfidf_svm','engine':'core'}))

    def test_expired_deadline_prevents_more_training(self):
        self.assertFalse(overnight.may_start('2026-11-08T09:00:00+08:00', datetime(2026, 11, 8, 1, 0, tzinfo=timezone.utc)))
        self.assertTrue(overnight.may_start('2026-11-08T09:00:00+08:00', datetime(2026, 11, 8, 0, 59, tzinfo=timezone.utc)))

    def test_runtime_model_path_comes_from_locked_relative_files(self):
        assets = {'asset_hash':'identity','models':{'distilbert':{'directory':'C:/old/assets/distilbert','revision':'a'*40}},'files':[{'model_id':'distilbert','path':'assets/distilbert/config.json'}]}
        original = copy.deepcopy(assets)
        with tempfile.TemporaryDirectory() as folder:
            project = Path(folder)
            (project/'assets/distilbert').mkdir(parents=True)
            (project/'assets/distilbert/config.json').write_text('{}')
            runtime = overnight.runtime_assets(assets, project)
            self.assertEqual(Path(runtime['models']['distilbert']['directory']), project/'assets/distilbert')
            self.assertEqual(runtime['asset_hash'], 'identity')
            self.assertEqual(assets, original)

    def test_undeclared_or_changed_run_is_rejected(self):
        source = {'runs':[{'run_id':'r1','run':{'epochs':10}}]}
        with self.assertRaises(ValueError):
            overnight.validate_entries([{'entry':{'run_id':'r1','run':{'epochs':1}}}], source)
        with self.assertRaises(ValueError):
            overnight.validate_entries([{'entry':{'run_id':'r2','run':{'epochs':10}}}], source)


if __name__ == '__main__':
    unittest.main()
