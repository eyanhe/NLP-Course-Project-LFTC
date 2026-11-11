from pathlib import Path
import importlib
import tempfile
import unittest
import numpy as np

from tests.research import test_training


class TimedInferenceTests(unittest.TestCase):
    def test_timing_preserves_predictions_and_separates_loading_from_prediction(self):
        try:api=importlib.import_module('research_v3.timed_inference')
        except ModuleNotFoundError:self.fail('timed inference implementation is missing')
        from research_v3.train import fit_run
        from research_v3.supplemental import score_checkpoint_cpu
        for model in ['tfidf_lr','bilstm']:
            with self.subTest(model=model),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);data,inputs,run=test_training.TrainingTests().fixture(root,model=model)
                output=root/'runs'/run.run_id;fit_run(run,data,inputs,{},output)
                expected,status=score_checkpoint_cpu(output,tuple(inputs.values()))
                actual,returned,timing=api.score_checkpoint_cpu_timed(output,tuple(inputs.values()))
                np.testing.assert_allclose(actual,expected,rtol=1e-6,atol=1e-7)
                self.assertEqual(returned['run_id'],status['run_id'])
                self.assertEqual(timing['rows'],2)
                self.assertGreater(timing['predict_seconds'],0)
                for name in ['verification_bundle_seconds','model_load_seconds']:
                    self.assertGreaterEqual(timing[name],0)
                self.assertGreaterEqual(timing['authenticated_inference_seconds'],timing['predict_seconds'])
                self.assertAlmostEqual(timing['rows_per_second'],2/timing['predict_seconds'])


if __name__=='__main__':unittest.main()
