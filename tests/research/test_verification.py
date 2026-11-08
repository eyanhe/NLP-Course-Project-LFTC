"""Readiness must fail closed for missing and altered real artifacts."""
import importlib
from pathlib import Path
import tempfile
import unittest

from research_v3.storage import atomic_write_json,hash_file


class VerificationTests(unittest.TestCase):
    def api(self):
        try:
            return importlib.import_module('research_v3.verification')
        except ModuleNotFoundError:
            self.fail('independent pretraining verification is missing')

    def test_missing_data_cannot_report_pretraining_complete(self):
        with tempfile.TemporaryDirectory() as temp:
            report=self.api().verify_pretraining({'_protocol_hash':'fixture','execution':
                {'training_enabled':False,'test_predictions_enabled':False,'actual_fitting_permitted':False}},Path(temp))
            self.assertFalse(report['complete'])
            self.assertGreater(len(report['failures']),0)
            self.assertFalse(report['actual_model_training_run'])

    def test_file_verification_reads_bytes_instead_of_trusting_manifest(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);path=root/'source.txt';path.write_text('a',encoding='utf-8')
            expected=hash_file(path);path.write_text('b',encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'hash|changed'):
                self.api().verified_file(root,'source.txt',expected)

    def test_forged_forward_success_with_gradients_is_rejected(self):
        evidence={'all_passed':True,'checks':[{'finite_outputs':True,'gradients_created':True,
                   'weight_versions_unchanged':True}],'actual_model_training_run':False,
                  'official_test_predictions_generated':False}
        with self.assertRaisesRegex(ValueError,'gradient|forward'):
            self.api().validate_forward_evidence(evidence)


if __name__=='__main__':
    unittest.main()
