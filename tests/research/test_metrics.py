"""The score scale and valid positive/negative AP mask affect interpretation."""
import importlib
import unittest
import numpy as np


class MetricTests(unittest.TestCase):
    def api(self):
        try:return importlib.import_module('research_v3.metrics')
        except ModuleNotFoundError:self.fail('research metrics implementation is missing')

    def test_perfect_single_label_predictions_keep_all_declared_labels(self):
        result=self.api().metric_bundle(np.array([0,1,2]),np.eye(3),'single_label','probability')
        self.assertEqual(result['macro_f1'],1)
        self.assertEqual(len(result['per_label']),3)

    def test_margin_zero_is_not_probability_half(self):
        y=np.array([[1,0],[0,1],[1,0]])
        score=np.array([[.2,-.2],[-.2,.2],[.1,-.1]])
        result=self.api().metric_bundle(y,score,'multilabel','svm_margin',label_mask=[0,1])
        self.assertEqual(result['macro_f1'],1)
        self.assertEqual(result['macro_ap'],1)

    def test_ap_constants_are_na_and_mask_cannot_use_them(self):
        y=np.array([[1,1,0],[0,1,0],[1,1,0]])
        score=np.array([[.9,.8,.2],[.2,.8,.2],[.8,.8,.2]])
        result=self.api().metric_bundle(y,score,'multilabel','probability',label_mask=[0])
        self.assertEqual(result['macro_ap'],1)
        self.assertIsNone(result['per_label'][1]['ap'])
        self.assertIsNone(result['per_label'][2]['ap'])
        with self.assertRaisesRegex(ValueError,'mask|positive|negative'):
            self.api().metric_bundle(y,score,'multilabel','probability',label_mask=[1])

    def test_nonperfect_ap_matches_hand_calculation(self):
        y=np.array([[1],[0],[1],[0]])
        scores=np.array([[.8],[.9],[.7],[.1]])
        result=self.api().metric_bundle(y,scores,'multilabel','probability',label_mask=[0])
        self.assertAlmostEqual(result['macro_ap'],7/12)

    def test_unknown_score_kind_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'score'):
            self.api().metric_bundle(np.array([0,1]),np.eye(2),'single_label','mystery')


if __name__=='__main__':unittest.main()
