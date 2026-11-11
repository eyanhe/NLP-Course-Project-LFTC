import unittest

from lftc.domain import Example, TaskType
from lftc.evaluation import (bootstrap_difference, diagnose_predictions, holm_adjust,
                             mcnemar_test, metric_bundle, select_threshold)


class EvaluationTests(unittest.TestCase):
    def test_full_label_space_and_pair_task(self):
        result = metric_bundle([0, 0], [[3, 0, -1], [1, 0, -1]],
                               TaskType.PAIR_CLASSIFICATION, 'svm_margin')
        self.assertAlmostEqual(result['macro_f1'], 1 / 3)
        self.assertEqual(result['confusion_matrix'], [[2, 0, 0], [0, 0, 0], [0, 0, 0]])

    def test_multilabel_mask_and_margin_threshold(self):
        result = metric_bundle([[1, 0, 1], [0, 1, 1]],
                               [[.2, -.4, .7], [-.2, .4, .7]], 'multilabel', 'svm_margin')
        self.assertEqual(result['threshold'], 0)
        self.assertEqual(result['ap_mask'], [0, 1])
        self.assertEqual(result['macro_ap'], 1)
        self.assertEqual(result['micro_f1'], 1)
        self.assertIsNone(result['per_label'][2]['ap'])
        with self.assertRaises(ValueError):
            metric_bundle([[1, 0], [0, 0]], [[.7, .1], [.1, .1]],
                          'multilabel', 'probability', label_mask=[1])

    def test_validation_threshold_and_probability_validation(self):
        threshold = select_threshold([[1], [0]], [[.4], [.2]], candidates=[.3, .5])
        self.assertEqual(threshold, .3)
        with self.assertRaises(ValueError):
            metric_bundle([0], [[2, -1]], 'single_label', 'probability')

    def test_one_column_multilabel_f1_has_one_declared_label(self):
        result = metric_bundle([[1], [0]], [[.8], [.7]], 'multilabel', 'probability')
        self.assertAlmostEqual(result['macro_f1'], 2 / 3)
        self.assertAlmostEqual(result['micro_f1'], 2 / 3)
        self.assertEqual(result['per_label'][0]['support'], 1)

    def test_paired_statistics(self):
        result = bootstrap_difference([0, 1, 0, 1], [[1, 0], [0, 1], [1, 0], [0, 1]],
                                      [[0, 1], [1, 0], [0, 1], [1, 0]],
                                      'single_label', metric='accuracy', n_resamples=50)
        self.assertEqual(result['difference'], 1)
        self.assertEqual(result['lower'], 1)
        self.assertEqual(mcnemar_test([0, 0, 0], [0, 0, 0], [1, 1, 1])['p_value'], .25)
        self.assertEqual(holm_adjust([.03, .01, .2]), [.06, .03, .2])

    def test_diagnostics_include_pair_text_lengths_and_errors(self):
        examples = (Example('a', 'one', (0,), 'test', 'two three'),
                    Example('b', 'four five', (1,), 'test'))
        result = diagnose_predictions(examples, [[.8, .2], [.7, .3]],
                                      'pair_classification', 2, 'probability')
        self.assertEqual(result['error_count'], 1)
        self.assertEqual(result['label_support'], [1, 1])
        self.assertEqual(result['errors'][0]['sample_id'], 'b')
        self.assertEqual(result['length']['max_words'], 3)


if __name__ == '__main__':
    unittest.main()
