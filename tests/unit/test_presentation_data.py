"""Keep the presentation's saved results within their original experiment scopes."""
import json
import math
from pathlib import Path
import tempfile
import unittest

import pandas as pd


def write_json(root, name, value):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def evaluation(source, target, model, score, seed=None):
    return {'source': source, 'target': target, 'model': model, 'seed': seed,
            'accuracy': score, 'macro_precision': score, 'macro_recall': score,
            'macro_f1': score, 'mcc': score}


class PresentationDataTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def test_summary_excludes_transfer_and_averages_deep_seeds(self):
        tmp_path = self.root
        from lftc.reporting.presentation import load_presentation_data, model_summary

        archive = tmp_path / 'archive/2026-11-08'
        write_json(archive, 'results-v2/evaluation.json', {'rows': [
            evaluation('imdb', 'imdb', 'logistic_regression', .8),
            evaluation('yelp', 'imdb', 'logistic_regression', .1),
            evaluation('imdb', 'imdb', 'distilbert', .7, 42),
            evaluation('imdb', 'imdb', 'distilbert', .9, 43),
        ]})
        frame = model_summary(load_presentation_data(tmp_path), 'imdb').set_index('model')
        assert len(frame) == 5
        assert frame.loc['logistic_regression', 'macro_f1'] == .8
        assert frame.loc['logistic_regression', 'runs'] == 1
        assert math.isclose(frame.loc['distilbert', 'macro_f1'], .8)
        assert frame.loc['distilbert', 'runs'] == 2
        assert math.isclose(frame.loc['distilbert', 'seed_sd'], .1414213562373095)
        assert pd.isna(frame.loc['bilstm', 'macro_f1'])


    def test_mnli_keeps_matched_and_mismatched_and_hypothesis_baseline_separate(self):
        tmp_path = self.root
        from lftc.reporting.presentation import load_presentation_data, model_summary

        write_json(tmp_path, 'results-hard/evaluation.json', {'rows': [
            evaluation('mnli', 'mnli_matched', 'linear_svm', .5),
            evaluation('mnli', 'mnli_mismatched', 'linear_svm', .4),
            evaluation('mnli', 'mnli_matched', 'hypothesis_only_svm', .9),
            evaluation('imdb', 'mnli_matched', 'linear_svm', .1),
        ]})
        data = load_presentation_data(tmp_path)
        matched = model_summary(data, 'mnli_matched').set_index('model')
        mismatched = model_summary(data, 'mnli_mismatched').set_index('model')
        assert matched.loc['linear_svm', 'macro_f1'] == .5
        assert mismatched.loc['linear_svm', 'macro_f1'] == .4
        assert 'hypothesis_only_svm' not in matched.index


    def test_cost_deduplicates_repeated_exports_without_pooling_devices_or_batches(self):
        tmp_path = self.root
        from lftc.reporting.presentation import cost_summary, load_presentation_data

        base = {'source': 'imdb', 'model': 'logistic_regression', 'device': 'cpu',
                'batch_size': 1, 'threads': 4, 'median_ms': 1.65, 'p95_ms': 3.57}
        write_json(tmp_path, 'results/benchmark.json', {'rows': [base]})
        write_json(tmp_path, 'results-v2/benchmark.json', {'rows': [
            base, {**base, 'device': 'cuda', 'median_ms': .2},
            {**base, 'batch_size': 16, 'median_ms': .3},
            {**base, 'threads': 1, 'median_ms': 2.0},
        ]})
        data = load_presentation_data(tmp_path)
        assert len(data['benchmark']) == 4
        cpu = cost_summary(data, 'imdb')
        assert cpu['median_ms'].tolist() == [1.65, 2.0]
        assert cpu['device'].tolist() == ['cpu', 'cpu']
        assert cost_summary(data, 'yelp').empty


    def test_missing_checkout_has_typed_empty_frames_and_no_fabricated_results(self):
        tmp_path = self.root
        from lftc.reporting.presentation import (
            cost_summary, evidence_downloads, load_presentation_data, model_summary,
        )

        data = load_presentation_data(tmp_path)
        assert data['available'] is False
        assert data['archive_root'] == tmp_path / 'archive/2026-11-08'
        assert data['evaluation'] == data['benchmark'] == []
        assert data['bootstrap'] == data['nli_bootstrap'] == data['research'] == {}
        assert data['transfer'].empty and str(data['transfer']['macro_f1'].dtype) == 'float64'
        assert data['imbalance'].empty and 'minority_recall' in data['imbalance']
        assert data['mitigation'].empty and 'macro_f1_gain_pp' in data['mitigation']
        assert model_summary(data, 'imdb')['macro_f1'].isna().all()
        assert cost_summary(data).empty
        assert evidence_downloads(tmp_path) == []


    def test_majority_rule_uses_deployment_prior_without_changing_macro_average(self):
        from lftc.reporting.presentation import always_world

        assert always_world(.25) == {'accuracy': .25, 'macro_f1': .1, 'minority_recall': 0.0}
        assert math.isclose(always_world(.94)['macro_f1'], .2422680412371134)
        with self.assertRaises(ValueError):
            always_world(0)
