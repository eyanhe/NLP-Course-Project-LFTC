"""Preserved benchmark inference, including original sentence-pair checkpoints."""
from pathlib import Path
import sys
import tempfile
import types
import unittest

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC


PROJECT = Path(__file__).resolve().parents[2]


def checkpoint(root, source, name, model, area='models'):
    folder = root / area / source / name
    folder.mkdir(parents=True)
    joblib.dump(model, folder / 'model.joblib')
    (folder / 'metadata.json').write_text('{}', encoding='utf-8')
    return folder


def check_serialized_single_models_preserve_probability_and_signed_margin(test, tmp_path):
    from lftc.training.benchmarks import load_benchmark_model
    texts = ['awful terrible bad', 'bad disappointing', 'wonderful excellent good', 'excellent good']
    for name, classifier in [('logistic_regression', LogisticRegression()),
                             ('linear_svm', LinearSVC())]:
        model = make_pipeline(TfidfVectorizer(), classifier).fit(texts, [0, 0, 1, 1])
        checkpoint(tmp_path, 'imdb', name, model)
        loaded = load_benchmark_model('imdb', name, root=tmp_path)
        result = loaded.predict('<b>awful</b> &amp; terrible')
        assert result['prediction'] == 'Negative'
        assert result['label'] == 0
        assert not result['truncated']
        assert result['milliseconds'] >= 0
        if name == 'logistic_regression':
            assert result['score_label'] == 'predicted-class probability'
            test.assertAlmostEqual(result['score'], model.predict_proba(['awful & terrible'])[0, 0])
        else:
            assert result['score_label'] == 'positive-class margin'
            assert result['score'] < 0


def check_pair_joblib_legacy_class_mapping_preserves_order_without_module_alias(test, tmp_path):
    from lftc.models.benchmark_compat import PairTfidf
    from lftc.training.benchmarks import load_benchmark_model
    legacy_module = types.ModuleType('pair_models')
    legacy_type = type('PairTfidf', (PairTfidf,), {'__module__': 'pair_models'})
    legacy_module.PairTfidf = legacy_type
    pairs = [{'premise': 'cat sleeps', 'hypothesis': 'animal sleeps'},
             {'premise': 'animal sleeps', 'hypothesis': 'cat sleeps'},
             {'premise': 'cat runs', 'hypothesis': 'cat sleeps'},
             {'premise': 'cat sleeps', 'hypothesis': 'cat runs'}] * 2
    labels = [0, 1, 2, 0] * 2
    prior = sys.modules.get('pair_models')
    try:
        sys.modules['pair_models'] = legacy_module
        model = make_pipeline(legacy_type(), LogisticRegression()).fit(pairs, labels)
        folder = tmp_path / 'models-hard' / 'logistic_regression'
        folder.mkdir(parents=True)
        joblib.dump(model, folder / 'model.joblib', compress=3)
        (folder / 'metadata.json').write_text('{}', encoding='utf-8')
    finally:
        if prior is None:
            sys.modules.pop('pair_models', None)
        else:
            sys.modules['pair_models'] = prior
    loaded = load_benchmark_model('mnli', 'logistic_regression', root=tmp_path)
    result = loaded.predict('animal sleeps', premise='cat sleeps')
    expected = model.predict_proba([pairs[0]])[0]
    assert result['label'] == int(expected.argmax())
    assert result['prediction'] == ['Entailment', 'Neutral', 'Contradiction'][expected.argmax()]
    test.assertAlmostEqual(result['score'], expected.max())
    assert isinstance(loaded.model.steps[0][1], PairTfidf)
    assert sys.modules.get('pair_models') is prior
    with test.assertRaisesRegex(ValueError, 'premise'):
        loaded.predict('animal sleeps')


def check_pair_word_means_match_original_concatenation():
    from lftc.models.benchmark_compat import PairWordMeans
    vectors = {'cat': np.array([1., 2.]), 'animal': np.array([3., 1.])}
    transformed = PairWordMeans(vectors).transform([
        {'premise': 'cat unknown', 'hypothesis': 'animal'},
        {'premise': '', 'hypothesis': 'unknown'}])
    np.testing.assert_allclose(transformed, [[1, 2, 3, 1, 2, 1, 3, 2], [0]*8])


def check_comparison_uses_injected_cached_loader_and_exposes_model_ids():
    from lftc.training.benchmarks import predict_benchmarks
    calls = []

    class Saved:
        def predict(self, text, premise=''):
            assert (text, premise) == ('hypothesis', 'premise')
            return {'prediction': 'Neutral', 'label': 1, 'score': -.4,
                    'score_label': 'predicted-class margin', 'milliseconds': 2., 'truncated': False}

    def loader(source, model):
        calls.append((source, model))
        return Saved()

    rows = predict_benchmarks('mnli', 'hypothesis', 'premise',
                              models=['linear_svm', 'bilstm'], loader=loader)
    assert calls == [('mnli', 'linear_svm'), ('mnli', 'bilstm')]
    assert [row['model'] for row in rows] == ['linear_svm', 'bilstm']
    assert rows[0]['score'] == -.4


def check_missing_and_unknown_checkpoints_are_explicit(test, tmp_path):
    from lftc.training.benchmarks import load_benchmark_model
    with test.assertRaisesRegex(FileNotFoundError, 'imdb.*linear_svm|linear_svm.*imdb'):
        load_benchmark_model('imdb', 'linear_svm', root=tmp_path)
    with test.assertRaisesRegex(ValueError, 'source'):
        load_benchmark_model('demo200', 'linear_svm', root=tmp_path)
    with test.assertRaisesRegex(ValueError, 'model'):
        load_benchmark_model('imdb', 'made_up', root=tmp_path)


def check_original_benchmark_checkpoint(test, source, name, text, premise):
    from lftc.training.benchmarks import load_benchmark_model
    if not (PROJECT / 'archive/2026-11-08/models').is_dir():
        test.skipTest('Original experiment weights are not included in this checkout')
    model = load_benchmark_model(source, name, root=PROJECT)
    result = model.predict(text, premise)
    assert result['prediction'] in model.labels
    assert np.isfinite(result['score'])
    assert result['milliseconds'] > 0
    assert not result['truncated']
    if source == 'imdb':
        test.assertIn(result['prediction'], ('Negative', 'Positive'))
    if name in ('bilstm', 'distilbert'):
        assert next(model.model.parameters()).device.type == 'cpu'
        assert not model.model.training
        assert 0 <= result['score'] <= 1
        assert model.predict('excellent ' * 260, premise or '')['truncated']


class BenchmarkModelsTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_serialized_single_probability_and_signed_margin(self):
        check_serialized_single_models_preserve_probability_and_signed_margin(self, self.root)

    def test_legacy_pair_joblib_mapping_preserves_order(self):
        check_pair_joblib_legacy_class_mapping_preserves_order_without_module_alias(self, self.root)

    def test_pair_word_means(self):
        check_pair_word_means_match_original_concatenation()

    def test_comparison_cached_loader(self):
        check_comparison_uses_injected_cached_loader_and_exposes_model_ids()

    def test_explicit_missing_checkpoint(self):
        check_missing_and_unknown_checkpoints_are_explicit(self, self.root)

    def test_original_checkpoints(self):
        for source, name in [('imdb', 'linear_svm'), ('imdb', 'word2vec_svm'),
                             ('mnli', 'linear_svm'), ('mnli', 'word2vec_svm'),
                             ('imdb', 'bilstm'), ('mnli', 'bilstm'), ('imdb', 'distilbert')]:
            with self.subTest(source=source, model=name):
                text = ('Someone is reading indoors.' if source == 'mnli' else
                        'An excellent film with wonderful performances.')
                premise = 'A woman is reading in a library.' if source == 'mnli' else ''
                check_original_benchmark_checkpoint(self, source, name, text, premise)


if __name__ == '__main__':
    unittest.main()
