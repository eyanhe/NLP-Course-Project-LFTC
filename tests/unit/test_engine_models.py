import tempfile
import unittest
from pathlib import Path
import numpy as np

from lftc.domain import Example, TaskType
from lftc import training


class EngineModelTests(unittest.TestCase):
    def test_full_single_label_space_train_only_features_and_roundtrip(self):
        rows = tuple(Example(str(i), t, (y,), 'train') for i, (t, y) in enumerate([
            ('good wonderful', 0), ('good happy', 0), ('bad awful', 2), ('bad terrible', 2)]))
        validation = (Example('v', 'validationonlyword', (1,), 'validation'),)
        model = training.train_model(rows, validation, TaskType.SINGLE_LABEL, 3, 'tfidf_lr')
        scores = model.predict_scores(rows)
        self.assertEqual(scores.shape, (4, 3))
        self.assertTrue(np.all(scores[:, 1] == 0))
        self.assertNotIn('validationonlyword', model.features.vocabulary)
        self.assertEqual(model.predict(rows).tolist(), [0, 0, 2, 2])
        self.assertEqual(model.predict_scores(()).shape, (0, 3))
        with tempfile.TemporaryDirectory() as tmp:
            training.save_model(model, tmp)
            np.testing.assert_allclose(training.load_model(tmp).predict_scores(rows), scores)

    def test_multilabel_constants_and_empty_vocabulary(self):
        rows = (Example('a', '!', (0, 1), 'train'), Example('b', '?', (1,), 'train'))
        for family in ('tfidf_lr', 'tfidf_svm'):
            model = training.train_model(rows, (), TaskType.MULTILABEL, 3, family)
            scores = model.predict_scores(rows)
            self.assertEqual(scores.shape, (2, 3))
            self.assertTrue(np.isfinite(scores).all())
            self.assertEqual(model.predict(rows)[:, 1:].tolist(), [[1, 0], [1, 0]])

    def test_pair_features_keep_order_and_both_fields(self):
        rows = (Example('a', 'alpha', (0,), 'train', 'beta'),
                Example('b', 'beta', (1,), 'train', 'alpha'))
        model = training.train_model(rows, (), TaskType.PAIR_CLASSIFICATION, 3, 'tfidf_svm')
        self.assertEqual(model.predict(rows).tolist(), [0, 1])
        x = model.features.transform(rows).toarray()
        self.assertFalse(np.array_equal(x[0], x[1]))

    def test_word2vec_is_train_only_and_paired(self):
        rows = (Example('a', 'alpha alpha', (0,), 'train', 'beta beta'),
                Example('b', 'beta beta', (1,), 'train', 'alpha alpha'))
        validation = (Example('v', 'unseen', (1,), 'validation', 'unseen2'),)
        model = training.train_model(rows, validation, TaskType.PAIR_CLASSIFICATION, 3, 'w2v_svm',
                                     {'vector_size': 8, 'w2v_epochs': 1})
        self.assertNotIn('unseen', model.features.vocabulary)
        self.assertEqual(model.features.transform(rows).shape, (2, 32))
        self.assertEqual(model.predict_scores(rows).shape, (2, 3))

    def test_bilstm_batched_pair_training_and_reload(self):
        rows = (Example('a', 'alpha alpha', (0,), 'train', 'beta'),
                Example('b', 'beta beta', (1,), 'train', 'alpha'))
        options = {'vector_size': 4, 'hidden_size': 4, 'epochs': 5, 'w2v_epochs': 1,
                   'batch_size': 1, 'max_length': 2, 'patience': 1, 'device': 'auto',
                   'learning_rate': 0.}
        model = training.train_model(rows, rows, TaskType.PAIR_CLASSIFICATION, 3, 'bilstm', options)
        scores = model.predict_scores(rows, batch_size=1)
        self.assertEqual(scores.shape, (2, 3))
        self.assertTrue(np.isfinite(scores).all())
        self.assertEqual(len(model.history), 2)
        self.assertEqual(model.selected_epoch, 1)
        import torch
        self.assertEqual(model.device.type, 'cuda' if torch.cuda.is_available() else 'cpu')
        self.assertEqual(model.predict_scores(()).shape, (0, 3))
        with tempfile.TemporaryDirectory() as tmp:
            training.save_model(model, tmp)
            # Reloads use CPU; auto can train on CUDA with slightly different
            # floating-point kernels and batching order.
            np.testing.assert_allclose(training.load_model(tmp).predict_scores(rows), scores, rtol=1e-4, atol=1e-5)

    def test_bilstm_multilabel_absent_and_always_present_labels_are_constant(self):
        rows = (Example('a', 'alpha', (0, 1), 'train'), Example('b', 'beta', (1,), 'train'))
        model = training.train_model(rows, (), TaskType.MULTILABEL, 3, 'bilstm',
                                     {'vector_size': 4, 'hidden_size': 4, 'epochs': 1, 'w2v_epochs': 1})
        self.assertEqual(model.device.type, 'cpu')
        self.assertEqual(model.predict(rows)[:, 1:].tolist(), [[1, 0], [1, 0]])

    def test_transformers_require_local_source_and_never_download(self):
        rows = (Example('a', 'alpha', (0,), 'train'),)
        for family in ('distilbert', 'modernbert'):
            with self.assertRaisesRegex(ValueError, 'local'):
                training.train_model(rows, (), TaskType.SINGLE_LABEL, 3, family)

    def test_empty_multilabel_targets_train_as_negatives(self):
        rows = (Example('a', 'alpha', (), 'train'), Example('b', 'beta', (1,), 'train'))
        model = training.train_model(rows, (), TaskType.MULTILABEL, 3, 'tfidf_lr')
        self.assertEqual(model.predict(rows)[:, 0].tolist(), [0, 0])

    def test_tiny_offline_transformer_families_train_and_reload_pairs(self):
        from transformers import (BertTokenizerFast, DistilBertConfig, DistilBertForSequenceClassification,
                                  ModernBertConfig, ModernBertForSequenceClassification)
        constructors = (
            ('distilbert', DistilBertForSequenceClassification, DistilBertConfig(
                vocab_size=8, dim=8, hidden_dim=16, n_layers=1, n_heads=2, max_position_embeddings=32,
                num_labels=3, dropout=0., attention_dropout=0.)),
            ('modernbert', ModernBertForSequenceClassification, ModernBertConfig(
                vocab_size=8, hidden_size=8, intermediate_size=16, num_hidden_layers=1, num_attention_heads=2,
                max_position_embeddings=32, local_attention=8, num_labels=3, reference_compile=False,
                pad_token_id=0, cls_token_id=2, sep_token_id=3, bos_token_id=2, eos_token_id=3)),
        )
        rows = (Example('a', 'alpha', (0,), 'train', 'beta'),
                Example('b', 'beta', (1,), 'train', 'alpha'))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for family, constructor, config in constructors:
                with self.subTest(family=family):
                    source = root/family
                    source.mkdir()
                    (source/'vocab.txt').write_text('[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nalpha\nbeta\ngamma\n', encoding='utf-8')
                    tokenizer = BertTokenizerFast(vocab_file=str(source/'vocab.txt'),
                                                   model_input_names=['input_ids', 'attention_mask'])
                    tokenizer.save_pretrained(source)
                    constructor(config).save_pretrained(source)
                    model = training.train_model(rows, (), TaskType.PAIR_CLASSIFICATION, 3, family,
                                                 {'epochs': 1, 'batch_size': 2, 'max_length': 12}, source)
                    scores = model.predict_scores(rows)
                    self.assertEqual(scores.shape, (2, 3))
                    self.assertTrue(np.isfinite(scores).all())
                    batch = model._batch(rows)
                    self.assertIn(6, batch['input_ids'][0].tolist())
                    training.save_model(model, root/(family+'-saved'))
                    restored = training.load_model(root/(family+'-saved'))
                    np.testing.assert_allclose(restored.predict_scores(rows), scores, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
