import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from lftc.domain import DatasetAsset, Example, TaskType
from lftc import data


class EngineDataTests(unittest.TestCase):
    def asset(self, files, dataset='toy', task=TaskType.SINGLE_LABEL):
        return DatasetAsset(dataset, dataset, '', '', tuple(files), task, 3, ('a', 'b', 'c'))

    def write(self, root, name, rows, dataset='toy'):
        path = root / dataset / name
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(rows), path, row_group_size=2)

    def test_holdout_is_training_only_stable_across_shards_and_roundtrips(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write(root, 'train-0.parquet', [{'text': 'word '+str(i), 'label': i % 3} for i in range(6)])
            self.write(root, 'train-1.parquet', [{'text': 'word '+str(i), 'label': i % 3} for i in range(6, 12)])
            self.write(root, 'test.parquet', [{'text': 'heldout', 'label': 2}])
            asset = self.asset(['train-0.parquet', 'train-1.parquet', 'test.parquet'])
            full = data.load_dataset(asset, root)
            self.assertEqual(len(full.train) + len(full.validation), 12)
            self.assertEqual(len({x.sample_id for x in full.train + full.validation}), 12)
            self.assertEqual(full.test[0].text, 'heldout')
            self.assertTrue(all(x.split == 'validation' for x in full.validation))
            self.assertEqual(full, data.load_dataset(asset, root))
            data.save_prepared(full, root / 'prepared')
            self.assertEqual(full, data.load_prepared(root / 'prepared'))

    def test_mnli_matched_validation_mismatched_test_filtering_and_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for split in ('train', 'validation_matched', 'validation_mismatched'):
                self.write(root, split+'.parquet', [
                    {'premise': 'p', 'hypothesis': 'h', 'label': -1},
                    {'premise': 'p', 'hypothesis': 'h', 'label': 1},
                    {'premise': 'q', 'hypothesis': 'z', 'label': 2},
                ], 'mnli')
            asset = self.asset([s+'.parquet' for s in ('train', 'validation_matched', 'validation_mismatched')],
                               'mnli', TaskType.PAIR_CLASSIFICATION)
            loaded = data.load_dataset(asset, root, limit_per_split=1)
            self.assertEqual([len(loaded.train), len(loaded.validation), len(loaded.test)], [1, 1, 1])
            self.assertEqual(loaded.test[0].text_pair, 'h')
            self.assertEqual(loaded.test[0].labels, (1,))

    def test_sampling_deterministic_kshot_and_fraction(self):
        rows = tuple(Example(str(i), 'x', (i % 3,), 'train') for i in range(30))
        self.assertEqual(len(data.sample_training(rows, mode='fraction', size=.2)), 6)
        sample = data.sample_training(rows, mode='kshot', k=2)
        self.assertEqual([sum(c in x.labels for x in sample) for c in range(3)], [2, 2, 2])
        self.assertEqual(sample, data.sample_training(rows, mode='kshot', k=2))
        tail = data.sample_training(rows, mode='long_tail', imbalance_ratio=10)
        self.assertGreater(sum(0 in x.labels for x in tail), sum(2 in x.labels for x in tail))

    def test_yahoo_retains_fields_and_multilabel_empty_targets_are_valid(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write(root, 'train.parquet', [{'question_title': 'title', 'question_content': 'question',
                                               'best_answer': 'answer', 'topic': 1}], 'yahoo_answers_topics')
            yahoo = data.load_dataset(self.asset(['train.parquet'], 'yahoo_answers_topics'), root)
            self.assertIn('title', yahoo.train[0].text)
            self.assertIn('question', yahoo.train[0].text)
            self.assertIn('answer', yahoo.train[0].text)
            self.write(root, 'train.parquet', [{'text': 'valid no positives', 'labels': []},
                                               {'text': 'positive', 'labels': [0, 2]}], 'go_emotions')
            emotions = data.load_dataset(self.asset(['train.parquet'], 'go_emotions', TaskType.MULTILABEL), root)
            self.assertEqual(len(emotions.train)+len(emotions.validation), 2)
            self.assertIn((), [row.labels for row in emotions.train+emotions.validation])


if __name__ == '__main__':
    unittest.main()
