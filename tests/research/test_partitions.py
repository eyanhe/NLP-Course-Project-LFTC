"""Catch leakage through duplicate chains and quarantine scoped only to test."""
import importlib
import json
from pathlib import Path
import unittest

from research_v3.types import Example

ROOT = Path(__file__).resolve().parents[2]


class PartitionTests(unittest.TestCase):
    def api(self):
        try:
            return importlib.import_module('research_v3.partitions')
        except ModuleNotFoundError:
            self.fail('research_v3.partitions is not implemented')

    def protocol(self):
        p = json.loads((ROOT / 'configs/research-v3.json').read_text(encoding='utf-8-sig'))
        p['_dataset_registry'] = {'banking77': {'task': 'single_label', 'label_names': ['c0', 'c1']}}
        return p

    def test_exact_train_test_overlap_retains_only_test(self):
        rows = (Example('banking77', 'banking77/train/0', 'The same text', (0,), 'train'),
                Example('banking77', 'banking77/test/0', 'The same text', (0,), 'test'))
        data = self.api().prepare_partitions(rows, self.protocol())['banking77']
        self.assertEqual(len(data.partitions['test']), 1)
        self.assertEqual(len(data.partitions['train']), 0)
        self.assertEqual(len(data.partitions['select']), 0)
        self.assertEqual(len(data.partitions['discovery']), 0)

    def test_exact_conflicting_labels_quarantine_every_partition(self):
        rows = (Example('banking77', 'banking77/train/0', 'Ambiguous same text', (0,), 'train'),
                Example('banking77', 'banking77/test/0', 'Ambiguous same text', (1,), 'test'))
        data = self.api().prepare_partitions(rows, self.protocol())['banking77']
        self.assertEqual(len(data.partitions['quarantine']), 2)
        self.assertEqual(sum(len(data.partitions[k]) for k in ['train','select','discovery','test']), 0)

    def test_verified_near_duplicate_chain_is_one_component(self):
        a = [f'w{i}' for i in range(100)]
        b = a.copy(); b[40] = 'replacementone'
        c = b.copy(); c[60] = 'replacementtwo'
        rows = tuple(Example('banking77', f'banking77/train/{i}', ' '.join(words), (i % 2,), 'train')
                     for i, words in enumerate([a, b, c]))
        groups = self.api().build_duplicate_groups(rows, 2026)
        self.assertEqual(len(set(groups.values())), 1)

    def test_banking_holdout_quota_is_per_class_and_groups_disjoint(self):
        rows = tuple(Example('banking77', f'banking77/train/{c}-{i}', f'unique short {c} sample {i}', (c,), 'train')
                     for c in [0, 1] for i in range(20))
        data = self.api().prepare_partitions(rows, self.protocol())['banking77']
        self.assertEqual(len(data.partitions['select']), 4)
        self.assertEqual(len(data.partitions['discovery']), 4)
        self.assertEqual(len(data.partitions['train']), 32)
        group_sets = [set(r.group_id for r in data.partitions[k]) for k in ['train','select','discovery','test']]
        for i, a in enumerate(group_sets):
            for b in group_sets[i+1:]:
                self.assertFalse(a & b)


if __name__ == '__main__':
    unittest.main()
