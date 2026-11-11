"""Prepared artifacts preserve labels and refuse changed bytes."""
import importlib
from pathlib import Path
import tempfile
import unittest

from research_v3.types import Example, PreparedDataset


class PreparationTests(unittest.TestCase):
    def api(self):
        try:
            return importlib.import_module('research_v3.preparation')
        except ModuleNotFoundError:
            self.fail('prepared artifact persistence has not been implemented')

    def data(self):
        partitions = {name: () for name in ['train','select','discovery','test','quarantine']}
        partitions['train'] = (Example('go_emotions','go/train/0','No! 🙂', (0,2), 'train', 'g0', {'raw':'kept'}),)
        partitions['select'] = (Example('go_emotions','go/val/0','x', (0,), 'validation','g1'),
                                Example('go_emotions','go/val/1','y', (1,), 'validation','g2'))
        return PreparedDataset('go_emotions','multi_label',('a','b','neutral'),partitions,'fixture-manifest')

    def test_roundtrip_and_separate_positive_negative_ap_support(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            meta=self.api().save_prepared_dataset(self.data(),root)
            restored=self.api().load_prepared_dataset(root,meta,include_text=True)
            self.assertEqual(restored,self.data())
            self.assertEqual(meta['partitions']['select']['ap_mask'],[0,1])
            self.assertEqual(meta['partitions']['train']['label_support'],[1,0,1])
            self.assertEqual(meta['partitions']['select']['rows'],2)

    def test_corrupted_partition_cannot_be_loaded_as_prepared(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            meta=self.api().save_prepared_dataset(self.data(),root)
            path=root/meta['partitions']['train']['file']
            path.write_bytes(b'not the frozen parquet')
            with self.assertRaisesRegex(ValueError,'hash|SHA|changed'):
                self.api().load_prepared_dataset(root,meta)

    def test_group_overlap_across_datasets_is_rejected(self):
        data=self.data()
        other=PreparedDataset('other','single_label',('a',),
            {'train':(), 'select':(), 'discovery':(), 'test':
             (Example('other','other/test/0','x',(0,),'test','g0'),), 'quarantine':()},'other')
        with self.assertRaisesRegex(ValueError,'group|overlap'):
            self.api().audit_global_partition_groups({'go':data,'other':other})


if __name__=='__main__':
    unittest.main()
