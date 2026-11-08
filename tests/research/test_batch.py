"""Resolved batches must dispatch exact fits and keep Q/strict holdouts closed."""
from dataclasses import asdict,replace
import hashlib
import importlib
from pathlib import Path
import tempfile
import unittest

from research_v3.config import canonical_hash
from research_v3.storage import atomic_write_json,hash_file
from research_v3.train import learner_code_hash
from research_v3.types import RunSpec


def ids_hash(ids):
    return hashlib.sha256('\n'.join(sorted(ids)).encode()).hexdigest()


class BatchTests(unittest.TestCase):
    def api(self):
        try:return importlib.import_module('research_v3.batch')
        except ModuleNotFoundError:self.fail('resolved batch execution is missing')

    def fixture(self,root,strict=True):
        import pyarrow as pa
        import pyarrow.parquet as pq
        rows=[{'sample_id':f'train/{i}','labels':[i%2],'official_split':'train','group_id':f'g{i}'} for i in range(4)]
        root.joinpath('data').mkdir()
        pq.write_table(pa.Table.from_pylist(rows),root/'data/train.parquet')
        visible=[{'sample_id':row['sample_id'],'partition':'train','text':'awful bad' if i%2==0 else 'great good',
            'start_char':0,'end_char':9 if i%2==0 else 10,'content_tokens':2,'full_tokens':2,
            'text_hash':hashlib.sha256(('awful bad' if i%2==0 else 'great good').encode()).hexdigest()}
            for i,row in enumerate(rows)]
        pq.write_table(pa.Table.from_pylist(visible),root/'data/inputs.parquet')
        meta={'dataset_id':'fixture','task':'single_label','label_names':['negative','positive'],'manifest_hash':'pool',
            'partitions':{'train':{'file':'data/train.parquet','rows':4},
                          'select':{'file':'DO-NOT-READ-V','rows':2},
                          'discovery':{'file':'DO-NOT-READ-D'},'test':{'file':'DO-NOT-READ-Q'}},
            'inputs':{'file':'data/inputs.parquet','input_hash':'input'}}
        ids=['train/0','train/1']
        condition={'condition_id':'condition','dataset_id':'fixture','train_ids':ids,
            'training_set_hash':ids_hash(ids),'metadata':{'pool_manifest_hash':'pool','rows':2}}
        atomic_write_json(root/'matrix/design_manifest.json',{'conditions':[condition]},root)
        hp={'C':1,'ngram_range':[1,2],'device':'cpu','role_ids_hash':{'train':ids_hash(ids),
            'select':ids_hash([]),'discovery':ids_hash([])},'protocol_hash':'protocol','label_count':2}
        run=RunSpec('','condition','tfidf_lr',None,hp,'input','assets',learner_code_hash(),strict)
        run=replace(run,run_id=canonical_hash(asdict(run)))
        entry={'run_id':run.run_id,'artifact_id':run.run_id,'planned_run_id':'planned',
            'run':asdict(run),'engine':'core','expected_model_request_hash':canonical_hash(asdict(run)),
            'dataset_id':'fixture','condition_id':'condition','model_id':'tfidf_lr'}
        return entry,{'datasets':{'fixture':meta}},{'asset_hash':'assets'},meta

    def test_strict_dispatch_reads_only_condition_T_and_reuses_real_verified_checkpoint(self):
        api=self.api()
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);entry,prepared,assets,_=self.fixture(root)
            result=api.fit_declared_run(entry,root,prepared,assets)
            self.assertEqual(result['status'],'complete')
            self.assertEqual(result['training_unique_rows'],2)
            self.assertEqual(result['holdout_labels_accessed'],0)
            second=api.fit_declared_run(entry,root,prepared,assets)
            self.assertTrue(second['reused_complete'])
            self.assertFalse((root/'evaluation-started.json').exists())

    def test_changed_manifest_model_identity_is_rejected_before_fit(self):
        api=self.api()
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);entry,prepared,assets,_=self.fixture(root)
            entry['run']['hparams']['C']=999
            with self.assertRaisesRegex(ValueError,'identity|request'):
                api.fit_declared_run(entry,root,prepared,assets)
            self.assertFalse((root/'runs').exists())

    def test_full_unlabeled_t_representation_uses_every_T_input_and_byte_checked_cache(self):
        api=self.api()
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);_,_,_,meta=self.fixture(root)
            features=api.shared_tfidf_features(root,meta)
            self.assertEqual(set(features.fitted_sample_ids),{f'train/{i}' for i in range(4)})
            self.assertEqual(features.vocabulary,api.shared_tfidf_features(root,meta).vocabulary)
            cache=list((root/'features/shared-tfidf').glob('*/features.joblib'))
            self.assertEqual(len(cache),1)
            cache[0].write_bytes(b'changed learned feature bytes')
            with self.assertRaisesRegex(ValueError,'hash|bytes|changed'):
                api.shared_tfidf_features(root,meta)

    def test_main_batch_requires_real_global_selection_lock_before_fitting(self):
        api=self.api()
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            with self.assertRaises((PermissionError,FileNotFoundError)):
                api.run_resolved_batch({'_protocol_hash':'protocol'},root,root/'missing-authorization.json')
            self.assertFalse((root/'runs').exists())

    def test_aliases_never_duplicate_a_fit_and_conflicting_specs_fail(self):
        api=self.api()
        with tempfile.TemporaryDirectory() as temp:
            import copy
            root=Path(temp);entry,_,_,_=self.fixture(root)
            alias=copy.deepcopy(entry);alias['planned_run_id']='other-plan-alias'
            self.assertEqual(len(api.unique_fit_entries({'status':'resolved','runs':[entry,alias]})),1)
            alias['run']['hparams']['C']=2
            with self.assertRaisesRegex(ValueError,'identity|conflict|request'):
                api.unique_fit_entries({'status':'resolved','runs':[entry,alias]})


if __name__=='__main__':unittest.main()
