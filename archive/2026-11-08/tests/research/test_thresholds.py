"""Threshold tests use CPU scores and authenticated temporary fixture files."""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import importlib
import tempfile
from pathlib import Path
import unittest

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from research_v3.config import canonical_hash
from research_v3.storage import atomic_write_json, hash_file, read_json
from research_v3.types import RunSpec


def ids_hash(ids):
    return hashlib.sha256('\n'.join(sorted(ids)).encode()).hexdigest()


class ThresholdTests(unittest.TestCase):
    def api(self):
        try:
            return importlib.import_module('research_v3.thresholds')
        except ModuleNotFoundError:
            self.fail('global threshold selection/freezing is not implemented')

    def test_global_candidates_and_masked_objective_preserve_fixed_operation(self):
        y = np.array([[0,1],[1,0],[0,1],[1,0]])
        scores = np.array([[.1,.95],[.3,.9],[.2,.85],[.4,.8]])
        result = self.api().choose_global_threshold(y,scores,'probability',[0])
        expected = sorted(set([.5,*np.percentile(scores,np.arange(5,100,5)).tolist()]))
        self.assertEqual(result['candidates'],expected)
        self.assertEqual(result['operative'],.5)
        self.assertLess(result['chosen'],.5)
        self.assertEqual(result['label_mask'],[0])
        self.assertEqual(len(result['metrics']),len(expected))
        from sklearn.metrics import f1_score
        values = {row['threshold']:row['masked_macro_f1'] for row in result['metrics']}
        for cutoff in expected:
            self.assertAlmostEqual(values[cutoff],f1_score(y[:,0],scores[:,0]>=cutoff,zero_division=0))
        self.assertEqual(result['chosen'],min(expected,key=lambda x:(-values[x],abs(x-.5),-x)))

    def test_tied_optima_choose_nearest_primary_and_score_kinds_are_explicit(self):
        y = [[0],[1]]
        for kind,scores,operative in [('probability',[[.1],[.9]],.5),
                                       ('logit',[[-1],[1]],0.),('svm_margin',[[-1],[1]],0.)]:
            result = self.api().choose_global_threshold(y,scores,kind,[0])
            self.assertEqual(result['chosen'],operative)
            self.assertEqual(result['operative'],operative)

    def test_exact_equidistant_optima_choose_the_larger_global_cutoff(self):
        scores = (np.arange(8,dtype=float)-3.5)[:,None]
        labels = np.array([0,0,1,0,0,0,1,0])[:,None]
        result = self.api().choose_global_threshold(labels,scores,'logit',[0])
        values = {r['threshold']:r['masked_macro_f1'] for r in result['metrics']}
        self.assertEqual(values[-1.75],values[1.75])
        self.assertGreater(values[1.75],values[0.])
        self.assertEqual(result['chosen'],1.75)

    def test_invalid_scores_targets_and_frozen_mask_fail(self):
        api = self.api()
        cases = [([[0],[1]],[[.1],[float('nan')]],'probability',[0]),
                 ([[0],[1]],[[.1],[1.1]],'probability',[0]),
                 ([[0],[1]],[[.1],[.9]],'unknown',[0]),
                 ([[0],[1]],[[.1],[.9]],'probability',[]),
                 ([[0],[1]],[[.1],[.9]],'probability',[0,0]),
                 ([[0],[0]],[[.1],[.9]],'probability',[0]),
                 ([[0],[2]],[[.1],[.9]],'probability',[0])]
        for args in cases:
            with self.subTest(args=args),self.assertRaises(ValueError):
                api.choose_global_threshold(*args)

    def fixture(self, root, *, strict=False, task='multilabel', score_kind='probability'):
        root = Path(root); (root/'data').mkdir(exist_ok=True)
        source = [{'sample_id':f'v{i}','labels':[i%2],'official_split':'validation','group_id':f'g{i}'}
                  for i in range(4)]
        visible = [{'sample_id':f'v{i}','partition':'select','text':f'text {i}','start_char':0,
                    'end_char':6,'content_tokens':2,'full_tokens':2,
                    'text_hash':hashlib.sha256(f'text {i}'.encode()).hexdigest()} for i in range(4)]
        pq.write_table(pa.Table.from_pylist(source),root/'data/select.parquet')
        pq.write_table(pa.Table.from_pylist(visible),root/'data/input.parquet')
        meta = {'dataset_id':'fixture','task':task,'label_names':['a','b'],'manifest_hash':'pool',
                'partitions':{'select':{'file':'data/select.parquet','sha256':hash_file(root/'data/select.parquet'),
                                        'rows':4,'ap_mask':[0,1]}},
                'inputs':{'file':'data/input.parquet','file_sha256':hash_file(root/'data/input.parquet'),
                          'input_hash':'input'}}
        hp = {'target_task':task,'role_ids_hash':{'train':'train','select':ids_hash([] if strict else ['v0','v1','v2','v3']),
                                                 'discovery':ids_hash([])}}
        spec = RunSpec('','condition','tfidf_lr',None,hp,'input','asset','code',strict)
        run = asdict(spec); run['run_id'] = canonical_hash(run)
        artifact_id = run['run_id']; output = root/'runs'/artifact_id; output.mkdir(parents=True,exist_ok=True)
        (output/'bundle.joblib').write_bytes(b'CPU fixture model; injected scorer only')
        status = {'status':'complete','reload_verified':True,'run_id':artifact_id,'run':run,
                  'request_hash':canonical_hash(run),'score_kind':score_kind,
                  'files':{'bundle.joblib':hash_file(output/'bundle.joblib')}}
        atomic_write_json(output/'status.json',status,root)
        entry = {'planned_run_id':'planned','artifact_id':artifact_id,'run_id':artifact_id,
                 'run':run,'dataset_id':'fixture','expected_model_request_hash':canonical_hash(run)}
        manifest = {'status':'resolved','design_hash':'design','runs':[entry]}
        scores = np.array([[.2,.7],[.8,.1],[.1,.8],[.7,.2]])
        return manifest,{'datasets':{'fixture':meta}},status,scores

    def test_freeze_covers_unique_artifacts_and_records_actual_v_provenance(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); manifest,prepared,status,scores = self.fixture(root)
            manifest['runs'].append({**manifest['runs'][0],'planned_run_id':'alias'})
            calls = []
            def score(output, inputs):
                calls.append((output,[i.sample_id for i in inputs])); return scores
            result = self.api().freeze_thresholds(root,manifest,prepared,scorer=score)
            self.assertEqual(len(calls),1)
            self.assertEqual(calls[0][1],['v0','v1','v2','v3'])
            self.assertEqual(set(result['runs']),{status['run_id']})
            self.assertEqual(result['design_hash'],'design')
            record = result['runs'][status['run_id']]
            self.assertEqual(record['operative'],.5)
            self.assertEqual(record['selection']['partition'],'select')
            self.assertEqual(record['selection']['ids_hash'],ids_hash(calls[0][1]))
            self.assertEqual(record['selection']['input_hash'],'input')
            self.assertEqual(record['auxiliary']['label_mask'],[0,1])
            self.assertEqual(result,read_json(root/'thresholds.json'))

    def test_strict_and_single_label_never_load_validation_or_score(self):
        for strict,task in [(True,'multilabel'),(True,'single_label'),(False,'single_label')]:
            with self.subTest(strict=strict,task=task), tempfile.TemporaryDirectory() as temp:
                manifest,prepared,status,_ = self.fixture(temp,strict=strict,task=task)
                prepared['datasets']['fixture'].pop('partitions')
                def forbidden(*args,**kwargs):
                    raise AssertionError('V/D was accessed')
                result = self.api().freeze_thresholds(temp,manifest,prepared,loader=forbidden,scorer=forbidden)
                record = result['runs'][status['run_id']]
                self.assertIsNone(record['auxiliary'])
                self.assertEqual(record['operative'],.5 if task=='multilabel' else None)
                self.assertEqual(record['selection_labels_accessed'],0)

    def test_pending_reload_tamper_and_wrong_model_identity_block_before_v_access(self):
        for mutation in ['pending','reload','artifact','run_identity','expected_identity']:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temp:
                root = Path(temp); manifest,prepared,status,_ = self.fixture(root)
                output = root/'runs'/status['run_id']
                if mutation=='pending': status['status']='running'
                elif mutation=='reload': status['reload_verified']=False
                elif mutation=='artifact': (output/'bundle.joblib').write_bytes(b'tampered')
                elif mutation=='run_identity': status['run']['code_hash']='different'
                else: manifest['runs'][0]['expected_model_request_hash']='different'
                atomic_write_json(output/'status.json',status,root)
                def forbidden(*args,**kwargs): raise AssertionError('V loaded before checkpoint validation')
                with self.assertRaises((ValueError,PermissionError)):
                    self.api().freeze_thresholds(root,manifest,prepared,loader=forbidden,scorer=forbidden)
                self.assertFalse((root/'thresholds.json').exists())

    def test_entire_checkpoint_set_is_preflighted_before_any_v_read(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); manifest,prepared,status,_ = self.fixture(root)
            pending = deepcopy(manifest['runs'][0]); pending['artifact_id']='pending'
            pending['run_id']='pending'; pending['planned_run_id']='second'
            manifest['runs'].append(pending)
            def forbidden(*args,**kwargs): raise AssertionError('V loaded before all checkpoints were checked')
            with self.assertRaisesRegex(PermissionError,'pending'):
                self.api().freeze_thresholds(root,manifest,prepared,loader=forbidden,scorer=forbidden)
            self.assertFalse((root/'thresholds.json').exists())

    def test_checkpoint_change_during_scoring_prevents_threshold_write(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); manifest,prepared,status,scores = self.fixture(root)
            def score(output,inputs):
                (output/'bundle.joblib').write_bytes(b'changed while scoring')
                return scores
            with self.assertRaisesRegex(ValueError,'artifact hash'):
                self.api().freeze_thresholds(root,manifest,prepared,scorer=score)
            self.assertFalse((root/'thresholds.json').exists())

    def test_wrong_v_ids_input_identity_and_source_bytes_are_rejected(self):
        for mutation in ['ids','input_identity','select_bytes','input_bytes']:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temp:
                root = Path(temp); manifest,prepared,status,scores = self.fixture(root)
                if mutation=='ids':
                    # Authenticated model request declares different accessible V IDs.
                    status['run']['hparams']['role_ids_hash']['select']='wrong'
                    status['request_hash']=canonical_hash(status['run'])
                    manifest['runs'][0]['run']=deepcopy(status['run'])
                    manifest['runs'][0]['expected_model_request_hash']=canonical_hash(status['run'])
                    atomic_write_json(root/'runs'/status['run_id']/'status.json',status,root)
                elif mutation=='input_identity': prepared['datasets']['fixture']['inputs']['input_hash']='other'
                elif mutation=='select_bytes': (root/'data/select.parquet').write_bytes(b'changed')
                else: (root/'data/input.parquet').write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError,'identity|hash|IDs|bytes'):
                    self.api().freeze_thresholds(root,manifest,prepared,scorer=lambda *args:scores)
                self.assertFalse((root/'thresholds.json').exists())


if __name__=='__main__':
    unittest.main()
