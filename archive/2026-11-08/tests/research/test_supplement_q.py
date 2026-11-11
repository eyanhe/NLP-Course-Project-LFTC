from copy import deepcopy
import importlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from research_v3.config import canonical_hash
from research_v3.storage import atomic_write_json,hash_file,read_json
from tests.research import test_scoped_evaluation as original_fixture


class SupplementQTests(unittest.TestCase):
    def api(self):
        return importlib.import_module('research_v3.supplement_q')

    def fixture(self,root):
        original_fixture.ScopedEvaluationTests().fixture(root)
        real=Path(__file__).resolve().parents[2]/'research-v3'
        manifest=read_json(real/'reports/overnight/supplement-1-4/manifest.json')
        atomic_write_json(root/'reports/overnight/supplement-1-4/manifest.json',manifest,root)
        design=read_json(real/'matrix/design_manifest.json')
        atomic_write_json(root/'matrix/design_manifest.json',design,root)
        for item in manifest['fits']:
            entry=item['entry'];run=entry['run'];out=root/'runs'/entry['run_id'];out.mkdir(parents=True)
            (out/'bundle.joblib').write_bytes(b'isolated-supplement-fixture')
            status={'status':'complete','run_id':entry['run_id'],'run':run,'request_hash':canonical_hash(run),
                'reload_verified':True,'actual_model_training_run':True,'training_ids_sha256':run['hparams']['role_ids_hash']['train'],
                'files':{'bundle.joblib':hash_file(out/'bundle.joblib')},
                'score_kind':'svm_margin' if item['model_id']=='fixed_tfidf_svm' else 'logit'}
            if item['supplement_id']=='S3':
                identity={'model_request_hash':canonical_hash(run),'diagnostics_code_hash':run['code_hash'],
                    'runtime_resource_fingerprint':canonical_hash('isolated-runtime')}
                status.update(identity,request_hash=canonical_hash(identity))
            atomic_write_json(out/'status.json',status,root)
        self.api().authorize(root)
        return manifest

    def scorer(self,output,inputs):
        status=read_json(output/'status.json');c=status['run']['hparams']['label_count']
        scores=np.full((len(inputs),c),-2.)
        for i,row in enumerate(inputs):scores[i,int(row.sample_id.rsplit('/',1)[1])]=2.
        timing={'predict_seconds':.01,'rows':len(inputs),'measurement':'fixture'}
        return scores,status,timing

    def test_freezes_exact_stage_and_original_scopes_unchanged_without_loading_Q(self):
        api=self.api()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);manifest=self.fixture(root)
            source=root/'reports/overnight/supplement-1-4/manifest.json';before=hash_file(source)
            with patch.object(api,'_load_Q',side_effect=AssertionError('premature Q')):
                s1=api.freeze(root,'S1');s3=api.freeze(root,'S3')
            self.assertEqual((len(s1['models']),len(s3['models'])),(16,8))
            self.assertFalse(set(e['run_id'] for e in s1['models'])&set(e['run_id'] for e in s3['models']))
            self.assertEqual(hash_file(source),before)
            self.assertTrue(all(e['tail_labels'] for e in s3['models']))
            self.assertFalse((root/api.folder('S1')/'evaluation-started.json').exists())

    def test_missing_or_changed_model_blocks_before_Q(self):
        api=self.api()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);manifest=self.fixture(root);lock=api.freeze(root,'S3')
            (root/'runs'/lock['models'][0]['run_id']/'bundle.joblib').write_bytes(b'changed')
            with patch.object(api,'_load_Q',side_effect=AssertionError('premature Q')):
                with self.assertRaises(ValueError):api.predict(root,'S3',scorer=self.scorer)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);manifest=self.fixture(root)
            (root/'runs'/manifest['fits'][0]['entry']['run_id']/'status.json').unlink()
            with self.assertRaises((ValueError,PermissionError)):api.freeze(root,'S1')

    def test_predict_stage_paths_cache_metrics_and_timing(self):
        api=self.api()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.fixture(root);calls=[]
            def scorer(output,inputs):calls.append(output.name);return self.scorer(output,inputs)
            with patch.dict('os.environ',{'SLURM_JOB_ID':'fixture-only'}):
                s3=api.run(root,'S3',scorer=scorer);s1=api.run(root,'S1',scorer=scorer)
            self.assertEqual((len(s3['models']),len(s1['models'])),(8,16));self.assertEqual(len(calls),24)
            again=api.predict(root,'S3',scorer=scorer);self.assertEqual(len(calls),24)
            self.assertEqual(s3['models'],again['models'])
            self.assertTrue(all(api.folder('S3').as_posix() in m['path'] for m in s3['models']))
            self.assertTrue(all('tail_macro_f1' in m['diagnostics'] for m in s3['models']))
            self.assertTrue(all(m['metrics']['rows']==2 for m in s1['models']))
            p=root/s3['models'][0]['path'];p=p.with_suffix('.manifest.json');info=read_json(p)
            info['timing']['predict_seconds']=999;atomic_write_json(p,info,root)
            with self.assertRaises(ValueError):api.predict(root,'S3',scorer=scorer)

    def test_extra_authorized_model_or_relabelled_subset_is_rejected(self):
        api=self.api()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.fixture(root);path=root/api.AUTH;auth=read_json(path)
            auth['stages']['S3'].append(deepcopy(auth['stages']['S1'][0]))
            auth['authorization_hash']=canonical_hash({k:v for k,v in auth.items() if k!='authorization_hash'})
            atomic_write_json(path,auth,root)
            with self.assertRaises(ValueError):api.freeze(root,'S3')

    def test_authorization_cannot_redefine_labels_or_threshold_rule(self):
        api=self.api()
        for change in ['labels','threshold']:
            with self.subTest(change=change),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);self.fixture(root);path=root/api.AUTH;auth=read_json(path)
                if change=='labels':auth['metadata']['banking77']['label_names'].reverse()
                else:auth['threshold_rule']='choose thresholds on Q'
                auth['authorization_hash']=canonical_hash({k:v for k,v in auth.items() if k!='authorization_hash'})
                atomic_write_json(path,auth,root)
                with self.assertRaises(ValueError):api.freeze(root,'S3')

    def test_diagnostic_model_identity_cannot_be_rebound(self):
        api=self.api()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);manifest=self.fixture(root)
            item=next(r for r in manifest['fits'] if r['supplement_id']=='S3')
            path=root/'runs'/item['entry']['run_id']/'status.json';status=read_json(path)
            status['model_request_hash']=canonical_hash('other-model')
            status['request_hash']=canonical_hash({k:status[k] for k in ['model_request_hash','diagnostics_code_hash','runtime_resource_fingerprint']})
            atomic_write_json(path,status,root)
            with self.assertRaises(ValueError):api.freeze(root,'S3')

    def test_rehashed_freeze_cannot_change_model_or_label_groups(self):
        api=self.api()
        for change in ['model','tail','metadata']:
            with self.subTest(change=change),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);self.fixture(root)
                # S1 exercises the current semantic-freeze gap without diagnostic identity parsing.
                lock=api.freeze(root,'S1');path=root/api.folder('S1')/'evaluation.freeze.json'
                if change=='model':lock['models'][0]=deepcopy(lock['models'][1])
                elif change=='tail':lock['models'][0]['tail_labels']=[0]
                else:lock['metadata']['banking77']['label_names'].reverse()
                lock['lock_hash']=canonical_hash({k:v for k,v in lock.items() if k!='lock_hash'})
                atomic_write_json(path,lock,root)
                with patch.object(api,'_load_Q',side_effect=AssertionError('premature Q')):
                    with self.assertRaises(ValueError):api.predict(root,'S1',scorer=self.scorer)


if __name__=='__main__':unittest.main()
