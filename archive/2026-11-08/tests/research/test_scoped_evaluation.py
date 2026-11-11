from copy import deepcopy
from dataclasses import asdict
import hashlib
import importlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from research_v3.config import canonical_hash
from research_v3.storage import atomic_write_json,hash_file,read_json
from research_v3 import convergence as driver


class ScopedEvaluationTests(unittest.TestCase):
    def api(self):
        try:return importlib.import_module('research_v3.scoped_evaluation')
        except ModuleNotFoundError:self.fail('scoped evaluation implementation is missing')

    def fixture(self,root):
        import pyarrow as pa
        import pyarrow.parquet as pq
        real=Path(__file__).resolve().parents[2]/'research-v3'
        proposal=read_json(real/driver.REL/'proposal.json')
        prepared=read_json(real/'data/prepared_manifest.json');datasets={}
        for spec in proposal['scope']:
            ds=spec['dataset_id'];meta=deepcopy(prepared['datasets'][ds]);folder=root/'data'/ds;folder.mkdir(parents=True)
            rows=[];visible=[]
            for i in range(2):
                sid=f'{ds}/test/{i}';text=f'hello {i}';digest=hashlib.sha256(text.encode()).hexdigest()
                rows.append({'sample_id':sid,'group_id':f'{ds}/g{i}','labels':[i],'official_split':'test'})
                visible.append({'sample_id':sid,'text':text,'start_char':0,'end_char':len(text),
                    'content_tokens':2,'full_tokens':2,'text_hash':digest,'partition':'test'})
            q=folder/'test.parquet';inp=folder/'inputs.parquet';pq.write_table(pa.Table.from_pylist(rows),q);pq.write_table(pa.Table.from_pylist(visible),inp)
            meta['partitions']['test'].update(file=q.relative_to(root).as_posix(),sha256=hash_file(q),rows=2,ap_mask=[0,1])
            meta['inputs'].update(file=inp.relative_to(root).as_posix(),file_sha256=hash_file(inp))
            for purpose in ['train','select']:
                path=folder/(purpose+'.parquet');path.write_bytes(b'fixture-not-evaluated')
                meta['partitions'][purpose].update(file=path.relative_to(root).as_posix(),sha256=hash_file(path))
            spec.update(test_rows=2,test_source=deepcopy(meta['partitions']['test']))
            datasets[ds]=meta
        atomic_write_json(root/'data/prepared_manifest.json',{'datasets':datasets},root)
        atomic_write_json(root/'matrix/design_manifest.json',{'fixture':True},root)
        assets=[];asset_dir=root/'fixture-model-assets';asset_dir.mkdir()
        for name in ['config.json','tokenizer_config.json','tokenizer.json','vocab.txt','model.safetensors']:
            path=asset_dir/name;path.write_bytes(b'fixture-model-asset')
            assets.append({'model_id':'distilbert','source_file':name,'path':path.relative_to(root.parent).as_posix(),'sha256':hash_file(path)})
        atomic_write_json(root/'assets/assets.lock.json',{'files':assets,'models':{'distilbert':{'files':[r['source_file'] for r in assets]}}},root)
        atomic_write_json(root/driver.REL/'proposal.json',proposal,root)
        manifest=driver.prepare(root,proposal,'cpu')
        def save(entry,value=.5):
            out=root/'runs'/entry['run_id'];out.mkdir(parents=True,exist_ok=True)
            (out/'bundle.joblib').write_bytes(b'isolated-fixture')
            run=entry['run'];status={'status':'complete','run_id':entry['run_id'],'run':run,
                'request_hash':canonical_hash(run),'reload_verified':True,'actual_model_training_run':True,
                'files':{'bundle.joblib':hash_file(out/'bundle.joblib')},'score_kind':'svm_margin' if run['model_id']=='tfidf_svm' else 'logit',
                'training_ids_sha256':run['hparams']['role_ids_hash']['train'],
                'convergence_verified':True,'stop_reason':'patience','validation':{'macro_f1':value,'macro_ap':value}}
            atomic_write_json(out/'status.json',status,root)
            if run['model_id']=='tfidf_svm':atomic_write_json(root/driver.REL/'solver-audits'/(entry['run_id']+'.json'),
                {'solver_verified':True,'status_sha256':hash_file(out/'status.json')},root)
        for index,entry in enumerate(manifest['candidates']):save(entry,.3+index/100)
        driver.update_selections(root,manifest)
        for entry in driver._all_entries(root,manifest):
            if entry['phase']=='confirmation':save(entry)
        for ref in proposal['fixed_budget_references']:
            source=next(b['source_run'] for b in proposal['source_bindings'] if b['source_run_id']==ref['run_id'])
            save({'run_id':ref['run_id'],'run':source})
        return manifest

    def test_freeze_never_loads_Q_and_missing_candidate_blocks(self):
        api=self.api()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);manifest=self.fixture(root)
            with patch.object(api,'_load_Q',side_effect=AssertionError('premature Q')):
                scope=api.freeze_scope(root)
            self.assertEqual(len(scope['models']),18)
            self.assertEqual(scope['primary_models'],14)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);manifest=self.fixture(root)
            (root/'runs'/manifest['candidates'][0]['run_id']/'status.json').unlink()
            with patch.object(api,'_load_Q',side_effect=AssertionError('premature Q')):
                with self.assertRaises(PermissionError):api.freeze_scope(root)

    def test_changed_checkpoint_is_rejected_before_Q_loader(self):
        api=self.api()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.fixture(root);scope=api.freeze_scope(root)
            (root/'runs'/scope['models'][0]['run_id']/'bundle.joblib').write_bytes(b'changed')
            with patch.object(api,'_load_Q',side_effect=AssertionError('premature Q')):
                with self.assertRaises(ValueError):api.predict_scope(root)

    def test_complete_predictions_reconstruct_metrics_and_cache(self):
        api=self.api()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.fixture(root);api.freeze_scope(root);calls=[]
            def scorer(output,inputs):
                status=read_json(output/'status.json');c=status['run']['hparams']['label_count']
                scores=np.zeros((len(inputs),c),dtype=np.float64)
                for i,row in enumerate(inputs):scores[i,int(row.sample_id.rsplit('/',1)[1])]=2.
                calls.append(status['run_id']);return scores,status
            result=api.predict_scope(root,scorer=scorer)
            self.assertEqual(result['status'],'complete')
            self.assertEqual(len(result['models']),18)
            self.assertEqual(len(calls),18)
            second=api.predict_scope(root,scorer=scorer)
            self.assertEqual(len(calls),18)
            self.assertEqual(result['models'],second['models'])
            first=result['models'][0];path=root/first['path'];info_path=path.with_suffix('.manifest.json')
            timing_original=read_json(info_path);timing_tampered=deepcopy(timing_original)
            timing_tampered['timing']['authenticated_inference_seconds']=999
            atomic_write_json(info_path,timing_tampered,root)
            with self.assertRaises(ValueError):api.predict_scope(root,scorer=scorer)
            atomic_write_json(info_path,timing_original,root)
            original=read_json(info_path);tampered=deepcopy(original);tampered['metrics']['rows']=999
            atomic_write_json(info_path,tampered,root)
            with self.assertRaises(ValueError):api.predict_scope(root,scorer=scorer)
            atomic_write_json(info_path,original,root)
            import pyarrow as pa
            import pyarrow.parquet as pq
            table=pq.read_table(path);rows=table.to_pylist();rows[0]['run_id']='wrong-run'
            pq.write_table(pa.Table.from_pylist(rows,schema=table.schema),path)
            original['sha256']=hash_file(path);atomic_write_json(info_path,original,root)
            with self.assertRaises(ValueError):api.predict_scope(root,scorer=scorer)

    def test_changed_tokenizer_is_rejected_before_Q(self):
        api=self.api()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.fixture(root);api.freeze_scope(root)
            (root/'fixture-model-assets/tokenizer.json').write_bytes(b'changed tokenizer')
            with patch.object(api,'_load_Q',side_effect=AssertionError('premature Q')):
                with self.assertRaises(ValueError):api.predict_scope(root)

    def test_self_reported_selection_metric_cannot_replace_declared_metric(self):
        api=self.api()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);manifest=self.fixture(root)
            path=next((root/driver.REL/'selections').glob('*.json'));selection=read_json(path)
            for entry in manifest['candidates']:
                if entry['group']!=selection['group']:continue
                p=root/'runs'/entry['run_id']/'status.json';status=read_json(p)
                status['validation']['micro_f1']=status['validation'][selection['metric']]
                atomic_write_json(p,status,root);selection['candidate_status_sha256'][entry['run_id']]=hash_file(p)
                if entry['model_id']=='tfidf_svm':atomic_write_json(root/driver.REL/'solver-audits'/(entry['run_id']+'.json'),
                    {'solver_verified':True,'status_sha256':hash_file(p)},root)
            selection['metric']='micro_f1'
            selection['selection_hash']=canonical_hash({k:v for k,v in selection.items() if k!='selection_hash'})
            atomic_write_json(path,selection,root)
            with self.assertRaises(ValueError):api.freeze_scope(root)


if __name__=='__main__':unittest.main()
