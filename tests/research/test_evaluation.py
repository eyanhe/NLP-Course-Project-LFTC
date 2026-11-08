"""Official-Q prediction is a frozen, byte-bound operation, not another search."""
from dataclasses import asdict,replace
import hashlib
import importlib
import json
from pathlib import Path
import tempfile
import unittest

from research_v3.config import canonical_hash,PROJECT_ROOT
from research_v3.freeze import freeze_stage
from research_v3.matrix import write_design_lock
from research_v3.storage import atomic_write_json,hash_file
from research_v3.train import fit_run
from research_v3.types import Example,PreparedDataset,PredictionRow,RunSpec,VisibleInput


def visible(sid,text):
    return VisibleInput(sid,text,0,len(text),len(text.split()),len(text.split()),hashlib.sha256(text.encode()).hexdigest())


class EvaluationTests(unittest.TestCase):
    def api(self):
        try:return importlib.import_module('research_v3.evaluate')
        except ModuleNotFoundError:self.fail('fixed official evaluation implementation is missing')

    def fitted_frozen_fixture(self,root):
        rows=(Example('fixture','T/0','raw hidden negative',(0,),'train','g0'),
              Example('fixture','T/1','raw hidden positive',(1,),'train','g1'))
        data=PreparedDataset('fixture','single_label',('negative','positive'),
            {'train':rows,'select':(),'discovery':(),'test':(),'quarantine':()},'pool')
        inputs={'T/0':visible('T/0','awful bad'),'T/1':visible('T/1','great good')}
        condition={'condition_id':'condition','dataset_id':'fixture','train_ids':['T/0','T/1'],
                   'metadata':{'pool_manifest_hash':'pool'}}
        design={'design_hash':'design','conditions':[condition],'physical_runs':[{'run_id':'run'}],
                'selection_nodes':[],'discovery_pilot_nodes':[]}
        atomic_write_json(root/'matrix/design_manifest.json',design,root)
        run=RunSpec('run','condition','tfidf_lr',None,{'C':1,'device':'cpu'},'input','assets','code',True)
        fit_run(run,data,inputs,{},root/'runs/run')
        examples=(Example('fixture','Q/0','unseen raw tail',(1,),'test','q0'),)
        qinputs={'Q/0':visible('Q/0','great good')}
        import pyarrow as pa
        import pyarrow.parquet as pq
        (root/'data/fixture').mkdir(parents=True);(root/'inputs').mkdir()
        pq.write_table(pa.Table.from_pylist([{'sample_id':'Q/0','labels':[1],'group_id':'q0','official_split':'test'}]),
                       root/'data/fixture/test.parquet')
        pq.write_table(pa.Table.from_pylist([{'partition':'test',**asdict(qinputs['Q/0'])}]),root/'inputs/fixture.parquet')
        meta={'datasets':{'fixture':{'task':'single_label','label_names':['negative','positive'],
            'inputs':{'input_hash':'input','file':'inputs/fixture.parquet'},
            'partitions':{'test':{'rows':1,'file':'data/fixture/test.parquet'}}}}}
        atomic_write_json(root/'data/prepared_manifest.json',meta,root)
        write_design_lock(root,{'design':'matrix/design_manifest.json','prepared':'data/prepared_manifest.json',
                           'q_fixture':'data/fixture/test.parquet','input_fixture':'inputs/fixture.parquet'},
                           {'protocol_hash':'protocol','design_hash':'design'})
        atomic_write_json(root/'run_manifest.json',{'status':'resolved','unresolved_nodes':[],
            'runs':[{'run_id':'run','artifact_id':'run','expected_request_hash':canonical_hash(asdict(run))}]},root)
        source={p.relative_to(PROJECT_ROOT).as_posix():hash_file(p) for p in (PROJECT_ROOT/'research_v3').glob('*.py')}
        # Synthetic previous-stage fixture only; no real research review state is written.
        selection={'schema_version':1,'stage':'selection','protocol_hash':'protocol','design_hash':'design',
            'files':{'runs':{'path':'run_manifest.json','sha256':hash_file(root/'run_manifest.json')}},
            'source_files':source,'source_inventory':sorted(source),'actual_human_pilots_verified':True}
        selection['lock_hash']=canonical_hash(selection)
        atomic_write_json(root/'selection.lock.json',selection,root)
        atomic_write_json(root/'thresholds.json',{'design_hash':'design','runs':{'run':{'operative':None}}},root)
        atomic_write_json(root/'contrasts.json',{'design_hash':'design','items':[{'contrast_id':'fixture'}]},root)
        freeze_stage('evaluation',root,{})
        return run,examples,qinputs

    def test_official_predictions_refuse_missing_global_evaluation_freeze(self):
        api=self.api()
        with tempfile.TemporaryDirectory() as temp:
            run=RunSpec('r','c','tfidf_lr',None,{},'i','a','c',False)
            with self.assertRaisesRegex(PermissionError,'freeze|evaluation'):
                api.predict_run(run,(),{},Path(temp))

    def test_saved_prediction_reconstructs_exact_scores_and_is_reused(self):
        api=self.api()
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);run,rows,inputs=self.fitted_frozen_fixture(root)
            first=api.predict_run(run,rows,inputs,root)
            second=api.predict_run(run,rows,inputs,root)
            self.assertEqual(first,second)
            self.assertEqual(first[0].prediction,(1,))
            self.assertEqual(first[0].group_id,'q0')
            self.assertEqual(len(first[0].scores),2)
            from research_v3.storage import read_json
            info=read_json(root/'predictions/run.manifest.json')
            self.assertIn('checkpoint_to_scores_seconds',info)
            self.assertGreaterEqual(info['checkpoint_to_scores_seconds'],0)

    def test_changed_visible_input_cannot_reuse_old_prediction(self):
        api=self.api()
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);run,rows,inputs=self.fitted_frozen_fixture(root)
            api.predict_run(run,rows,inputs,root)
            changed={'Q/0':visible('Q/0','awful bad')}
            with self.assertRaisesRegex(ValueError,'input|request|cache|identity'):
                api.predict_run(run,rows,changed,root)

    def test_empty_multilabel_prediction_is_valid_but_empty_gold_is_not(self):
        api=self.api()
        row=PredictionRow('r','q','g',(1,),(),(-1.,-2.),'svm_margin','i')
        api.validate_prediction_rows((row,),2,'multilabel')
        with self.assertRaisesRegex(ValueError,'gold|label'):
            api.validate_prediction_rows((replace(row,labels=()),),2,'multilabel')
        with self.assertRaisesRegex(ValueError,'score'):
            api.validate_prediction_rows((replace(row,score_kind='mystery'),),2,'multilabel')
        with self.assertRaisesRegex(ValueError,'duplicate|unique'):
            api.validate_prediction_rows((row,row),2,'multilabel')

    def test_historical_tests_are_not_relabelled_as_unseen(self):
        api=self.api()
        self.assertTrue(hasattr(api,'test_exposure_status'),'test exposure provenance is missing')
        self.assertEqual(api.test_exposure_status('yelp'),'historical_Q_seen_exploratory_replication')
        self.assertEqual(api.test_exposure_status('imdb'),'historical_Q_seen_exploratory_replication')
        self.assertEqual(api.test_exposure_status('go_emotions'),'new_source_first_v3_evaluation')


if __name__=='__main__':unittest.main()
