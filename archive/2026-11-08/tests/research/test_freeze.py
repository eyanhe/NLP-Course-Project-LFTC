"""Global phase locks must bind bytes and reject unresolved training/pilots."""
import importlib
from pathlib import Path
import tempfile
import unittest

from research_v3.config import canonical_hash,PROJECT_ROOT
from research_v3.matrix import write_design_lock
from research_v3.storage import atomic_write_json,hash_file


class FreezeTests(unittest.TestCase):
    def api(self):
        try:return importlib.import_module('research_v3.freeze')
        except ModuleNotFoundError:self.fail('global phase freeze implementation is missing')

    def test_modern_budget_must_match_the_exact_frozen_request(self):
        api=self.api();self.assertTrue(hasattr(api,'verify_token_budget'),'Modern budget freeze validation is missing')
        with tempfile.TemporaryDirectory() as temp:
            from dataclasses import asdict
            from tests.research.test_costs import CostTests
            from research_v3.costs import record_token_budget
            from research_v3.types import RunSpec
            import hashlib
            root=Path(temp);_,train,validation,directory=CostTests().fixture(root)
            ih=lambda ids:hashlib.sha256('\n'.join(sorted(ids)).encode()).hexdigest()
            spec=RunSpec('modern','condition','modernbert',42,{'role_ids_hash':{
                'train':ih(['T/0']),'select':ih(['V/0'])}},'input','assets','code',False)
            run=asdict(spec)
            entry={'run_id':'modern','run':run}
            with self.assertRaisesRegex(PermissionError,'budget|token'):
                api.verify_token_budget(root,entry,{})
            assets={'asset_hash':'assets','models':{'modernbert':{'directory':str(directory),'revision':'a'*40}}}
            atomic_write_json(root/'assets/assets.lock.json',assets,root)
            budget=record_token_budget(spec,train,validation,assets,root)
            path=root/'training/token-budgets/modern.json';atomic_write_json(path,budget,root)
            api.verify_token_budget(root,entry,{})
            budget['run_request_hash']='wrong';budget['record_hash']=canonical_hash({k:v for k,v in budget.items() if k!='record_hash'})
            atomic_write_json(path,budget,root)
            with self.assertRaisesRegex(ValueError,'identity|request'):
                api.verify_token_budget(root,entry,{})

    def test_modern_budget_header_without_actual_token_counts_is_rejected(self):
        api=self.api()
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);run={'run_id':'modern','model_id':'modernbert','input_hash':'input','asset_hash':'assets'}
            budget={'run_id':'modern','run_request_hash':canonical_hash(run),'input_hash':'input','asset_hash':'assets'}
            budget['record_hash']=canonical_hash(budget)
            atomic_write_json(root/'training/token-budgets/modern.json',budget,root)
            with self.assertRaisesRegex(ValueError,'budget|token|schema'):
                api.verify_token_budget(root,{'run_id':'modern','run':run},{})

    def base(self,root,runs=()):
        design={'design_hash':'design','selection_nodes':[],
                'discovery_pilot_nodes':[{'pilot_id':p} for p in ['p1','p2','p3']],
                'physical_runs':[{'run_id':r['run_id']} for r in runs]}
        atomic_write_json(root/'matrix/design_manifest.json',design,root)
        write_design_lock(root,{'design':'matrix/design_manifest.json'},
                           {'protocol_hash':'protocol','design_hash':'design'})

    def synthetic_selection_fixture(self,root,run_manifest):
        self.base(root,run_manifest['runs'])
        atomic_write_json(root/'run_manifest.json',run_manifest,root)
        source={p.relative_to(PROJECT_ROOT).as_posix():hash_file(p)
                for p in (PROJECT_ROOT/'research_v3').glob('*.py')}
        lock={'schema_version':1,'stage':'selection','protocol_hash':'protocol','design_hash':'design',
              'files':{'run_manifest':{'path':'run_manifest.json','sha256':hash_file(root/'run_manifest.json')}},
              'source_files':source,'source_inventory':sorted(source),
              'actual_human_pilots_verified':True,'official_test_predictions_allowed':False}
        lock['lock_hash']=canonical_hash(lock)
        atomic_write_json(root/'selection.lock.json',lock,root)

    def test_pending_real_discovery_pilots_cannot_form_selection_lock(self):
        api=self.api()
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);self.base(root)
            with self.assertRaisesRegex(PermissionError,'pilot|human|试点'):
                api.freeze_stage('selection',root,{'choices':{},'discovery_pilots':{}})
            self.assertFalse((root/'selection.lock.json').exists())

    def test_incomplete_checkpoint_blocks_global_evaluation_even_if_others_complete(self):
        api=self.api()
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            self.synthetic_selection_fixture(root,{'status':'resolved','runs':[
                {'run_id':'done','artifact_id':'done'},{'run_id':'pending','artifact_id':'pending'}]})
            model=root/'runs/done/model.bin';model.parent.mkdir(parents=True);model.write_bytes(b'fixture model')
            atomic_write_json(model.parent/'status.json',{'status':'complete','reload_verified':True,
                'files':{'model.bin':hash_file(model)}},root)
            with self.assertRaisesRegex(PermissionError,'pending|checkpoint|incomplete'):
                api.freeze_stage('evaluation',root,{})
            self.assertFalse((root/'evaluation.freeze.json').exists())

    def test_selection_byte_change_is_not_ignored_by_later_stage(self):
        api=self.api()
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);self.synthetic_selection_fixture(root,{'status':'resolved','runs':[]})
            api.verify_freeze('selection',root)
            (root/'run_manifest.json').write_text('{"changed":true}',encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'hash|changed'):
                api.verify_freeze('selection',root)

    def test_complete_global_lock_rejects_a_later_weight_change(self):
        api=self.api()
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);runs=[{'run_id':'done','artifact_id':'done'}]
            self.synthetic_selection_fixture(root,{'status':'resolved','runs':runs,'unresolved_nodes':[]})
            model=root/'runs/done/model.bin';model.parent.mkdir(parents=True);model.write_bytes(b'fixture weights')
            atomic_write_json(model.parent/'status.json',{'status':'complete','reload_verified':True,
                'files':{'model.bin':hash_file(model)}},root)
            atomic_write_json(root/'thresholds.json',{'design_hash':'design','runs':{'done':{'operative':None}}},root)
            atomic_write_json(root/'contrasts.json',{'design_hash':'design','items':[{'contrast_id':'fixture_primary'}]},root)
            api.freeze_stage('evaluation',root,{})
            api.verify_freeze('evaluation',root)
            model.write_bytes(b'changed weights')
            with self.assertRaisesRegex(ValueError,'hash|changed'):
                api.verify_freeze('evaluation',root)

    def test_selection_cannot_claim_ready_without_verified_human_pilot_flag(self):
        from research_v3.storage import read_json
        api=self.api()
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);self.synthetic_selection_fixture(root,{'status':'resolved','runs':[]})
            path=root/'selection.lock.json';lock=read_json(path)
            lock['actual_human_pilots_verified']=False
            lock['lock_hash']=canonical_hash({k:v for k,v in lock.items() if k!='lock_hash'})
            atomic_write_json(path,lock,root)
            with self.assertRaisesRegex(PermissionError,'human|pilot'):
                api.verify_freeze('selection',root)

    def test_duplicate_planned_run_is_rejected_before_global_freeze(self):
        api=self.api()
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);runs=[{'run_id':'done','artifact_id':'done'}]*2
            self.synthetic_selection_fixture(root,{'status':'resolved','runs':runs,'unresolved_nodes':[]})
            model=root/'runs/done/model.bin';model.parent.mkdir(parents=True);model.write_bytes(b'fixture weights')
            atomic_write_json(model.parent/'status.json',{'status':'complete','reload_verified':True,
                'files':{'model.bin':hash_file(model)}},root)
            atomic_write_json(root/'thresholds.json',{'design_hash':'design','runs':{'done':{'operative':None}}},root)
            atomic_write_json(root/'contrasts.json',{'design_hash':'design','items':[{'contrast_id':'fixture_primary'}]},root)
            with self.assertRaisesRegex(ValueError,'[Dd]uplicate'):
                api.freeze_stage('evaluation',root,{})

    def test_resolved_manifest_must_include_auxiliary_mechanism_runs(self):
        api=self.api()
        self.assertTrue(hasattr(api,'assert_resolved_coverage'),'global resolved coverage check is missing')
        design={'physical_runs':[{'run_id':'main'},{'run_id':'M2_required'}]}
        with self.assertRaisesRegex(PermissionError,'coverage|incomplete|M2'):
            api.assert_resolved_coverage(design,{'status':'resolved','runs':[{'run_id':'main'}]})

    def test_evaluation_binds_declared_model_request_for_diagnostic_checkpoints(self):
        api=self.api()
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);declared={'run_id':'done','model_id':'distilbert','arm':'random_frozen'}
            runs=[{'run_id':'done','artifact_id':'done',
                   'expected_model_request_hash':canonical_hash(declared)}]
            self.synthetic_selection_fixture(root,{'status':'resolved','runs':runs,'unresolved_nodes':[]})
            model=root/'runs/done/model.bin';model.parent.mkdir(parents=True);model.write_bytes(b'fixture weights')
            status={'status':'complete','reload_verified':True,'run':dict(declared,arm='pretrained_frozen'),
                    'request_hash':'additional-runtime-identity','files':{'model.bin':hash_file(model)}}
            atomic_write_json(model.parent/'status.json',status,root)
            atomic_write_json(root/'thresholds.json',{'design_hash':'design','runs':{'done':{'operative':0}}},root)
            atomic_write_json(root/'contrasts.json',{'design_hash':'design','items':[{'contrast_id':'fixture_primary'}]},root)
            with self.assertRaisesRegex(ValueError,'request identity'):
                api.freeze_stage('evaluation',root,{})
            status['run']=declared
            atomic_write_json(model.parent/'status.json',status,root)
            api.freeze_stage('evaluation',root,{})
            api.verify_freeze('evaluation',root)

    def test_selection_verifies_all_six_candidates_and_the_declared_validation_winner(self):
        api=self.api()
        self.assertTrue(hasattr(api,'verify_anchor_choice'),'anchor evidence verification is missing')
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);grid=[{'C':c} for c in [0.01,0.1,1,10,100,1000]]
            node={'node_id':'select:fixture:tfidf_svm','model_id':'tfidf_svm',
                  'anchor_condition_id':'condition','training_seed':None,'metric':'macro_f1','candidates':grid}
            records=[]
            for i,hp in enumerate(grid):
                output=root/'runs'/f'run{i}';output.mkdir(parents=True)
                model=output/'model.bin';model.write_bytes(f'fixture model {i}'.encode())
                status={'status':'complete','reload_verified':True,'run_id':f'run{i}',
                    'run':{'run_id':f'run{i}','condition_id':'condition','model_id':'tfidf_svm',
                           'training_seed':None,'strict_no_val':False,'hparams':hp},
                    'validation':{'macro_f1':i/10},'files':{'model.bin':hash_file(model)}}
                atomic_write_json(output/'status.json',status,root)
                records.append({'candidate_index':i,'hparams':hp,'run_id':f'run{i}',
                    'validation':status['validation'],'run_status_sha256':hash_file(output/'status.json')})
            choice={'status':'complete','hparams':grid[2],'selected_run_id':'run2',
                'evidence_sha256':records[2]['run_status_sha256'],'validation':records[2]['validation'],
                'candidates':records}
            with self.assertRaisesRegex(ValueError,'winner|validation|selection'):
                api.verify_anchor_choice(root,node,choice,{})
            choice.update(hparams=grid[5],selected_run_id='run5',
                evidence_sha256=records[5]['run_status_sha256'],validation=records[5]['validation'])
            api.verify_anchor_choice(root,node,choice,{})
            choice['candidates']=records[:5]
            with self.assertRaisesRegex(PermissionError,'six|candidate|complete'):
                api.verify_anchor_choice(root,node,choice,{})


if __name__=='__main__':unittest.main()
