from copy import deepcopy
import importlib
from pathlib import Path
import tempfile
import unittest

from research_v3.storage import atomic_write_json,read_json
from research_v3 import convergence as driver
from tests.research import test_scoped_evaluation as fixtures
from unittest.mock import patch


class BudgetedFinishTests(unittest.TestCase):
    def api(self):
        try:return importlib.import_module('research_v3.budgeted_finish')
        except ModuleNotFoundError:self.fail('budget-complete selection amendment is missing')

    def test_budget_cap_is_eligible_without_falsifying_convergence(self):
        api=self.api()
        records=[{'status':'complete','reload_verified':True,'convergence_verified':reason=='patience',
            'stop_reason':reason,'epochs_completed':30,'selected_epoch':20,
            'run':{'model_id':'distilbert','hparams':{'lr':lr,'max_epochs':30}},
            'validation':{'macro_ap':value}} for lr,reason,value in
            [(2e-5,'patience',.3),(5e-5,'max_epochs',.5),(1e-4,'patience',.4)]]
        before=deepcopy(records)
        winner=api.choose_completed(records,3,'macro_ap')
        self.assertEqual(winner['run']['hparams']['lr'],5e-5)
        self.assertFalse(winner['convergence_verified'])
        self.assertEqual(records,before)
        records[1]['status']='incomplete';records[1]['stop_reason']='deadline'
        with self.assertRaises(PermissionError):api.choose_completed(records,3,'macro_ap')

    def test_capped_candidates_and_replica_can_proceed_to_frozen_test(self):
        api=self.api()
        from research_v3.scoped_evaluation import freeze_scope
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);manifest=fixtures.ScopedEvaluationTests().fixture(root)
            group='banking77/bilstm'
            path=root/driver.REL/'selections'/'banking77--bilstm.json';path.unlink()
            for entry in driver._all_entries(root,manifest):
                if entry['group']!=group:continue
                p=root/'runs'/entry['run_id']/'status.json';status=read_json(p)
                status.update(stop_reason='max_epochs',convergence_verified=False,
                    epochs_completed=entry['run']['hparams']['max_epochs'],selected_epoch=30)
                atomic_write_json(p,status,root)
            api.update_completed_selections(root,manifest)
            selection=read_json(path)
            self.assertEqual(selection['selection_policy'],'completed_budget_V_best')
            for entry in selection['confirmation_entries']:
                p=root/'runs'/entry['run_id']/'status.json';status=read_json(p)
                status.update(stop_reason='max_epochs',convergence_verified=False,
                    epochs_completed=entry['run']['hparams']['max_epochs'],selected_epoch=30)
                atomic_write_json(p,status,root)
            scope=freeze_scope(root)
            self.assertEqual(len(scope['models']),18)
            self.assertTrue(any(m.get('stop_reason')=='max_epochs' for m in scope['models']))

    def test_finalizer_only_lists_unclaimed_seed_repeats(self):
        api=self.api()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);manifest=fixtures.ScopedEvaluationTests().fixture(root)
            repeats=[e for e in driver._all_entries(root,manifest) if e['phase']=='confirmation']
            for entry in [repeats[0],repeats[1],manifest['candidates'][0]]:
                (root/'runs'/entry['run_id']/'status.json').unlink()
            claim=root/driver.REL/'claims'/(repeats[1]['run_id']+'.json');claim.parent.mkdir(parents=True)
            claim.write_text('{}',encoding='utf-8')
            self.assertEqual([e['run_id'] for e in api.pending_repeats(root,manifest)],[repeats[0]['run_id']])

    def test_asset_changes_are_rejected_before_any_repeat(self):
        api=self.api()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);fixtures.ScopedEvaluationTests().fixture(root)
            (root/'fixture-model-assets/tokenizer.json').write_bytes(b'changed asset')
            with patch.dict('os.environ',{'SLURM_JOB_ID':'fixture'}),patch.object(driver,'fit_entry') as fitter:
                with self.assertRaises(ValueError):api.finish(root)
                fitter.assert_not_called()


if __name__=='__main__':unittest.main()
