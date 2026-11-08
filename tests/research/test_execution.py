"""Explicit execution authorization cannot bypass research-stage locks."""
import importlib
from pathlib import Path
import tempfile
import unittest
from research_v3.matrix import write_design_lock
from research_v3.storage import atomic_write_json


class ExecutionTests(unittest.TestCase):
    def api(self):
        try:return importlib.import_module('research_v3.execution')
        except ModuleNotFoundError:self.fail('training phase authorization is missing')

    def fixture(self,root):
        (root/'manifest.json').write_text('{}')
        lock=write_design_lock(root,{'manifest':'manifest.json'},{'protocol_hash':'protocol','design_hash':'design'})
        path=root/'authorization.json'
        atomic_write_json(path,{'user_instruction':'开始按计划完成训练','protocol_hash':'protocol',
            'design_lock_hash':lock['lock_hash'],'allowed_training_phases':['engineering','anchors','main'],
            'training_enabled':True,'test_predictions_enabled':False},root)
        return path

    def test_engineering_allowed_but_main_requires_actual_selection_lock(self):
        api=self.api()
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);authorization=self.fixture(root)
            p={'_protocol_hash':'protocol'}
            self.assertTrue(api.authorize_phase(p,root,authorization,'engineering')['training_enabled'])
            with self.assertRaisesRegex(PermissionError,'selection|pilot|锚点'):
                api.authorize_phase(p,root,authorization,'main')

    def test_changed_protocol_cannot_use_old_authorization(self):
        api=self.api()
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);authorization=self.fixture(root)
            with self.assertRaisesRegex(ValueError,'protocol|identity'):
                api.authorize_phase({'_protocol_hash':'changed'},root,authorization,'engineering')

    def test_complete_flag_without_15_real_smoke_artifacts_cannot_allow_anchors(self):
        api=self.api()
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);authorization=self.fixture(root)
            atomic_write_json(root/'training/engineering-smoke.json',{'complete':True,'cases':[]},root)
            with self.assertRaisesRegex(PermissionError,'E0|engineering|artifact|case'):
                api.authorize_phase({'_protocol_hash':'protocol'},root,authorization,'anchors')

    def test_same_process_cannot_take_two_exclusive_run_leases(self):
        api=self.api()
        with tempfile.TemporaryDirectory() as temp:
            output=Path(temp)/'run'
            with api.run_lease(output,'request'):
                with self.assertRaisesRegex(RuntimeError,'running|lease|exclusive'):
                    with api.run_lease(output,'request'):pass
            with api.run_lease(output,'request'):pass

    def test_unverified_complete_status_is_not_usable_for_prediction(self):
        self.api()
        from research_v3.train import score_saved
        with tempfile.TemporaryDirectory() as temp:
            output=Path(temp)
            atomic_write_json(output/'status.json',{'status':'complete','reload_verified':False,'files':{}},output)
            with self.assertRaisesRegex(ValueError,'verified|reload|complete'):
                score_saved(output,())


if __name__=='__main__':unittest.main()
