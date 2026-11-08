"""The post-anchor entrypoint must not resolve missing real dependencies."""
import importlib
from pathlib import Path
import tempfile
import unittest

from research_v3.matrix import write_design_lock
from research_v3.storage import atomic_write_json


class WorkflowTests(unittest.TestCase):
    def api(self):
        try:return importlib.import_module('research_v3.workflow')
        except ModuleNotFoundError:self.fail('post-anchor workflow entrypoint is missing')

    def fixture(self,root):
        design={'design_hash':'design','protocol_hash':'protocol','selection_nodes':[],
                'discovery_pilot_nodes':[{'pilot_id':p} for p in ['p1','p2','p3']]}
        atomic_write_json(root/'matrix/design_manifest.json',design,root)
        write_design_lock(root,{'design':'matrix/design_manifest.json'},
                          {'protocol_hash':'protocol','design_hash':'design'})
        atomic_write_json(root/'selections.json',{'status':'anchors_complete_pilots_pending',
            'design_hash':'design','choices':{},'discovery_pilots':{}},root)

    def test_missing_genuine_pilots_cannot_write_resolved_manifests_or_locks(self):
        api=self.api()
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);self.fixture(root)
            with self.assertRaisesRegex(PermissionError,'pilot|human'):
                api.finalize_selection({'_protocol_hash':'protocol'},root)
            for name in ['run_manifest.json','contrasts.json','selection.lock.json']:
                self.assertFalse((root/name).exists())

    def test_protocol_mismatch_cannot_reuse_another_design(self):
        api=self.api()
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);self.fixture(root)
            with self.assertRaisesRegex(ValueError,'protocol|design'):
                api.finalize_selection({'_protocol_hash':'changed'},root)

    def test_entrypoint_rejects_training_before_an_execution_authorization(self):
        api=self.api()
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            with self.assertRaises((PermissionError,FileNotFoundError)):
                api.execute_workflow('train-batch',{'_protocol_hash':'protocol'},root,root/'missing.json')
            self.assertFalse((root/'runs').exists())


if __name__=='__main__':unittest.main()
