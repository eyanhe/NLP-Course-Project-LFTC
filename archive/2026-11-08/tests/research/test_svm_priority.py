import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


class PriorityTests(unittest.TestCase):
    def api(self):
        path=Path(__file__).resolve().parents[2]/'research-v3/training/convergence-v1/svm-priority.py'
        self.assertTrue(path.exists(),'SVM-first queue controller is missing')
        spec=importlib.util.spec_from_file_location('svm_priority',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        return module

    def test_reservation_preserves_running_claim_and_release_only_owns_its_files(self):
        api=self.api()
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);folder=root/'training/convergence-v1';folder.mkdir(parents=True)
            entries=[{'run_id':'active','model_id':'distilbert'},{'run_id':'later','model_id':'bilstm'},
                {'run_id':'svm','model_id':'tfidf_svm'}]
            claims=folder/'claims';claims.mkdir()
            original={'worker_id':'real-running-fit','job_id':'1'}
            (claims/'active.json').write_text(json.dumps(original),encoding='utf-8')
            owned=api.reserve(root,entries,'nonce','1')
            self.assertEqual(owned,['later'])
            self.assertEqual(json.loads((claims/'active.json').read_text()),original)
            self.assertFalse((claims/'svm.json').exists())
            api.release(root,owned,'wrong-nonce')
            self.assertTrue((claims/'later.json').exists())
            api.release(root,owned,'nonce')
            self.assertFalse((claims/'later.json').exists())
            self.assertTrue((claims/'active.json').exists())

    def test_interrupted_publication_can_release_its_complete_reservation(self):
        api=self.api()
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);owned=[];original=api.os.link
            def interrupted(source,target):original(source,target);raise InterruptedError('signal after atomic link')
            with patch.object(api.os,'link',side_effect=interrupted):
                with self.assertRaises(InterruptedError):api.reserve(root,[{'run_id':'later','model_id':'bilstm'}],'nonce','1',owned=owned)
            api.release(root,owned,'nonce')
            self.assertFalse((root/'training/convergence-v1/claims/later.json').exists())

    def test_bad_unrelated_claim_does_not_prevent_releasing_own_claim(self):
        api=self.api()
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);owned=api.reserve(root,[{'run_id':'later','model_id':'bilstm'}],'nonce','1')
            folder=root/'training/convergence-v1/claims';(folder/'unrelated.json').write_text('{',encoding='utf-8')
            api.release(root,['unrelated',*owned],'nonce')
            self.assertTrue((folder/'unrelated.json').exists())
            self.assertFalse((folder/'later.json').exists())

    def test_watchdog_releases_on_missing_or_stale_heartbeat(self):
        api=self.api()
        self.assertFalse(api.stale_heartbeat(50,0,None))
        self.assertTrue(api.stale_heartbeat(61,0,None))
        self.assertFalse(api.stale_heartbeat(100,0,90))
        self.assertTrue(api.stale_heartbeat(121,0,90))


if __name__=='__main__':unittest.main()
