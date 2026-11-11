import importlib
import json
from pathlib import Path
import tempfile
import unittest


class SVMFirstWorkerTests(unittest.TestCase):
    def api(self):
        try:return importlib.import_module('research_v3.svm_first_worker')
        except ModuleNotFoundError:self.fail('SVM-first dispatcher is missing')

    def test_claims_only_svm_and_never_overwrites_live_claim(self):
        api=self.api()
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);entries=[{'run_id':'neural','model_id':'bilstm'},
                {'run_id':'svm1','model_id':'tfidf_svm'},{'run_id':'svm2','model_id':'tfidf_svm'}]
            original={'worker_id':'live','job_id':'1'}
            folder=root/'training/convergence-v1/claims';folder.mkdir(parents=True)
            (folder/'svm1.json').write_text(json.dumps(original),encoding='utf-8')
            result=api.claim_svm(root,entries,'new')
            self.assertEqual(result['run_id'],'svm2')
            self.assertEqual(json.loads((folder/'svm1.json').read_text()),original)
            self.assertFalse((folder/'neural.json').exists())
            self.assertIsNone(api.claim_svm(root,entries,'again'))


if __name__=='__main__':unittest.main()
