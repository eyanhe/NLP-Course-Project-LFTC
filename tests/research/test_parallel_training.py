"""Training may precede human pilots only under the user's explicit amendment."""
from copy import deepcopy
import importlib
import inspect
from pathlib import Path
import tempfile
import unittest

from research_v3.config import canonical_hash
from research_v3.fixed_policy import default_candidate
from research_v3.storage import atomic_write_json
from tests.research.test_registry import RegistryTests


class ParallelTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):RegistryTests.setUpClass()

    def api(self):
        try:return importlib.import_module('research_v3.parallel_training')
        except ModuleNotFoundError:self.fail('authorized training-before-pilots dispatch is missing')

    def test_training_manifest_keeps_real_human_pilots_pending_and_cannot_pass_global_freeze(self):
        from research_v3.registry import build_run_manifest
        self.assertIn('training_only',inspect.signature(build_run_manifest).parameters,
                      'training-only registry is missing')
        selections=deepcopy(RegistryTests.selections);selections['selection_policy']='fixed_defaults'
        for node in RegistryTests.design['selection_nodes']:
            selections['choices'][node['node_id']]['hparams']=default_candidate(node['model_id'],RegistryTests.p,node['candidates'])
        for pilot in selections['discovery_pilots'].values():pilot['status']='not_run'
        kwargs={'asset_hash':'assets','training_only':True}
        manifest=build_run_manifest(RegistryTests.design,selections,RegistryTests.metadata,RegistryTests.p,
            'core','diagnostics',{ds:f'V-{ds}' for ds in RegistryTests.metadata},**kwargs)
        self.assertEqual(manifest['status'],'training_resolved')
        self.assertEqual(len(manifest['runs']),len(RegistryTests.design['physical_runs']))
        self.assertFalse(manifest['actual_human_pilots_verified'])
        self.assertEqual(len(manifest['pending_discovery_pilots']),3)
        self.assertTrue(all(p['status']=='not_run' for p in selections['discovery_pilots'].values()))
        from research_v3.freeze import assert_resolved_coverage
        with self.assertRaises(PermissionError):assert_resolved_coverage(RegistryTests.design,manifest)
        kwargs['training_only']=False
        with self.assertRaises(PermissionError):
            build_run_manifest(RegistryTests.design,selections,RegistryTests.metadata,RegistryTests.p,
                'core','diagnostics',{ds:f'V-{ds}' for ds in RegistryTests.metadata},**kwargs)

    def test_actual_execution_requires_explicit_user_order_amendment(self):
        api=self.api()
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);path=root/'authorization.json'
            atomic_write_json(path,{'training_enabled':True,'test_predictions_enabled':False},root)
            with self.assertRaisesRegex(PermissionError,'user|amendment'):
                api.authorize_training({},root,path)
            self.assertFalse((root/'runs').exists())

    def test_scheduling_retains_every_run_and_places_small_primary_conditions_first(self):
        api=self.api()
        design={'conditions':[{'condition_id':'large','experiment_id':'E1','metadata':{'rows':1000000}},
                              {'condition_id':'small','experiment_id':'E1','metadata':{'rows':1000}}]}
        rows=[{'run_id':'large','condition_id':'large','dataset_id':'imdb','model_id':'tfidf_lr'},
              {'run_id':'small','condition_id':'small','dataset_id':'banking77','model_id':'tfidf_lr'}]
        ordered=api.order_entries(rows,design)
        self.assertEqual([r['run_id'] for r in ordered],['small','large'])
        self.assertEqual({r['run_id'] for r in ordered},{r['run_id'] for r in rows})

    def test_dispatch_runs_a_real_tiny_cpu_fit_and_records_complete_progress(self):
        from unittest.mock import patch
        from tests.research.test_batch import BatchTests
        from research_v3.storage import read_json
        api=self.api()
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);entry,prepared,assets,_=BatchTests().fixture(root)
            entry.update(data_seed=2026,training_seed=None)
            design=read_json(root/'matrix/design_manifest.json')
            design['conditions'][0].update(experiment_id='E1')
            design['conditions'][0]['metadata']['rows']=2
            atomic_write_json(root/'matrix/design_manifest.json',design,root)
            atomic_write_json(root/'data/prepared_manifest.json',prepared,root)
            atomic_write_json(root/'assets/assets.lock.json',assets,root)
            manifest={'status':'training_resolved','training_only':True,'runs':[entry]}
            # These mocks isolate already separately-tested approval/lock boundaries.
            with patch.object(api,'prepare',return_value=manifest),patch.object(api,'verify_training_lock',return_value=({'lock_hash':'synthetic-lock'},manifest)):
                result=api.run({},root,root/'authorization.json')
            self.assertEqual(result['completed_fits'],1)
            self.assertFalse(result['official_test_predictions_generated'])
            self.assertEqual(read_json(root/'runs'/entry['run_id']/'status.json')['status'],'complete')


if __name__=='__main__':unittest.main()
