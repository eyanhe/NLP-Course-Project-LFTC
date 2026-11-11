"""The design graph must enumerate all planned work without inventing selections."""
import importlib
import json
from pathlib import Path
import tempfile
import unittest
from copy import deepcopy
from research_v3.types import Example,PreparedDataset

ROOT=Path(__file__).resolve().parents[2]


class MatrixTests(unittest.TestCase):
    def api(self):
        try:
            return importlib.import_module('research_v3.matrix')
        except ModuleNotFoundError:
            self.fail('the design matrix is not implemented')

    def fixtures(self):
        p=json.loads((ROOT/'configs/research-v3.json').read_text(encoding='utf-8-sig'))
        p['experiments']['E1']['requested_sizes']=[8,24,'full']
        p['experiments']['E1']['anchor']=24
        p['experiments']['E2']['shots_per_class']=[1,5]
        p['experiments']['E3']['budget_cap']=30
        p['experiments']['E4']['rows']=24
        p['experiments']['E5']['sizes']=[24,'min_100000_or_full']
        data={}
        for dataset in p['scope']['single_text_datasets']:
            rows=tuple(Example(dataset,f'{dataset}/train/{c}/{i}','',(c,), 'train',f'{dataset}-{c}-{i}')
                       for c in range(3) for i in range(30))
            task='multi_label' if dataset=='go_emotions' else 'single_label'
            data[dataset]=PreparedDataset(dataset,task,('a','b','c'),
                {'train':rows,'select':(),'discovery':(),'test':(),'quarantine':()},f'pool-{dataset}')
        return data,p

    def test_experiment_coverage_and_deterministic_models_have_no_fake_seeds(self):
        data,p=self.fixtures();design=self.api().build_design_matrix(data,p)
        by_exp={name:set() for name in ['E1','E2','E3','E4','E5']}
        for node in design['logical_runs']:
            if node['experiment_id'] in by_exp:
                by_exp[node['experiment_id']].add(node['dataset_id'])
        self.assertEqual([len(by_exp[e]) for e in by_exp],[7,6,3,1,4])
        for run in design['physical_runs']:
            if run['model_id'] in ['tfidf_lr','tfidf_svm','w2v_svm']:
                self.assertIsNone(run['training_seed'])
        self.assertEqual(design['status'],'design_locked_selection_pending')
        self.assertEqual(len(design['selection_nodes']),39)
        self.assertTrue(all(len(node['candidates'])==6 for node in design['selection_nodes']))

    def test_aliases_reuse_actual_runs_and_mechanisms_remain_in_the_graph(self):
        data,p=self.fixtures();design=self.api().build_design_matrix(data,p)
        self.assertLess(len(design['physical_runs']),len(design['logical_runs']))
        self.assertEqual(set(design['mechanism_nodes']),{'M1','M2','M3','M4','M5'})
        self.assertEqual(len(design['discovery_pilot_nodes']),3)
        self.assertTrue(all(node['status']=='not_run' for node in design['discovery_pilot_nodes']))
        self.assertTrue(all(node['status']!='complete' for node in design['physical_runs']))
        self.assertTrue(any(node['experiment_id']=='M3' for node in design['logical_runs']))

    def test_missing_selection_cannot_resolve_or_create_a_selection_lock(self):
        data,p=self.fixtures();design=self.api().build_design_matrix(data,p)
        with self.assertRaisesRegex(ValueError,'selection|unresolved|pilot'):
            self.api().resolve_runs(design,{}, {},'code')

    def test_design_lock_detects_tampered_manifest(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            (root/'manifest.json').write_text('{"status":"design"}',encoding='utf-8')
            self.api().write_design_lock(root,{'manifest':'manifest.json'},{'protocol':'p'})
            self.api().verify_design_lock(root)
            (root/'manifest.json').write_text('{"status":"changed"}',encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'hash|changed'):
                self.api().verify_design_lock(root)

    def test_changed_training_budget_changes_design_and_resolved_identity(self):
        data,p=self.fixtures();api=self.api()
        first=api.build_design_matrix(data,p)
        changed=deepcopy(p);changed['training']['bilstm']['max_epochs']=20
        second=api.build_design_matrix(data,changed)
        self.assertNotEqual(first['design_hash'],second['design_hash'])
        self.assertNotEqual({r['run_id'] for r in first['physical_runs']},
                            {r['run_id'] for r in second['physical_runs']})
        def fixture_selections(design):
            selection={node['node_id']:{'hparams':node['candidates'][0],'evidence_sha256':'a'*64}
                       for node in design['selection_nodes']}
            selection['discovery_pilots']={n['pilot_id']:{'status':'complete'} for n in design['discovery_pilot_nodes']}
            selection['all_primary_contrasts']=['fixture-contrast']
            return selection
        one=api.resolve_runs(first,fixture_selections(first),{'asset_hash':'fixture'},'code')
        two=api.resolve_runs(second,fixture_selections(second),{'asset_hash':'fixture'},'code')
        self.assertNotEqual({r.run_id for r in one},{r.run_id for r in two})
        ordinary=next(r for r in two if r.model_id=='bilstm' and not r.strict_no_val)
        self.assertEqual(ordinary.hparams['max_epochs'],20)
        self.assertEqual(ordinary.hparams['effective_batch'],64)


if __name__=='__main__':
    unittest.main()
