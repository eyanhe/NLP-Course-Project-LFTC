"""Pure registries use synthetic fixture evidence; never resolve the real study."""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import importlib
import json
from pathlib import Path
import unittest

from research_v3.config import canonical_hash
from research_v3.matrix import build_design_matrix
from research_v3.types import Example, PreparedDataset, RunSpec

ROOT = Path(__file__).resolve().parents[2]
EMPTY = hashlib.sha256(b'').hexdigest()


class RegistryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        p = json.loads((ROOT/'configs/research-v3.json').read_text(encoding='utf-8-sig'))
        p['experiments']['E1'].update(anchor=6, requested_sizes=[6,12,'full',999])
        p['experiments']['E2']['shots_per_class'] = [1,5,50]
        p['experiments']['E3']['budget_cap'] = 12
        p['experiments']['E4']['rows'] = 6
        p['experiments']['E5']['sizes'] = [6,'min_100000_or_full']
        p['experiments']['E6']['mechanism_rules']['M2']['sizes'] = [6,12]
        p['experiments']['E6']['mechanism_rules']['M5']['sizes'] = [12,'full']
        data = {}
        metadata = {}
        for ds in p['scope']['single_text_datasets']:
            rows = tuple(Example(ds,f'{ds}/{c}/{i}','text',(c,),'train')
                         for c in range(2) for i in range(30))
            task = 'multilabel' if ds == 'go_emotions' else 'single_label'
            data[ds] = PreparedDataset(ds,task,('a','b'),{'train':rows},f'pool-{ds}')
            metadata[ds] = {'dataset_id':ds,'task':task,'label_names':['a','b'],
                            'manifest_hash':f'pool-{ds}','inputs':{'input_hash':f'input-{ds}'},
                            'partitions':{'select':{'ap_mask':[0,1]},'test':{'ap_mask':[1]}}}
        cls.p, cls.metadata = p, metadata
        cls.design = build_design_matrix(data,p)
        cls.selections = {
            'choices': {n['node_id']:{'status':'complete','hparams':n['candidates'][0],
                                      'evidence_sha256':'a'*64}
                        for n in cls.design['selection_nodes']},
            'discovery_pilots':{n['pilot_id']:{'status':'complete','source_partition':'discovery',
                                              'n_planned':400,'results_sha256':'b'*64}
                                for n in cls.design['discovery_pilot_nodes']}}

    def api(self):
        try:
            return importlib.import_module('research_v3.registry')
        except ModuleNotFoundError:
            self.fail('resolved run and contrast registry is not implemented')

    def manifest(self, selections=None, design=None, **kwargs):
        return self.api().build_run_manifest(design or self.design,selections or self.selections,
            self.metadata,self.p,'core-hash','diagnostics-hash',
            {ds:f'select-{ds}' for ds in self.metadata},asset_hash='assets',**kwargs)

    def contrasts(self, manifest=None, plans=None):
        return self.api().build_contrast_manifest(self.design,manifest or self.manifest(),
            self.metadata,self.p,plans or self.selections['discovery_pilots'])

    def test_unresolved_anchor_or_any_real_pilot_blocks_resolution(self):
        for kind, key in [('choices',next(iter(self.selections['choices']))),
                          *[('discovery_pilots',p) for p in self.selections['discovery_pilots']]]:
            selections = deepcopy(self.selections)
            selections[kind][key]['status'] = 'not_run'
            with self.assertRaisesRegex((ValueError,PermissionError),'[Uu]nresolved|pending'):
                self.manifest(selections)
        selections = deepcopy(self.selections)
        selections['choices'][next(iter(selections['choices']))]['hparams'] = {'C':999}
        with self.assertRaisesRegex(ValueError,'candidate'):
            self.manifest(selections)

    def test_executed_identity_matches_anchor_spec_and_keeps_exact_conditions(self):
        manifest = self.manifest()
        self.assertEqual(manifest['status'],'resolved')
        self.assertEqual({r['planned_run_id'] for r in manifest['runs']},
                         {r['run_id'] for r in self.design['physical_runs']})
        self.assertEqual(len(manifest['runs']),len(self.design['physical_runs']))
        conditions = {c['condition_id']:c for c in self.design['conditions']}
        for entry in manifest['runs']:
            run = entry['run']; empty = {**run,'run_id':''}
            self.assertEqual(entry['artifact_id'],canonical_hash(empty))
            self.assertEqual(entry['run_id'],run['run_id'])
            self.assertEqual(entry['expected_model_request_hash'],canonical_hash(run))
            self.assertEqual(run['hparams']['role_ids_hash']['train'],
                             conditions[run['condition_id']]['training_set_hash'])
        anchor = next(n for n in self.design['selection_nodes'] if n['model_id']=='distilbert')
        entry = next(r for r in manifest['runs'] if r['run']['condition_id']==anchor['anchor_condition_id']
                     and r['model_id']=='distilbert' and r['training_seed']==42)
        from research_v3.matrix import training_policy
        hp = {**training_policy('distilbert',self.p),**anchor['candidates'][0],
              'role_ids_hash':{'train':conditions[anchor['anchor_condition_id']]['training_set_hash'],
                               'select':f"select-{anchor['dataset_id']}",'discovery':EMPTY},
              'target_task':'single_label',**{k:self.p['runtime'][k] for k in ['device','micro_batch','cpu_threads']},
              'protocol_hash':canonical_hash(self.p),'label_count':2}
        expected = RunSpec('',anchor['anchor_condition_id'],'distilbert',42,hp,
                           f"input-{anchor['dataset_id']}",'assets','core-hash',False)
        self.assertEqual(entry['artifact_id'],canonical_hash(asdict(expected)))

    def test_strict_runs_have_no_selection_binding_and_fixed_epochs(self):
        for entry in self.manifest()['runs']:
            run = entry['run']; hp = run['hparams']
            if run['strict_no_val']:
                self.assertEqual(hp['role_ids_hash']['select'],EMPTY)
                self.assertEqual(hp['role_ids_hash']['discovery'],EMPTY)
                self.assertNotIn('select_ap_mask',hp)
                if run['model_id'] in {'bilstm','distilbert'}:
                    self.assertEqual(hp['epochs'],10 if run['model_id']=='bilstm' else 5)
                self.assertNotIn('selection_node',json.dumps(hp))
            if run['model_id'] in {'tfidf_lr','tfidf_svm','w2v_svm','fixed_tfidf_svm'}:
                self.assertIsNone(run['training_seed'])

    def test_mechanism_engines_and_multilabel_selection_mask_are_explicit(self):
        manifest = self.manifest()
        planned = {r['run_id']:r for r in self.design['physical_runs']}
        for entry in manifest['runs']:
            model = entry['model_id']; source = planned[entry['planned_run_id']]
            diagnostics = model in {'m2_distilbert','fixed_tfidf_svm','frozen_generic_distilbert_head'} or (
                model=='bilstm' and bool(source['variant']))
            self.assertEqual(entry['engine'],'diagnostics' if diagnostics else 'core')
            self.assertEqual(entry['run']['code_hash'],'diagnostics-hash' if diagnostics else 'core-hash')
            if entry['dataset_id']=='go_emotions' and not entry['run']['strict_no_val']:
                self.assertEqual(entry['run']['hparams']['select_ap_mask'],[0,1])
        self.assertEqual(manifest['unique_fit_count'],len({r['artifact_id'] for r in manifest['runs']}))
        self.assertEqual(sum(len(r['semantic_aliases']) for r in manifest['runs']),len(self.design['logical_runs']))

    def test_resolution_is_pure_and_missing_selection_hash_is_rejected(self):
        before = deepcopy((self.design,self.selections,self.metadata,self.p))
        self.manifest()
        self.assertEqual(before,(self.design,self.selections,self.metadata,self.p))
        with self.assertRaisesRegex(ValueError,'selection.*hash|select.*hash'):
            self.api().build_run_manifest(self.design,self.selections,self.metadata,self.p,
                'core','diag',{},asset_hash='assets')

    def test_all_families_auxiliaries_and_infeasible_conditions_are_preserved(self):
        registry = self.contrasts()
        self.assertEqual(registry['items'],registry['contrasts'])
        self.assertEqual({r['family'] for r in registry['contrasts']},
                         {'F1','F2','F3','F4','F5','F6','AUX_M2','AUX_M3','AUX_M5'})
        self.assertEqual(len([r for r in registry['contrasts'] if r['family']=='F6']),3)
        self.assertEqual(registry['design_hash'],self.design['design_hash'])
        self.assertEqual(len(registry['inactive_conditions']),len(self.design['infeasible_conditions']))
        self.assertTrue(all(r['status']=='inactive_infeasible' for r in registry['inactive_conditions']))
        self.assertTrue(all(r['active'] is False for r in registry['inactive_conditions']))

    def test_draws_average_per_run_metrics_and_full_is_one_seed_zero(self):
        registry = self.contrasts()
        estimates = {e['estimate_id']:e for e in registry['estimates']}
        full = next(e for e in estimates.values() if e.get('experiment_id')=='E1' and e.get('level')=='full')
        self.assertEqual([d['data_seed'] for d in full['draws']],[0])
        for estimate in estimates.values():
            datasets = set()
            for draw in estimate['draws']:
                for arm in ['left_runs','right_runs']:
                    self.assertEqual(len(draw[arm]),len(set(draw[arm])))
                    seeds = [registry['runs'][r]['training_seed'] for r in draw[arm]]
                    self.assertEqual(len(seeds),len(set(seeds)))
                    datasets.update(registry['runs'][r]['dataset_id'] for r in draw[arm])
            self.assertEqual(len(datasets),1)
        self.assertEqual(registry['datasets']['go_emotions']['ap_mask'],[1])
        self.assertTrue(all(e['metric']=='macro_ap' for e in estimates.values()
                            if e.get('dataset_id')=='go_emotions'))

    def test_f3_heads_share_rho_one_baseline_and_duplicate_binary_orders_are_aliases(self):
        registry = self.contrasts()
        e3 = [e for e in registry['estimates'] if e.get('experiment_id')=='E3' and e['dataset_id']=='yelp']
        self.assertEqual(len([e for e in e3 if e['level']==1]),1)
        skew = [e for e in e3 if e['level']==10]
        self.assertEqual(len(skew),len({tuple(x['context']['head_order']) for x in self.design['logical_runs']
                                       if x['experiment_id']=='E3' and x['dataset_id']=='yelp' and x['context']['ratio']==10}))
        self.assertTrue(any(len(e['head_seed_aliases'])>1 for e in skew))
        baseline = next(e['estimate_id'] for e in e3 if e['level']==1)
        interactions = [c for c in registry['contrasts'] if c['family']=='F3' and c['dataset_id']=='yelp'
                        and c['kind']=='interaction' and c.get('levels')==[1,10]]
        self.assertEqual(len(interactions),len(skew))
        self.assertTrue(all(baseline in [p['estimate_id'] for p in c['components']] for c in interactions))

    def test_f6_views_share_checkpoint_but_distinct_evaluations_and_base_groups(self):
        registry = self.contrasts()
        for estimate in [e for e in registry['estimates'] if e.get('experiment_id')=='E6']:
            self.assertEqual(estimate['metric'],'correctness')
            draw = estimate['draws'][0]
            left,right = (registry['runs'][draw[k][0]] for k in ['left_runs','right_runs'])
            self.assertEqual(left['checkpoint_run_id'],right['checkpoint_run_id'])
            self.assertNotEqual(left['evaluation_id'],right['evaluation_id'])
            self.assertEqual(left['base_group_namespace'],right['base_group_namespace'])
            self.assertEqual(left['n_planned'],400)

    def test_auxiliary_factorials_include_interactions_and_optimizer_sensitivity(self):
        registry = self.contrasts()
        for family in ['AUX_M2','AUX_M3','AUX_M5']:
            self.assertTrue(any(c['family']==family and c['kind']=='interaction' for c in registry['contrasts']))
        self.assertTrue(any(c.get('effect')=='encoder_lr_sensitivity' for c in registry['contrasts']))

    def test_contrasts_reject_incomplete_coverage_and_unresolved_pilot_plans(self):
        manifest = self.manifest(); manifest['runs'].pop()
        with self.assertRaisesRegex((ValueError,PermissionError),'coverage|incomplete'):
            self.contrasts(manifest)
        plans = deepcopy(self.selections['discovery_pilots'])
        plans[next(iter(plans))]['status']='not_run'
        with self.assertRaisesRegex((ValueError,PermissionError),'pilot|pending|unresolved'):
            self.contrasts(plans=plans)

    def test_supplied_selected_anchor_id_must_match_the_actual_resolved_request(self):
        selections = deepcopy(self.selections)
        node = self.design['selection_nodes'][0]
        selections['choices'][node['node_id']]['selected_run_id'] = 'wrong-checkpoint'
        with self.assertRaisesRegex(ValueError,'[Aa]nchor.*identity|selected.*checkpoint'):
            self.manifest(selections)

    def test_e4_zero_and_m5_baseline_reuse_the_exact_main_artifact(self):
        manifest = self.manifest()
        for entry in manifest['runs']:
            aliases = entry['semantic_aliases']
            for row in aliases:
                if row['experiment_id']=='E4' and row['context']['alpha']==0:
                    self.assertTrue(any(a['experiment_id']=='E1' for a in aliases))
                if row['experiment_id']=='M5' and not next(p['variant'] for p in self.design['physical_runs']
                                                          if p['run_id']==entry['planned_run_id']):
                    self.assertEqual(entry['engine'],'core')
                    self.assertTrue(any(a['experiment_id']=='E1' for a in aliases))

    def test_all_generated_families_are_consumable_by_the_statistics_engine(self):
        from research_v3.statistics import estimate_contrasts
        from research_v3.types import PredictionRow
        registry = self.contrasts()
        rows = []
        for run_id,meta in registry['runs'].items():
            multi = meta['dataset_id']=='go_emotions'
            for index in range(4):
                gold = (index%2,)
                prediction = gold
                rows.append(PredictionRow(run_id,f'sample-{index}',f'group-{index}',gold,prediction,
                                          (.8,.2) if index%2==0 else (.2,.8),'probability','fixture'))
        for family in {c['family'] for c in registry['contrasts']}:
            result = estimate_contrasts(tuple(rows),dict(registry,family=family))
            self.assertTrue(result)


if __name__=='__main__':
    unittest.main()
