"""Hand-checked estimands and synchronized cluster uncertainty, without training."""
import copy
import importlib
import unittest
from dataclasses import replace

import numpy as np
from research_v3.types import PredictionRow


def row(run, i, y, prediction, scores=None, group=None):
    scores = tuple(scores if scores is not None else ([.9, .1] if prediction == 0 else [.1, .9]))
    return PredictionRow(run, str(i), group or str(i), (y,), (prediction,), scores, 'probability', 'fixture')


def manifest(left=('d',), right=('s',), metric='macro_f1'):
    return {
        'datasets': {'a': {'task': 'single_label', 'label_count': 2, 'partition': 'Q'}},
        'runs': {r: {'dataset_id': 'a', 'model_id': 'distilbert' if r in left else 'tfidf_svm',
                     'training_seed': 42+i if r in left else None, 'data_seed': 2026}
                 for i, r in enumerate(left+right)},
        'estimates': [{'estimate_id': 'delta', 'metric': metric,
                       'draws': [{'data_seed': 2026, 'left_runs': list(left), 'right_runs': list(right)}]}],
        'contrasts': [{'contrast_id': 'delta', 'family': 'F1', 'kind': 'difference',
                       'components': [{'estimate_id': 'delta', 'coefficient': 1}]}],
    }


class StatisticsTests(unittest.TestCase):
    def api(self):
        try:
            return importlib.import_module('research_v3.statistics')
        except ModuleNotFoundError:
            self.fail('statistical inference implementation is missing')

    def test_exact_mcnemar_is_a_base_group_sensitivity_with_known_binomial_probability(self):
        api=self.api()
        self.assertTrue(hasattr(api,'exact_mcnemar'),'exact McNemar sensitivity is missing')
        out=api.exact_mcnemar([1,1,1,1,1,0],[0,0,0,0,0,1],list('abcdef'))
        self.assertEqual(out['intervention_only_correct'],5)
        self.assertEqual(out['control_only_correct'],1)
        self.assertAlmostEqual(out['p_value'],.21875)
        self.assertEqual(out['role'],'sensitivity_only')
        self.assertEqual(api.exact_mcnemar([1,0],[1,0],['a','b'])['p_value'],1)
        with self.assertRaisesRegex(ValueError,'group|independent'):
            api.exact_mcnemar([1,0],[0,1],['same','same'])

    def test_estimator_averages_run_f1_not_ensemble_probabilities(self):
        # d1 perfect; d2 always class 0: macro F1 1/3. Their mean is 2/3.
        # Average probabilities are perfect, giving 1: the wrong estimand.
        m = manifest(('d1', 'd2'))
        p = (row('d1', 0, 0, 0, [.99,.01]), row('d1', 1, 1, 1, [.01,.99]),
             row('d2', 0, 0, 0, [.51,.49]), row('d2', 1, 1, 0, [.51,.49]),
             row('s', 0, 0, 0), row('s', 1, 1, 0))
        result = self.api().estimate_contrasts(p, m)
        self.assertAlmostEqual(result['contrasts']['delta']['estimate'], 1/3)
        self.assertEqual(set(result['estimates']['delta']['run_metrics']), {'d1','d2','s'})
        self.assertIsNone(result['estimates']['delta']['training_sd']['2026']['right'])
        self.assertAlmostEqual(result['estimates']['delta']['training_sd']['2026']['left'], np.sqrt(2)/3)

    def test_averages_actual_data_draws_and_single_full_pool_draw(self):
        m = manifest()
        m['runs'].update({'d0': {'dataset_id':'a','data_seed':0,'training_seed':42},
                          's0': {'dataset_id':'a','data_seed':0,'training_seed':None}})
        m['estimates'][0]['draws'].append({'data_seed':0,'left_runs':['d0'],'right_runs':['s0']})
        p = tuple(row(r,i,i,pred) for r,preds in {'d':[0,1],'s':[0,0],'d0':[0,1],'s0':[0,1]}.items()
                  for i,pred in enumerate(preds))
        out = self.api().estimate_contrasts(p,m)
        self.assertAlmostEqual(out['contrasts']['delta']['estimate'],1/3)
        self.assertAlmostEqual(out['contrasts']['delta']['seed_deltas']['2026'],2/3)
        self.assertEqual(out['contrasts']['delta']['seed_deltas']['0'],0.)

    def test_fixed_ap_mask_hand_calculation_and_bootstrap_degeneracy(self):
        m=manifest(metric='macro_ap')
        m['datasets']['a'].update(task='multilabel', label_count=2, ap_mask=[0])
        p=[]
        for r in ('d','s'):
            for i,(y,score) in enumerate(zip([1,0,1,0],[.8,.9,.7,.1])):
                p.append(PredictionRow(r,str(i),'pos' if y else 'neg',
                    (0,1) if y else (1,), (0,1) if score>=.5 else (1,),
                    (score,.8),'probability','fixture'))
        out=self.api().bootstrap_family(tuple(p),m,replicates=400,seed=7)
        self.assertAlmostEqual(out['estimates']['delta']['run_metrics']['d'],7/12)
        self.assertEqual(out['contrasts']['delta']['estimate'],0)
        self.assertEqual(out['critical'],0)
        diag=out['ap_degeneracy']['a']['0']
        self.assertGreater(diag['no_positive_rate'],.05)
        self.assertGreater(diag['no_negative_rate'],.05)
        self.assertTrue(diag['unstable'])
        self.assertEqual(out['ap_masks']['a'],[0])

    def test_ap_requires_a_frozen_valid_mask_never_infers_it_from_models(self):
        m=manifest(metric='macro_ap')
        m['datasets']['a'].update(task='multilabel',label_count=2,ap_mask=[1])
        p=tuple(PredictionRow(r,str(i),str(i),(0,1) if i else (1,),
                             (0,1) if i else (1,),(.1+.8*i,.8),'probability','fixture')
                for r in ('d','s') for i in (0,1))
        with self.assertRaisesRegex(ValueError,'mask'):
            self.api().estimate_contrasts(p,m)
        del m['datasets']['a']['ap_mask']
        with self.assertRaisesRegex(ValueError,'mask'):
            self.api().estimate_contrasts(p,m)

    def test_simultaneous_error_critical_centered_p_and_holm_are_hand_checked(self):
        out=self.api().compute_simultaneous_bands({'a':.2,'b':-.1},
                [[.2,-.1],[.3,-.3],[.5,-.1],[.1,-.5],[.2,.4]])
        self.assertAlmostEqual(out['critical'],.48)
        self.assertAlmostEqual(out['contrasts']['a']['simultaneous_lower'],-.28)
        self.assertAlmostEqual(out['contrasts']['b']['simultaneous_upper'],.38)
        self.assertAlmostEqual(out['contrasts']['a']['centered_p'],2/6)
        self.assertAlmostEqual(out['contrasts']['b']['centered_p'],4/6)
        self.assertAlmostEqual(out['contrasts']['a']['holm_p'],4/6)
        self.assertAlmostEqual(out['contrasts']['b']['holm_p'],4/6)

    def test_effect_labels_use_margin_and_all_data_seed_signs(self):
        fn=self.api().classify_effect
        for args,want in [((.03,.02,.04,(.02,.04)),'positive_advantage'),
                          ((-.03,-.04,-.02,(-.02,-.04)),'negative_advantage'),
                          ((.03,.02,.04,(.07,-.01)),'condition_sensitive'),
                          ((0,-.005,.005,(0,)),'practically_close'),
                          ((.02,-.001,.04,(.02,)),'uncertain'),
                          ((.02,.01,.04,(.02,)),'uncertain'),
                          ((0,-.01,.01,(0,)),'practically_close')]:
            with self.subTest(args=args): self.assertEqual(fn(*args),want)

    def test_shared_group_multiplicities_are_synchronized_across_datasets(self):
        m=manifest(metric='accuracy')
        m['datasets']['b']=copy.deepcopy(m['datasets']['a'])
        p=[]
        for ds in ('a','b'):
            for arm,preds in [('d',[0,0,1,0]),('s',[1,0,1,1])]:
                run=ds+arm
                m['runs'][run]={'dataset_id':ds,'data_seed':2026,'training_seed':42 if arm=='d' else None}
                p.extend(row(run,i,i%2,pred,group='base'+str(i//2)) for i,pred in enumerate(preds))
            m['estimates'].append({'estimate_id':ds,'metric':'accuracy','draws':[
                {'data_seed':2026,'left_runs':[ds+'d'],'right_runs':[ds+'s']}]})
        m['estimates']=m['estimates'][1:]
        m['contrasts']=[{'contrast_id':ds,'family':'F6','components':[{'estimate_id':ds,'coefficient':1}]}
                        for ds in ('a','b')]
        out=self.api().bootstrap_family(tuple(p),m,replicates=50,seed=11)
        self.assertEqual(out['group_ids'],['base0','base1'])
        self.assertEqual(out['bootstrap_estimates']['a'],out['bootstrap_estimates']['b'])
        self.assertEqual(set(out['bootstrap_estimates']['a']),{-.5,0.,.5})

    def test_original_and_edited_binary_correctness_share_base_group(self):
        m=manifest(metric='correctness')
        m['contrasts'][0]['family']='F6'
        m['runs']['d']['evaluation_id']='edited'
        m['runs']['s']['evaluation_id']='original'
        p=(row('d',0,0,0,group='base0'),row('d',1,1,0,group='base1'),
           row('s',0,0,0,group='base0'),row('s',1,1,1,group='base1'))
        out=self.api().bootstrap_family(p,m,replicates=40,seed=4)
        self.assertEqual(out['contrasts']['delta']['estimate'],-.5)
        self.assertEqual(set(out['bootstrap_estimates']['delta']),{-1.,-.5,0.})

    def test_identical_predictions_have_exact_zero_bands(self):
        m=manifest()
        p=tuple(row(r,i,i%2,0) for r in ('d','s') for i in range(8))
        out=self.api().bootstrap_family(p,m,replicates=25,seed=4)
        self.assertEqual(out['critical'],0)
        self.assertEqual(out['contrasts']['delta']['simultaneous_lower'],0)
        self.assertEqual(out['contrasts']['delta']['simultaneous_upper'],0)
        self.assertEqual(out['contrasts']['delta']['classification'],'practically_close')

    def test_tail_groups_distinguish_permutation_ties_and_uniform_natural(self):
        mod=self.api()
        self.assertEqual(mod.tail_group(head_order=[3,0,4,2,1,5]),
                         {'tail_definition':'current_head_permutation','label_ids':[1,5]})
        self.assertEqual(mod.tail_group(head_order=[1,0])['label_ids'],[0])
        self.assertEqual(mod.tail_group(frequencies=[1,2,2,10,12,15])['label_ids'],[0,1,2])
        self.assertIsNone(mod.tail_group(frequencies=[4,4,4])['label_ids'])

    def test_missing_run_duplicate_draw_and_fake_svm_repeats_are_rejected(self):
        p=tuple(row(r,i,i,i) for r in ('d','s') for i in (0,1))
        m=manifest()
        with self.assertRaisesRegex(ValueError,'missing|Missing'):
            self.api().estimate_contrasts(p[:-1],m)
        m['estimates'][0]['draws']*=2
        with self.assertRaisesRegex(ValueError,'draw|seed'):
            self.api().estimate_contrasts(p,m)
        m=manifest()
        m['estimates'][0]['draws'][0]['right_runs']=['s','s']
        with self.assertRaisesRegex(ValueError,'duplicate|Duplicate'):
            self.api().estimate_contrasts(p,m)

    def test_reversal_requires_opposite_advantages_and_interaction_band(self):
        fn=self.api().classify_reversal
        self.assertTrue(fn('positive_advantage','negative_advantage',.02,.05))
        self.assertFalse(fn('positive_advantage','negative_advantage',-.02,.05))
        self.assertFalse(fn('uncertain','negative_advantage',.02,.05))

    def test_interactions_are_linear_contrasts_and_multiple_families_require_selection(self):
        m=manifest()
        m['estimates'].append({'estimate_id':'opposite','metric':'macro_f1','draws':[
            {'data_seed':2026,'left_runs':['s'],'right_runs':['d']}]})
        m['contrasts'].append({'contrast_id':'interaction','family':'F1','kind':'interaction','components':[
            {'estimate_id':'delta','coefficient':1},{'estimate_id':'opposite','coefficient':-1}]})
        p=tuple(row(r,i,i,pred) for r,preds in {'d':[0,1],'s':[0,0]}.items() for i,pred in enumerate(preds))
        out=self.api().bootstrap_family(p,m,replicates=15)
        self.assertAlmostEqual(out['contrasts']['interaction']['estimate'],4/3)
        m['contrasts'][1]['family']='aux'
        with self.assertRaisesRegex(ValueError,'family'):
            self.api().bootstrap_family(p,m,replicates=5)
        m['family']='aux'
        self.assertEqual(set(self.api().bootstrap_family(p,m,replicates=5)['contrasts']),{'interaction'})

    def test_ap_instability_combines_disjoint_positive_and_negative_degeneracy(self):
        # Seed0's100 draws contain3 all-positive and3 all-negative resamples.
        # Their union exceeds5%, although each individual rate is below5%.
        m=manifest(metric='macro_ap')
        m['datasets']['a'].update(task='multilabel',label_count=2,ap_mask=[0])
        p=tuple(PredictionRow(r,str(i),str(i),(0,1) if i<3 else (1,),
                             (0,1) if i<3 else (1,),
                             (.8 if i<3 else .2,.8),'probability','fixture')
                for r in ('d','s') for i in range(6))
        out=self.api().bootstrap_family(p,m,replicates=100,seed=0)
        diag=out['ap_degeneracy']['a']['0']
        self.assertEqual(diag['no_positive_rate'],.03)
        self.assertEqual(diag['no_negative_rate'],.03)
        self.assertTrue(diag['unstable'])
        self.assertEqual(diag['degenerate_rate'],.06)

    def test_logical_aliases_of_same_checkpoint_cannot_create_training_replicates(self):
        m=manifest(('d1','d2'))
        for r in ('d1','d2'): m['runs'][r]['checkpoint_run_id']='same_physical_fit'
        p=tuple(row(r,i,i,i) for r in ('d1','d2','s') for i in (0,1))
        with self.assertRaisesRegex(ValueError,'physical|alias'):
            self.api().estimate_contrasts(p,m)

    def test_cross_task_metric_totals_are_rejected_in_components(self):
        m=manifest(metric='accuracy')
        m['datasets']['b']=copy.deepcopy(m['datasets']['a'])
        for r in ('bd','bs'): m['runs'][r]={'dataset_id':'b','data_seed':2026,'training_seed':None}
        m['estimates'].append({'estimate_id':'b','metric':'accuracy','draws':[
            {'data_seed':2026,'left_runs':['bd'],'right_runs':['bs']}]})
        m['contrasts'][0]['components'].append({'estimate_id':'b','coefficient':1})
        p=tuple(row(r,i,i,i) for r in ('d','s','bd','bs') for i in (0,1))
        with self.assertRaisesRegex(ValueError,'dataset|task'):
            self.api().estimate_contrasts(p,m)

    def test_global_union_uses_shared_and_dataset_unique_groups_as_weights(self):
        m=manifest(metric='accuracy');m['estimates']=[];m['contrasts']=[]
        m['datasets']['b']=copy.deepcopy(m['datasets']['a'])
        p=[]
        for ds in ('a','b'):
            for arm,preds in [('d',[0,0]),('s',[1,0])]:
                run=ds+arm
                m['runs'][run]={'dataset_id':ds,'data_seed':2026,'training_seed':None}
                p.extend(row(run,i,i,pred,group='common' if i==0 else 'only_'+ds)
                         for i,pred in enumerate(preds))
            m['estimates'].append({'estimate_id':ds,'metric':'accuracy','draws':[
                {'data_seed':2026,'left_runs':[ds+'d'],'right_runs':[ds+'s']}]})
            m['contrasts'].append({'contrast_id':ds,'family':'F6','components':[{'estimate_id':ds,'coefficient':1}]})
        # Seed1: [1,1,2], [2,0,0], [2,2,0], [0,2,1], [0,2,0].
        out=self.api().bootstrap_family(tuple(p),m,replicates=5,seed=1)
        self.assertEqual(out['group_ids'],['common','only_a','only_b'])
        np.testing.assert_allclose(out['bootstrap_estimates']['a'],[0,1,1,.5,1])
        np.testing.assert_allclose(out['bootstrap_estimates']['b'],[0,2/3,1/3,.5,2/3])

    def test_full_pool_interaction_broadcasts_fixed_draw_without_fake_repeats(self):
        m=manifest(metric='accuracy')
        m['runs'].update({'d0':{'dataset_id':'a','data_seed':0,'training_seed':42},
                          's0':{'dataset_id':'a','data_seed':0,'training_seed':None}})
        m['estimates'].append({'estimate_id':'full','metric':'accuracy','draws':[
            {'data_seed':0,'left_runs':['d0'],'right_runs':['s0']}]})
        m['contrasts'][0]['components'].append({'estimate_id':'full','coefficient':-1})
        p=tuple(row(r,i,i,pred) for r,preds in {'d':[0,1],'s':[0,0],'d0':[0,1],'s0':[1,0]}.items()
                for i,pred in enumerate(preds))
        out=self.api().estimate_contrasts(p,m)
        self.assertEqual(out['contrasts']['delta']['estimate'],-.5)
        self.assertEqual(out['contrasts']['delta']['seed_deltas'],{'2026':-.5})
        self.assertEqual(len(out['estimates']['full']['seed_deltas']),1)
        self.assertEqual(out['contrasts']['delta']['seed_sensitivity_status'],'aligned_with_fixed_full_pool')

    def test_empty_dataset_replicate_is_retained_and_blocks_inference(self):
        m=manifest(metric='accuracy');m['estimates']=[];m['contrasts']=[]
        m['datasets']['b']=copy.deepcopy(m['datasets']['a'])
        p=[]
        for ds in ('a','b'):
            for arm,pred in [('d',0),('s',1)]:
                run=ds+arm;m['runs'][run]={'dataset_id':ds,'data_seed':2026,'training_seed':None}
                p.append(row(run,0,0,pred,group=ds))
            m['estimates'].append({'estimate_id':ds,'metric':'accuracy','draws':[
                {'data_seed':2026,'left_runs':[ds+'d'],'right_runs':[ds+'s']}]})
            m['contrasts'].append({'contrast_id':ds,'family':'F6','components':[{'estimate_id':ds,'coefficient':1}]})
        out=self.api().bootstrap_family(tuple(p),m,replicates=5,seed=0)
        self.assertEqual(out['status'],'undefined_bootstrap')
        self.assertTrue(out['invalid_replicates'])
        self.assertIsNone(out['critical'])
        self.assertIn(None,out['bootstrap_estimates']['a'])
        self.assertEqual(len(out['bootstrap_estimates']['a']),5)

    def test_canonical_sparse_multilabel_variable_cardinality_and_empty_prediction(self):
        m=manifest()
        m['datasets']['a'].update(task='multilabel',label_count=3)
        gold=[(0,2),(1,),(0,),(2,)]
        predictions={'d':[(0,2),(),(0,1),(2,)],'s':[(0,),(1,),(),(2,)]}
        p=tuple(PredictionRow(run,str(i),str(i),gold[i],pred,
                             tuple(.9 if c in pred else .1 for c in range(3)),
                             'probability','fixture')
                for run,preds in predictions.items() for i,pred in enumerate(preds))
        try:
            out=self.api().bootstrap_family(p,m,replicates=20,seed=4)
        except ValueError as exc:
            self.fail(f'Canonical sparse PredictionRow records must be accepted: {exc}')
        self.assertAlmostEqual(out['estimates']['delta']['run_metrics']['d'],2/3)
        self.assertAlmostEqual(out['estimates']['delta']['run_metrics']['s'],7/9)
        self.assertAlmostEqual(out['contrasts']['delta']['estimate'],-1/9)

    def test_sparse_multilabel_rejects_duplicate_unsorted_out_of_range_and_empty_gold(self):
        m=manifest();m['datasets']['a'].update(task='multilabel',label_count=3)
        p=tuple(PredictionRow(r,str(i),str(i),(i,),(i,),(.8,.1,.1),'probability','fixture')
                for r in ('d','s') for i in (0,1))
        for field,bad in [('labels',(0,0)),('prediction',(1,1)),('labels',(3,)),
                          ('prediction',(-1,)),('labels',(2,0)),('labels',())]:
            with self.subTest(field=field,bad=bad):
                malformed=(replace(p[0],**{field:bad}),)+p[1:]
                with self.assertRaisesRegex(ValueError,'label|indices|canonical'):
                    self.api().estimate_contrasts(malformed,m)

    def test_cross_view_estimate_rejects_missing_or_changed_base_identity(self):
        m=manifest(metric='correctness')
        m['runs']['d']['evaluation_id']='edit'
        m['runs']['s']['evaluation_id']='control'
        p=tuple(row(r,i,i,i,group='base'+str(i)) for r in ('d','s') for i in (0,1))
        variants={'missing':p[:-1], 'sample':p[:-1]+(replace(p[-1],sample_id='other'),),
                  'gold':p[:-1]+(replace(p[-1],labels=(0,)),),
                  'group':p[:-1]+(replace(p[-1],group_id='other'),)}
        for name,malformed in variants.items():
            with self.subTest(identity=name):
                with self.assertRaisesRegex(ValueError,'paired|base|identity'):
                    self.api().estimate_contrasts(malformed,m)

    def test_unrelated_estimates_allow_full_q_and_smaller_paired_subset(self):
        m=manifest(metric='accuracy')
        for r in ('de','se'):
            m['runs'][r]={'dataset_id':'a','data_seed':2026,'training_seed':None,'evaluation_id':r}
        m['estimates'].append({'estimate_id':'edit','metric':'correctness','draws':[
            {'data_seed':2026,'left_runs':['de'],'right_runs':['se']}]})
        m['contrasts'].append({'contrast_id':'edit','family':'F1','components':[{'estimate_id':'edit','coefficient':1}]})
        p=tuple(row(r,i,i,i) for r in ('d','s') for i in (0,1))+(row('de',0,0,0),row('se',0,0,1))
        out=self.api().estimate_contrasts(p,m)
        self.assertEqual(out['contrasts']['delta']['estimate'],0)
        self.assertEqual(out['contrasts']['edit']['estimate'],1)


if __name__=='__main__': unittest.main()
