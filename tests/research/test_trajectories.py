"""Failure content follows explicit paired paths; it does not establish causes."""
import copy
import importlib
import unittest
from dataclasses import replace

from research_v3.types import PredictionRow


def paths(seed=42):
    return {'learning': {'model_id':'distilbert','data_seed':2026,'training_seed':seed,
        'steps':[{'run_id':f'r{i}','condition_id':f'n{i}','condition_name':f'{i}k'} for i in (1,2,3)]}}


def single(run, sample, predicted, gold=0, group=None):
    return PredictionRow(run,sample,group or 'g-'+sample,(gold,),(predicted,),
                         (.8,.2) if predicted==0 else (.2,.8),'probability','visible-'+sample)


class TrajectoryTests(unittest.TestCase):
    def api(self):
        try: return importlib.import_module('research_v3.trajectories')
        except ModuleNotFoundError: self.fail('Failure trajectory implementation is missing')

    def test_additional_data_corrects_errors_introduces_errors_and_preserves_instability(self):
        # All gold0; each literal sequence independently defines the expected pattern.
        sequences={'old_error':[1,1,1], 'always_right':[0,0,0], 'fixed':[1,0,0],
                   'regressed':[0,0,1], 'flips':[1,0,1]}
        p=tuple(single('r'+str(i+1),sample,pred) for sample,seq in sequences.items() for i,pred in enumerate(seq))
        out=self.api().trace_failures(p,paths(),'single_label',2)
        path=out['paths']['learning'];samples={s['sample_id']:s for s in path['samples']}
        want={'old_error':'persistent_error','always_right':'persistent_correct',
              'fixed':'corrected','regressed':'new_error','flips':'unstable'}
        self.assertEqual({k:s['classification'] for k,s in samples.items()},want)
        self.assertEqual(path['counts'],{k:1 for k in want.values()})
        fixed=samples['fixed']
        self.assertEqual(fixed['correctness'],[False,True,True])
        self.assertEqual(fixed['change_count'],1)
        self.assertEqual([t['transition'] for t in fixed['transitions']],['corrected','remained_correct'])
        self.assertEqual(fixed['runs'][0]['scores'],[.2,.8])
        self.assertEqual(fixed['runs'][0]['prediction'],[1])
        self.assertEqual(fixed['runs'][0]['false_negative_label_ids'],[0])
        self.assertEqual(fixed['runs'][0]['false_positive_label_ids'],[1])
        self.assertEqual(fixed['group_id'],'g-fixed')
        self.assertEqual(fixed['labels'],[0])
        self.assertEqual(fixed['runs'][0]['input_hash'],'visible-fixed')
        self.assertEqual(fixed['cause_status'],'unresolved')
        self.assertEqual(path['cause_status'],'unresolved')

    def test_sparse_multilabel_empty_decisions_retain_per_label_false_negatives_and_positives(self):
        spec=paths();spec['learning']['steps']=spec['learning']['steps'][:2]
        p=(PredictionRow('r1','x','base',(0,2),(1,2),(.1,.8,.9),'probability','h1'),
           PredictionRow('r2','x','base',(0,2),(),(.1,.2,.1),'probability','h2'))
        out=self.api().trace_failures(p,spec,'multilabel',3)
        sample=out['paths']['learning']['samples'][0]
        self.assertEqual(sample['classification'],'persistent_error')
        self.assertEqual(sample['runs'][0]['false_negative_label_ids'],[0])
        self.assertEqual(sample['runs'][0]['false_positive_label_ids'],[1])
        self.assertEqual(sample['runs'][1]['false_negative_label_ids'],[0,2])
        self.assertEqual(sample['runs'][1]['false_positive_label_ids'],[])
        labels={label['label_id']:label for label in sample['per_label']}
        self.assertEqual(labels[0]['false_negative'],[True,True])
        self.assertEqual(labels[0]['fn_classification'],'persistent_error')
        self.assertEqual(labels[1]['false_positive'],[True,False])
        self.assertEqual(labels[1]['fp_classification'],'corrected')
        self.assertEqual(labels[2]['false_negative'],[False,True])
        self.assertEqual(labels[2]['fn_classification'],'new_error')

    def test_independent_seed_paths_share_full_checkpoint_without_ensembling_or_extra_fits(self):
        spec=paths()
        spec['learning']['steps']=[{'run_id':'s42','condition_id':'small'},
                                    {'run_id':'full','condition_id':'full','data_seed':0}]
        spec['draw2027']=copy.deepcopy(spec['learning'])
        spec['draw2027']['data_seed']=2027
        spec['draw2027']['steps'][0]['run_id']='s27'
        spec['seed43']=copy.deepcopy(spec['learning'])
        spec['seed43']['training_seed']=43
        spec['seed43']['steps'][0]['run_id']='s43'
        spec['seed43']['steps'][1]['run_id']='full43'
        p=(single('s42','x',1),single('s27','x',0),single('s43','x',0),
           single('full','x',0),single('full43','x',1))
        out=self.api().trace_failures(p,spec,'single_label',2)
        self.assertEqual(out['paths']['learning']['samples'][0]['classification'],'corrected')
        self.assertEqual(out['paths']['draw2027']['samples'][0]['classification'],'persistent_correct')
        self.assertEqual(out['paths']['seed43']['samples'][0]['classification'],'new_error')
        self.assertEqual(out['paths']['learning']['training_seed'],42)
        self.assertEqual(out['paths']['seed43']['training_seed'],43)
        self.assertEqual(out['unique_run_ids'],['full','full43','s27','s42','s43'])
        self.assertEqual(out['unique_run_count'],5)
        self.assertEqual(out['paths']['learning']['steps'][1]['data_seed'],0)

    def test_declared_condition_order_drives_transitions_not_lexical_run_order(self):
        spec=paths();spec['learning']['steps']=[{'run_id':'z','condition_id':'first'},
                                               {'run_id':'a','condition_id':'last'}]
        out=self.api().trace_failures((single('a','x',0),single('z','x',1)),spec,'single_label',2)
        sample=out['paths']['learning']['samples'][0]
        self.assertEqual([r['run_id'] for r in sample['runs']],['z','a'])
        self.assertEqual(sample['classification'],'corrected')
        self.assertEqual(sample['transitions'][0]['from_condition_id'],'first')
        self.assertEqual(sample['transitions'][0]['to_condition_id'],'last')

    def test_missing_duplicate_or_mismatched_base_rows_cannot_form_a_path(self):
        spec=paths();p=tuple(single('r'+str(i),'x',0) for i in (1,2,3))
        variants={'missing_run':p[:-1], 'duplicate':p+(p[0],),
                  'gold':p[:-1]+(replace(p[-1],labels=(1,)),),
                  'sample':p[:-1]+(replace(p[-1],sample_id='other'),),
                  'group':p[:-1]+(replace(p[-1],group_id='other'),)}
        for name,invalid in variants.items():
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError,'[Mm]issing|[Dd]uplicate|paired|identity'):
                    self.api().trace_failures(invalid,spec,'single_label',2)

    def test_invalid_sparse_labels_and_scores_are_rejected(self):
        spec=paths();p=tuple(PredictionRow('r'+str(i),'x','g',(0,2),(0,),(.8,.1,.1),'probability','h') for i in (1,2,3))
        for field,bad in [('labels',(0,0)),('labels',(2,0)),('labels',()),('prediction',(3,)),
                          ('scores',(.8,)),('scores',(.8,float('nan'),.1))]:
            with self.subTest(field=field,bad=bad):
                with self.assertRaisesRegex(ValueError,'label|indices|score'):
                    self.api().trace_failures((replace(p[0],**{field:bad}),)+p[1:],spec,'multilabel',3)

    def test_paths_reject_aliased_steps_and_mixed_model_or_seed_metadata(self):
        p=tuple(single('r'+str(i),'x',0) for i in (1,2,3))
        variants=[]
        for field,value in [('run_id','r1'),('condition_id','n1'),('model_id','svm'),
                            ('training_seed',43),('data_seed',2027)]:
            spec=paths();spec['learning']['steps'][1][field]=value;variants.append(spec)
        for spec in variants:
            with self.subTest(spec=spec):
                with self.assertRaisesRegex(ValueError,'path|[Dd]uplicate|seed|model'):
                    self.api().trace_failures(p,spec,'single_label',2)

    def test_failure_patterns_never_become_mechanism_claims(self):
        spec=paths();spec['learning']['phenomenon']='negation'
        spec['learning']['cause']='pretraining'
        out=self.api().trace_failures(tuple(single('r'+str(i),'x',1) for i in (1,2,3)),spec,'single_label',2)
        sample=out['paths']['learning']['samples'][0]
        self.assertEqual(sample['cause_status'],'unresolved')
        self.assertEqual(sample['cause_evidence'],[])
        self.assertNotIn('cause',sample)


if __name__=='__main__': unittest.main()
