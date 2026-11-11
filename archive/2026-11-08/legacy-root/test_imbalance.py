"""Data isolation, prior-shift math and saved robustness result checks."""
import unittest
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score,recall_score
import imbalance as exp
from pipeline import read_json

class ImbalanceChecks(unittest.TestCase):
    def test_majority_dummy_and_simulated_prior(self):
        y=np.repeat(np.arange(4),250);pred=np.zeros(1000,dtype=int)
        m=exp.measures(y,pred,'severe')
        self.assertAlmostEqual(m['accuracy'],.25);self.assertAlmostEqual(m['macro_f1'],.1)
        self.assertEqual(m['minority_recall'],0);self.assertAlmostEqual(m['simulated_prior_weighted_accuracy'],.94)
        self.assertAlmostEqual(m['balanced_accuracy'],m['macro_recall'])
    @unittest.skipUnless((exp.DATA/'audit.json').exists(),'Prepare data')
    def test_fixed_total_nested_minority_and_isolation(self):
        frames={c:exp.load_train(c) for c in exp.CFG['conditions']}
        val=exp.BASE_LOAD('ag_news','validation');test=exp.BASE_LOAD('ag_news','test')
        forbidden=set(val.text_hash)|set(test.text_hash)
        original=exp.BASE_LOAD('ag_news','train')
        self.assertEqual(set(frames['balanced'].sample_id),set(original.sample_id))
        for c,f in frames.items():
            self.assertEqual(len(f),6000);self.assertFalse(f.text_hash.duplicated().any())
            self.assertFalse(set(f.text_hash)&forbidden)
            self.assertEqual(f.groupby('label').size().tolist(),exp.CFG['conditions'][c])
        for a,b in zip(list(frames)[:-1],list(frames)[1:]):
            for label in [1,2,3]:self.assertTrue(set(frames[b].loc[frames[b].label==label,'sample_id'])<=set(frames[a].loc[frames[a].label==label,'sample_id']))
            self.assertTrue(set(frames[a].loc[frames[a].label==0,'sample_id'])<=set(frames[b].loc[frames[b].label==0,'sample_id']))
    @unittest.skipUnless((exp.OUT/'evaluation.json').exists(),'Finish evaluation')
    def test_frozen_checkpoints_and_recomputed_metrics(self):
        frozen=read_json(exp.OUT/'protocol.freeze.json');result=read_json(exp.OUT/'evaluation.json');design=read_json(exp.OUT/'design.freeze.json')
        self.assertEqual(len(frozen['artifacts']),40);self.assertEqual(len(result['rows']),40)
        self.assertLess(datetime.fromisoformat(design['created_utc']),datetime.fromisoformat(frozen['created_utc']))
        self.assertLess(datetime.fromisoformat(frozen['created_utc']),datetime.fromisoformat(result['created_utc']))
        pred=pd.read_csv(exp.OUT/'predictions.csv');self.assertEqual(len(pred),40000)
        legacy=pd.read_csv(exp.ROOT/'results-v2'/'predictions.csv')
        for r in result['rows']:
            f=pred[(pred.condition==r['condition'])&(pred.model==r['model'])&(pred.weighted==r['weighted'])]
            f=f[f.seed==r['seed']] if r['seed'] is not None else f[f.seed.isna()]
            self.assertEqual(len(f),1000)
            self.assertAlmostEqual(r['macro_f1'],f1_score(f.label,f.prediction,average='macro'))
            self.assertAlmostEqual(r['minority_recall'],recall_score(f.label,f.prediction,labels=[1,2,3],average='macro'))
            if r['condition']=='balanced':
                old=legacy[(legacy.source=='ag_news')&(legacy.target=='ag_news')&(legacy.model==r['model'])]
                old=old[old.seed==r['seed']] if r['seed'] is not None else old[old.seed.isna()]
                old=old.set_index('sample_id').loc[f.sample_id]
                np.testing.assert_array_equal(f.prediction.to_numpy(),old.prediction.to_numpy())
        for a in frozen['artifacts']:self.assertEqual(exp.sha(exp.ROOT/a['path']),a['sha256'])
    @unittest.skipUnless((exp.directory('severe','distilbert',44,True)/'metadata.json').exists(),'Finish weighted models')
    def test_frozen_hyperparameters_and_weighted_loss(self):
        for condition,name,seed,weighted in exp.runs():
            if condition=='balanced':continue
            m=read_json(exp.directory(condition,name,seed,weighted)/'metadata.json');exp.check(m)
            base=read_json(exp.directory('balanced',name,seed)/'metadata.json')
            if name in ['bilstm','distilbert']:
                self.assertEqual(m['learning_rate'],base['learning_rate'])
                self.assertEqual(len(m['history']),3 if name=='distilbert' else 5)
                self.assertAlmostEqual(m['best_validation_macro_f1'],max(x['validation_macro_f1'] for x in m['history']))
                if weighted:np.testing.assert_allclose(m['class_weights'],6000/(4*np.array([5640,120,120,120])))
            else:self.assertEqual(m['selected'],base['selected'])

if __name__=='__main__':unittest.main(verbosity=2)
