"""Sentence-pair boundary, leakage and MultiNLI evaluation checks."""
import unittest
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score,precision_score,recall_score
import hard_benchmark as hard
from pair_models import PairTfidf,PairBiLSTM
from pipeline import read_json

class PairBenchmarkChecks(unittest.TestCase):
    def test_three_class_metrics(self):
        y=np.array([0,1,2]);pred=np.zeros(3,dtype=int);m=hard.metrics(y,pred)
        self.assertEqual(len(m['confusion_matrix']),3)
        self.assertAlmostEqual(m['macro_precision'],1/9)
        self.assertAlmostEqual(m['macro_f1'],1/6)
        self.assertAlmostEqual(m['micro_f1'],m['accuracy'])
    def test_sparse_pair_direction_and_training_vocabulary(self):
        pairs=[{'premise':'Cats chase mice','hypothesis':'Animals chase mice'},{'premise':'Cats chase mice','hypothesis':'Animals chase mice'}]
        model=PairTfidf(ngram_range=(1,1)).fit(pairs)
        forward=model.transform(pairs[:1]);back=model.transform([{'premise':pairs[0]['hypothesis'],'hypothesis':pairs[0]['premise']}])
        self.assertGreater((forward!=back).nnz,0)
        vocabulary=dict(model.vectorizer_.vocabulary_)
        model.transform([{'premise':'zzzneverseenintraining','hypothesis':'zzzneverseenintraining'}])
        self.assertEqual(vocabulary,model.vectorizer_.vocabulary_)
    def test_shared_pair_encoder_ignores_padding(self):
        model=PairBiLSTM(np.random.default_rng(1).normal(size=(12,8)).astype(np.float32),hidden=4).eval()
        with torch.no_grad():
            a=model(torch.tensor([[2,3]]),torch.tensor([2]),torch.tensor([[4,5,6]]),torch.tensor([3]))
            b=model(torch.tensor([[2,3,0,0]]),torch.tensor([2]),torch.tensor([[4,5,6,0]]),torch.tensor([3]))
        torch.testing.assert_close(a,b)
    def test_transformer_pair_boundaries_and_budget(self):
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(hard.ROOT/'assets'/'distilbert',local_files_only=True)
        row=hard.encode_transformer([{'premise':'A cat sits. '*500,'hypothesis':'An animal sits.'}],tokenizer)[0]
        self.assertLessEqual(len(row['input_ids']),256)
        self.assertEqual(row['input_ids'].count(tokenizer.sep_token_id),2)
    @unittest.skipUnless((hard.DATA/'audit.json').exists(),'Prepare MultiNLI')
    def test_real_premise_group_separation_and_genres(self):
        frames={s:hard.load_split(s) for s in ['train','validation','matched','mismatched']}
        for a,fa in frames.items():
            self.assertEqual(set(fa.label),{0,1,2})
            self.assertEqual(len(set(fa.groupby('label').size())),1)
            for b,fb in frames.items():
                if a!=b:self.assertFalse(set(fa.premise_hash)&set(fb.premise_hash))
        self.assertTrue(set(frames['matched'].genre).issubset(set(frames['train'].genre)))
        self.assertFalse(set(frames['mismatched'].genre)&set(frames['train'].genre))
        self.assertEqual(set(frames['matched'].official_split),{'validation_matched'})
    @unittest.skipUnless((hard.OUT/'bootstrap.json').exists(),'Finish hard evaluation')
    def test_saved_pair_metrics_and_freeze(self):
        from datetime import datetime
        frozen=read_json(hard.OUT/'protocol.freeze.json');evaluation=read_json(hard.OUT/'evaluation.json')
        self.assertEqual(len(frozen['artifacts']),10)
        self.assertLess(datetime.fromisoformat(frozen['created_utc']),datetime.fromisoformat(evaluation['created_utc']))
        p=pd.read_csv(hard.OUT/'predictions.csv');self.assertEqual(len(p),19980)
        for r in evaluation['rows']:
            q=p[(p.split==r['target'].removeprefix('mnli_'))&(p.model==r['model'])]
            if r['seed'] is not None:q=q[q.seed==r['seed']]
            self.assertEqual(len(q),999)
            self.assertAlmostEqual(r['macro_f1'],f1_score(q.label,q.prediction,average='macro'))
            self.assertAlmostEqual(r['macro_precision'],precision_score(q.label,q.prediction,average='macro'))
            self.assertAlmostEqual(r['macro_recall'],recall_score(q.label,q.prediction,average='macro'))
    @unittest.skipUnless((hard.directory('distilbert',44)/'metadata.json').exists(),'Finish all hard models')
    def test_real_three_way_prediction_outputs(self):
        pair=[{'premise':'A woman reads in a library.','hypothesis':'Someone is reading indoors.'}]
        for name in hard.MODELS:
            model,tok=hard.load_model(name,42,'cpu');pred,score,kind=hard.predict(model,tok,pair,name)
            self.assertEqual(np.asarray(score).shape,(1,3))
            self.assertIn(int(pred[0]),[0,1,2])
            if kind=='probability':np.testing.assert_allclose(np.sum(score,axis=1),1,atol=1e-5)

if __name__=='__main__':unittest.main(verbosity=2)
