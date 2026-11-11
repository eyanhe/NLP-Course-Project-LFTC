"""Checks for multiclass correctness, sequence padding and the expanded GUI."""
import unittest
import re
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score,precision_score,recall_score
from extension import ROOT,OUT,LABELS,MODELS,SOURCES,metric_set,load_split,load_model,predict,directory
from models_extra import BiLSTM,words
from pipeline import read_json

class V2Checks(unittest.TestCase):
    def test_four_class_metrics_do_not_collapse_to_binary(self):
        y=np.array([0,1,2,3]);p=np.zeros(4,dtype=int);m=metric_set(y,p,'ag_news')
        self.assertEqual(len(m['confusion_matrix']),4)
        self.assertAlmostEqual(m['macro_precision'],.0625)
        self.assertAlmostEqual(m['macro_recall'],.25)
        self.assertAlmostEqual(m['macro_f1'],.1)
        self.assertAlmostEqual(m['micro_f1'],m['accuracy'])
        self.assertEqual(set(LABELS['ag_news']),set(m['classification_report'])-{'accuracy','macro avg','weighted avg'})

    def test_bilstm_padding_is_not_a_real_token(self):
        torch.manual_seed(1)
        model=BiLSTM(np.random.default_rng(1).normal(size=(10,8)).astype(np.float32),4,4).eval()
        with torch.no_grad():
            a=model(torch.tensor([[2,3,4]]),torch.tensor([3]))
            b=model(torch.tensor([[2,3,4,0,0]]),torch.tensor([3]))
        torch.testing.assert_close(a,b)

    def test_gui_has_no_chinese_text(self):
        self.assertIsNone(re.search(r'[\u4e00-\u9fff]',(ROOT/'app.py').read_text('utf-8')))

    @unittest.skipUnless((ROOT/'data-v2'/'audit.json').exists(),'Prepare news first')
    def test_news_partition_labels_and_balance(self):
        frames=[load_split('ag_news',s) for s in ['train','validation','test']]
        all_rows=pd.concat(frames)
        self.assertFalse(all_rows.text_hash.duplicated().any())
        for f in frames:
            self.assertEqual(set(f.label),{0,1,2,3})
            self.assertEqual(len(set(f.groupby('label').size())),1)
        self.assertEqual(set(frames[-1].official_split),{'test'})

    @unittest.skipUnless((OUT/'evaluation.json').exists(),'Complete V2 evaluation first')
    def test_recorded_metrics_reproduce_from_all_predictions(self):
        from datetime import datetime
        evaluation=read_json(OUT/'evaluation.json');frozen=read_json(OUT/'protocol.freeze.json')
        self.assertLess(datetime.fromisoformat(frozen['created_utc']),datetime.fromisoformat(evaluation['created_utc']))
        self.assertEqual(len(frozen['artifacts']),36)
        p=pd.read_csv(OUT/'predictions.csv')
        self.assertEqual(len(p),66000)
        for r in evaluation['rows']:
            subset=p[(p.source==r['source'])&(p.target==r['target'])&(p.model==r['model'])]
            if r['seed'] is not None:subset=subset[subset.seed==r['seed']]
            self.assertEqual(len(subset),1000)
            self.assertAlmostEqual(r['macro_precision'],precision_score(subset.label,subset.prediction,average='macro',zero_division=0))
            self.assertAlmostEqual(r['macro_recall'],recall_score(subset.label,subset.prediction,average='macro',zero_division=0))
            self.assertAlmostEqual(r['weighted_f1'],f1_score(subset.label,subset.prediction,average='weighted',zero_division=0))

    @unittest.skipUnless((ROOT/'models-v2'/'ag_news'/'distilbert_seed44'/'metadata.json').exists(),'Train all news models first')
    def test_news_predictions_have_four_class_outputs(self):
        texts=['The team won the final match.','The company reported quarterly earnings.']
        for name in MODELS:
            model,tok=load_model('ag_news',name,42,'cpu');pred,score,kind=predict(model,tok,texts,name)
            self.assertEqual(np.asarray(score).shape,(2,4))
            self.assertTrue(set(pred).issubset({0,1,2,3}))
            if kind=='probability':np.testing.assert_allclose(score.sum(axis=1),1,atol=1e-5)

if __name__=='__main__':unittest.main(verbosity=2)
