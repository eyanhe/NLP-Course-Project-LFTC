"""Focused checks for data integrity, evaluation correctness and inference reload."""
import unittest
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from pipeline import ROOT, NAMES, clean_text, text_hash, fast_macro_f1, paired_bootstrap, load_split, load_model, predict_loaded, read_json

class ProtocolChecks(unittest.TestCase):
    def test_normalization_preserves_negation(self):
        self.assertEqual(clean_text('<br />This is NOT good! &amp; it is slow.'), 'This is NOT good! & it is slow.')
        self.assertEqual(text_hash('  Not   GOOD '), text_hash('not good'))

    def test_macro_f1_and_paired_resampling(self):
        y = np.array([0,0,0,1,1,1]); a = np.array([0,1,0,1,0,1]); b = y.copy()
        self.assertAlmostEqual(float(fast_macro_f1(y,a)), f1_score(y,a,average='macro'))
        result = paired_bootstrap(y,a,b,200)
        self.assertGreater(result['delta_macro_f1'],0)
        equal = paired_bootstrap(y,a,a,200)
        self.assertEqual(equal['ci95_low'],0); self.assertEqual(equal['ci95_high'],0)

    @unittest.skipUnless((ROOT / 'data' / 'audit.json').exists(), 'Prepare the real datasets first')
    def test_real_data_separation_and_labels(self):
        manifest = pd.read_csv(ROOT / 'data' / 'manifest.csv')
        self.assertFalse(manifest.text_hash.duplicated().any())
        self.assertFalse(manifest.sample_id.duplicated().any())
        self.assertEqual(set(manifest.label.unique()), {0,1})
        for name in NAMES:
            sets = {split:load_split(name,split) for split in ['train','validation','test']}
            for split, frame in sets.items():
                self.assertGreater(len(frame),0)
                self.assertTrue(frame.text.map(lambda t: isinstance(t,str) and bool(t.strip())).all())
                self.assertEqual(int((frame.label==0).sum()),int((frame.label==1).sum()))
                if split == 'test': self.assertEqual(set(frame.official_split),{'test'})

    @unittest.skipUnless((ROOT / 'models' / 'imdb' / 'linear_svm' / 'model.joblib').exists(), 'Train baselines first')
    def test_reload_and_vocabulary_isolation(self):
        model,tok=load_model('imdb','linear_svm')
        text=['A wonderful and moving film.','A dull and disappointing film.']
        a,_=predict_loaded(model,tok,text,'linear_svm')
        reloaded,_=load_model('imdb','linear_svm')
        b,_=predict_loaded(reloaded,tok,text,'linear_svm')
        np.testing.assert_array_equal(a,b)
        train=load_split('imdb','train')
        analyzer=model['tfidf'].build_analyzer()
        allowed=set()
        for text in train.text: allowed.update(analyzer(text))
        self.assertTrue(set(model['tfidf'].vocabulary_).issubset(allowed))
        before=dict(model['tfidf'].vocabulary_)
        model.predict(['zzzxnewunseenvalidationtokenxxx'])
        self.assertEqual(before,model['tfidf'].vocabulary_)

    @unittest.skipUnless((ROOT / 'results' / 'evaluation.json').exists(), 'Complete frozen evaluation first')
    def test_saved_predictions_reproduce_metrics_and_protocol_order(self):
        from datetime import datetime
        frozen=read_json(ROOT/'results'/'protocol.freeze.json')
        evaluation=read_json(ROOT/'results'/'evaluation.json')
        self.assertLess(datetime.fromisoformat(frozen['frozen_utc']),datetime.fromisoformat(evaluation['created_utc']))
        self.assertEqual(len(frozen['artifacts']),15)
        self.assertEqual(len(evaluation['rows']),45)
        predictions=pd.read_csv(ROOT/'results'/'predictions.csv')
        self.assertEqual(len(predictions),45000)
        for row in evaluation['rows']:
            subset=predictions[(predictions.source==row['source'])&(predictions.target==row['target'])&(predictions.model==row['model'])]
            if row['seed'] is not None: subset=subset[subset.seed==row['seed']]
            self.assertEqual(len(subset),row['test_n'])
            self.assertAlmostEqual(f1_score(subset.label,subset.prediction,average='macro'),row['macro_f1'])

if __name__=='__main__': unittest.main(verbosity=2)
