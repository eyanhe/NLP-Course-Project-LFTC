"""Catch validation vocabulary leakage and degenerate train-only features."""
import importlib
import unittest
from research_v3.types import Example,VisibleInput


def visible(identifier,text):
    return VisibleInput(identifier,text,0,len(text),len(text.split()),len(text.split()),identifier)


class FeatureTests(unittest.TestCase):
    def api(self):
        try:return importlib.import_module('research_v3.features')
        except ModuleNotFoundError:self.fail('train-only feature implementation is missing')

    def test_validation_only_words_do_not_enter_tfidf_vocabulary(self):
        features=self.api().fit_features('tfidf_lr',(visible('a','ordinary training words'),visible('b','more training words')),{})
        matrix=features.transform((visible('v','validationonlyxyz'),))
        self.assertEqual(matrix.nnz,0)
        self.assertNotIn('validationonlyxyz',features.vocabulary)

    def test_empty_training_text_has_explicit_degenerate_feature_state(self):
        features=self.api().fit_features('tfidf_svm',(visible('a',''),visible('b',' ')),{})
        self.assertTrue(features.degenerate)
        self.assertEqual(features.transform((visible('v','new unseen vocabulary'),)).shape,(1,1))

    def test_raw_example_is_rejected_instead_of_reading_invisible_tail(self):
        with self.assertRaisesRegex(TypeError,'VisibleInput'):
            self.api().fit_features('tfidf_lr',(Example('x','a','hidden raw tail',(0,),'train'),),{})

    def test_word_vectors_and_unknown_inputs_use_only_selected_training_text(self):
        api=self.api()
        features=api.fit_features('w2v_svm',(visible('a','warm ordinary words'),visible('b','cold ordinary words')),
                                  {'vector_size':8,'epochs':2,'vocabulary_limit':20})
        unknown=features.transform((visible('v','validationonlyxyz'),))
        self.assertEqual(unknown.shape,(1,8))
        self.assertEqual(float(abs(unknown).sum()),0)
        self.assertNotIn('validationonlyxyz',features.vocabulary)

    def test_bilstm_encoding_does_not_apply_a_second_word_limit(self):
        api=self.api()
        features=api.fit_features('bilstm',(visible('a','ordinary words repeated'),),{'vector_size':8,'epochs':1})
        rows=features.sequences((visible('v','ordinary '*400),))
        self.assertEqual(len(rows[0]),400)


if __name__=='__main__':unittest.main()
