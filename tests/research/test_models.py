"""Full target dimensions and constant-label fallbacks must survive small data."""
import importlib
import unittest
import numpy as np
from scipy.sparse import csr_matrix


class ModelTests(unittest.TestCase):
    def api(self):
        try:return importlib.import_module('research_v3.models')
        except ModuleNotFoundError:self.fail('research learner implementation is missing')

    def test_multilabel_constant_columns_keep_all_28_dimensions(self):
        for family in ['tfidf_lr','tfidf_svm']:
            with self.subTest(family=family):
                learner=self.api().build_model(family,'multilabel',28,{}, {'C':1})
                x=csr_matrix([[1,0],[0,1],[1,1],[0,0]])
                y=np.zeros((4,28),dtype=int);y[:,0]=[1,0,1,0];y[:,27]=1
                learner.fit(x,y)
                scores=learner.scores(x)
                self.assertEqual(scores.shape,(4,28))
                predicted=learner.predict(x)
                self.assertTrue((predicted[:,27]==1).all())
                self.assertTrue((predicted[:,1:27]==0).all())

    def test_unseen_single_classes_do_not_shrink_score_space(self):
        learner=self.api().build_model('tfidf_svm','single_label',4,{}, {'C':1})
        x=csr_matrix([[1,0],[0,1],[2,0],[0,2]])
        learner.fit(x,np.array([0,2,0,2]))
        self.assertEqual(learner.scores(x).shape,(4,4))
        self.assertEqual(learner.predict(x).tolist(),[0,2,0,2])

    def test_degenerate_features_use_training_prior(self):
        learner=self.api().build_model('tfidf_lr','single_label',3,{}, {'feature_degenerate':True})
        x=csr_matrix((4,1));learner.fit(x,np.array([1,1,1,2]))
        self.assertEqual(learner.predict(csr_matrix((2,1))).tolist(),[1,1])
        self.assertTrue(learner.feature_degenerate)
        self.assertEqual(learner.scores(csr_matrix((1,1))).tolist(),[[0,.75,.25]])

    def test_single_training_class_is_a_valid_full_space_prior(self):
        learner=self.api().build_model('w2v_svm','single_label',3,{}, {})
        learner.fit(np.ones((2,4)),np.array([2,2]))
        self.assertEqual(learner.predict(np.zeros((1,4))).tolist(),[2])
        self.assertEqual(learner.scores(np.zeros((1,4))).shape,(1,3))

    def test_bilstm_updates_do_not_mutate_the_shared_initial_feature_matrix(self):
        api=self.api()
        from types import SimpleNamespace
        import torch
        features=SimpleNamespace(embedding_matrix=np.zeros((4,8),dtype=np.float32))
        model=api.build_model('bilstm','single_label',2,{}, {},features)
        with torch.no_grad():model.embedding.weight.add_(1)
        self.assertEqual(float(features.embedding_matrix.sum()),0)


if __name__=='__main__':unittest.main()
