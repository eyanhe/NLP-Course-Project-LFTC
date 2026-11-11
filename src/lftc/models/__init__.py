"""Classifiers with full declared label spaces and explicit score semantics."""
import numpy as np


class TraditionalClassifier:
    def __init__(self, model_id, task, label_count, options=None):
        self.model_id, self.task, self.label_count = model_id, task, label_count
        self.options = dict(options or {})
        self.score_kind = 'probability' if model_id == 'tfidf_lr' else 'svm_margin'

    def _base(self):
        common = dict(C=float(self.options.get('C', 1)),
                      random_state=int(self.options.get('seed', 42)),
                      max_iter=int(self.options.get('max_iter', 3000)),
                      class_weight=self.options.get('class_weight'))
        if self.model_id == 'tfidf_lr':
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression(**common)
        from sklearn.svm import LinearSVC
        return LinearSVC(**common, dual='auto')

    def fit(self, x, y):
        y = np.asarray(y)
        degenerate = x.nnz == 0 if hasattr(x, 'nnz') else not np.any(x)
        if self.task == 'multilabel':
            self.priors = y.mean(axis=0)
            self.estimators = [None if degenerate or len(np.unique(y[:, i])) == 1
                               else self._base().fit(x, y[:, i]) for i in range(self.label_count)]
        else:
            self.priors = np.bincount(y, minlength=self.label_count)/len(y)
            self.estimator = None if degenerate or len(np.unique(y)) == 1 else self._base().fit(x, y)
        return self

    def scores(self, x):
        n = x.shape[0]
        if not n:
            return np.empty((0, self.label_count))
        constants = self.priors if self.score_kind == 'probability' else 2*self.priors-1
        result = np.broadcast_to(constants, (n, self.label_count)).copy()
        if self.task == 'multilabel':
            for i, model in enumerate(self.estimators):
                if model is not None:
                    result[:, i] = (model.predict_proba(x)[:, 1] if self.score_kind == 'probability'
                                    else model.decision_function(x))
        elif self.estimator is not None:
            model = self.estimator
            if self.score_kind == 'probability':
                result[:] = 0
                result[:, model.classes_.astype(int)] = model.predict_proba(x)
            else:
                result[:] = -1e9
                margin = model.decision_function(x)
                if margin.ndim == 1:
                    margin = np.column_stack((-margin, margin))
                result[:, model.classes_.astype(int)] = margin
        return result
