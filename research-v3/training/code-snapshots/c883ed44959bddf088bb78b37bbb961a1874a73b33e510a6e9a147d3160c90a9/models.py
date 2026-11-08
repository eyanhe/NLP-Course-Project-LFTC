"""Five learning families with stable full-label score contracts."""
from __future__ import annotations

import re
from typing import Any
import warnings
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC


class TraditionalClassifier:
    def __init__(self,model_id: str,task: str,label_count: int,hp: dict[str,Any]):
        self.model_id=model_id;self.task=task;self.label_count=label_count;self.hp=dict(hp)
        self.score_kind='probability' if model_id=='tfidf_lr' else 'svm_margin'
        self.feature_degenerate=bool(hp.get('feature_degenerate',False))

    def _base(self):
        if self.model_id=='tfidf_lr':
            return LogisticRegression(C=float(self.hp.get('C',1)),max_iter=int(self.hp.get('max_iter',1000)),
                                      random_state=2026,tol=float(self.hp.get('tol',1e-4)))
        return LinearSVC(C=float(self.hp.get('C',1)),max_iter=int(self.hp.get('max_iter',10000)),
                         tol=float(self.hp.get('tol',1e-4)),random_state=2026,dual='auto')

    def fit(self,x,y,sample_weight=None):
        y=np.asarray(y);n=len(y)
        if n==0 or x.shape[0]!=n:
            raise ValueError('Nonempty aligned training rows are required')
        self.feature_degenerate=self.feature_degenerate or (x.nnz==0 if hasattr(x,'nnz') else not np.any(x))
        weights=np.ones(n) if sample_weight is None else np.asarray(sample_weight,dtype=float)
        if weights.shape!=(n,) or not np.isfinite(weights).all() or (weights<0).any() or weights.sum()<=0:
            raise ValueError('Invalid training sample weights')
        self.estimators=[];self.constants=[]
        with warnings.catch_warnings():
            warnings.simplefilter('error',ConvergenceWarning)
            if self.task in {'multilabel','multi_label'}:
                if y.shape!=(n,self.label_count) or not np.isin(y,[0,1]).all():
                    raise ValueError('Multilabel training targets have incorrect shape')
                for col in range(self.label_count):
                    prior=float(np.average(y[:,col],weights=weights))
                    constant=self.feature_degenerate or len(np.unique(y[:,col]))==1
                    model=None if constant else self._base().fit(x,y[:,col],sample_weight=weights)
                    self.estimators.append(model);self.constants.append(prior)
            else:
                if y.ndim!=1 or not np.isin(y,np.arange(self.label_count)).all():
                    raise ValueError('Single-label targets outside declared space')
                self.classes=np.unique(y).astype(int)
                self.priors=np.bincount(y.astype(int),weights=weights,minlength=self.label_count)/weights.sum()
                self.estimator=None if self.feature_degenerate or len(self.classes)==1 else self._base().fit(x,y,sample_weight=weights)
        return self

    def _constant(self,prior):
        return prior if self.score_kind=='probability' else 2*prior-1

    def scores(self,x) -> np.ndarray:
        n=x.shape[0]
        if self.task in {'multilabel','multi_label'}:
            result=np.empty((n,self.label_count),dtype=np.float64)
            for col,model in enumerate(self.estimators):
                if model is None:result[:,col]=self._constant(self.constants[col])
                elif self.score_kind=='probability':result[:,col]=model.predict_proba(x)[:,1]
                else:result[:,col]=model.decision_function(x)
            return result
        if self.estimator is None:
            priors=self.priors if self.score_kind=='probability' else 2*self.priors-1
            return np.broadcast_to(priors,(n,self.label_count)).copy()
        if self.score_kind=='probability':
            result=np.zeros((n,self.label_count));result[:,self.estimator.classes_.astype(int)]=self.estimator.predict_proba(x)
        else:
            result=np.full((n,self.label_count),-np.finfo(np.float32).max,dtype=np.float64)
            margin=self.estimator.decision_function(x)
            if margin.ndim==1:margin=np.column_stack([-margin,margin])
            result[:,self.estimator.classes_.astype(int)]=margin
        if not np.isfinite(result).all():raise ValueError('Nonfinite classifier scores')
        return result

    def predict(self,x):
        score=self.scores(x)
        return ((score>=(.5 if self.score_kind=='probability' else 0)).astype(int)
                if self.task in {'multilabel','multi_label'} else score.argmax(axis=1))


def build_model(model_id: str,task: str,label_count: int,assets: dict[str,Any],
                hp: dict[str,Any],features: Any=None):
    if task not in {'single_label','multilabel','multi_label'} or label_count<2:
        raise ValueError('Unknown task or invalid label count')
    if model_id in {'tfidf_lr','tfidf_svm','w2v_svm','fixed_tfidf_svm'}:
        settings={**hp,'feature_degenerate':bool(hp.get('feature_degenerate',False) or getattr(features,'degenerate',False))}
        return TraditionalClassifier(model_id,task,label_count,settings)
    if model_id=='bilstm':
        if features is None or not hasattr(features,'embedding_matrix'):
            raise ValueError('BiLSTM requires train-only word features')
        from models_extra import BiLSTM
        return BiLSTM(features.embedding_matrix.copy(),hidden_size=int(hp.get('hidden_per_direction',64)),
                      n_classes=label_count,dropout=float(hp.get('dropout',.3)))
    if model_id not in {'distilbert','modernbert','m2_distilbert','frozen_generic_distilbert_head'}:
        raise ValueError(f'Unknown model family: {model_id}')
    family='modernbert' if model_id=='modernbert' else 'distilbert'
    source=assets['models'][family]
    if not re.fullmatch('[0-9a-f]{40}',source.get('revision','')):
        raise ValueError('An immutable model revision is required')
    from transformers import AutoConfig,AutoModelForSequenceClassification
    config=AutoConfig.from_pretrained(source['directory'],local_files_only=True)
    config.num_labels=label_count
    config.problem_type='single_label_classification' if task=='single_label' else 'multi_label_classification'
    if family=='modernbert':config.reference_compile=False
    if hp.get('random_encoder',False):
        return AutoModelForSequenceClassification.from_config(config,attn_implementation='sdpa')
    return AutoModelForSequenceClassification.from_pretrained(source['directory'],config=config,
        local_files_only=True,attn_implementation='sdpa')
