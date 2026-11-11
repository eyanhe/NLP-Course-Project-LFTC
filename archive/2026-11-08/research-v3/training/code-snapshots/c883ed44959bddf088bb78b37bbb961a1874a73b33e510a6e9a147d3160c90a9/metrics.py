"""Full-target metrics with explicit margin/logit/probability semantics."""
from __future__ import annotations

from typing import Any
import numpy as np
from sklearn.metrics import (average_precision_score,confusion_matrix,f1_score,
                             precision_recall_fscore_support)


def metric_bundle(labels: Any,scores: Any,task: str,score_kind: str,
                  threshold: float|None=None,label_mask: Any=None) -> dict[str,Any]:
    if score_kind not in {'probability','svm_margin','logit'}:
        raise ValueError(f'Unknown score kind: {score_kind}')
    y=np.asarray(labels);s=np.asarray(scores,dtype=np.float64)
    if s.ndim!=2 or len(s)!=len(y) or not len(y) or not np.isfinite(s).all():
        raise ValueError('Scores must be a finite nonempty N by declared-label matrix')
    if score_kind=='probability' and ((s<0).any() or (s>1).any()):
        raise ValueError('Probability scores must be in [0,1]')
    c=s.shape[1]
    multilabel=task in {'multilabel','multi_label'}
    if multilabel:
        if y.shape!=s.shape or not np.isin(y,[0,1]).all():
            raise ValueError('Multilabel labels must be a binary matrix with the full target shape')
        cutoff=(.5 if score_kind=='probability' else 0.) if threshold is None else float(threshold)
        pred=(s>=cutoff).astype(int)
        support=y.sum(axis=0).astype(int)
        valid=[i for i,n in enumerate(support) if 0<n<len(y)]
        mask=valid if label_mask is None else [int(i) for i in label_mask]
        if label_mask is not None and (not mask or len(mask)!=len(set(mask)) or not set(mask)<=set(valid)):
            raise ValueError('AP mask requires both positive and negative examples in this partition')
        aps=[float(average_precision_score(y[:,i],s[:,i])) if i in valid else None for i in range(c)]
        precision,recall,f1,_=precision_recall_fscore_support(y,pred,average=None,zero_division=0)
        return {'task':'multilabel','score_kind':score_kind,'threshold':cutoff,
            'rows':len(y),'macro_ap':float(np.mean([aps[i] for i in mask])) if mask else None,
            'ap_mask':mask,'macro_f1':float(f1.mean()),
            'masked_macro_f1':float(f1[mask].mean()) if mask else None,
            'micro_f1':float(f1_score(y,pred,average='micro',zero_division=0)),
            'per_label':[{'label_id':i,'precision':float(precision[i]),'recall':float(recall[i]),
                          'f1':float(f1[i]),'support':int(support[i]),'ap':aps[i]} for i in range(c)]}
    if task!='single_label' or y.ndim!=1 or not np.isin(y,np.arange(c)).all():
        raise ValueError('Single-label targets must use the declared target indices')
    pred=s.argmax(axis=1)
    precision,recall,f1,support=precision_recall_fscore_support(y,pred,labels=np.arange(c),average=None,zero_division=0)
    return {'task':task,'score_kind':score_kind,'rows':len(y),'macro_f1':float(f1.mean()),
        'micro_f1':float(f1_score(y,pred,labels=np.arange(c),average='micro',zero_division=0)),
        'accuracy':float((pred==y).mean()),'worst_label_f1':float(f1.min()),
        'confusion_matrix':confusion_matrix(y,pred,labels=np.arange(c)).tolist(),
        'per_label':[{'label_id':i,'precision':float(precision[i]),'recall':float(recall[i]),
                      'f1':float(f1[i]),'support':int(support[i])} for i in range(c)]}
