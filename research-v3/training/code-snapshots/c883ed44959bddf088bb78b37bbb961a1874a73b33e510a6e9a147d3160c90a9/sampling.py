"""Deterministic unique-example sampling, with explicit feasibility checks."""
from __future__ import annotations

from collections import Counter, OrderedDict
import hashlib
import math
from typing import Any
import numpy as np

from .config import canonical_hash
from .partitions import stable_order
from .types import Condition, Example, PreparedDataset

_ORDER_CACHE: OrderedDict[tuple,tuple[PreparedDataset,tuple[Example,...]]]=OrderedDict()


def _support(data: PreparedDataset) -> list[int]:
    counts=[0]*len(data.label_names)
    for row in data.partitions['train']:
        for label in row.labels:
            counts[label]+=1
    return counts


def _uniform(sample_id: str, seed: int) -> float:
    return (int.from_bytes(stable_order(sample_id,seed)[:8],'big')+.5)/2**64


def _ordered(data: PreparedDataset, seed: int, alpha: float=0) -> tuple[Example,...]:
    key=(id(data),data.dataset_id,data.manifest_hash,data.task,seed,float(alpha))
    if key in _ORDER_CACHE:
        _ORDER_CACHE.move_to_end(key)
        return _ORDER_CACHE[key][1]
    rows=data.partitions['train']
    if data.task not in {'multilabel','multi_label'}:
        result=tuple(sorted(rows,key=lambda r:stable_order(r.sample_id,seed)))
    else:
        counts=_support(data)
        uncovered={c for c,n in enumerate(counts) if n>0}
        covering=[];chosen=set()
        by_label=[[] for _ in counts]
        for row in rows:
            for label in row.labels:
                by_label[label].append(row)
        while uncovered:
            label=min(uncovered,key=lambda c:(counts[c],c))
            row=min(by_label[label],key=lambda r:(-len(uncovered.intersection(r.labels)),r.sample_id))
            covering.append(row);chosen.add(row.sample_id);uncovered.difference_update(row.labels)
        mean=sum(counts)/len(counts)
        weights=[(n/mean)**alpha if n else 0.0 for n in counts]
        def priority(row):
            weight=sum(weights[c] for c in row.labels)/len(row.labels)
            return (-math.log(_uniform(row.sample_id,seed))/weight,row.sample_id)
        result=tuple(covering)+tuple(sorted((r for r in rows if r.sample_id not in chosen),key=priority))
    _ORDER_CACHE[key]=(data,result)
    while len(_ORDER_CACHE)>6:
        _ORDER_CACHE.popitem(last=False)
    return result


def _condition(data: PreparedDataset, rows: tuple[Example,...], experiment: str, seed: int,
               metadata: dict[str,Any]) -> Condition:
    ids=tuple(r.sample_id for r in rows)
    if len(ids)!=len(set(ids)):
        raise ValueError('Repeated example in an independent-support condition')
    counts=[0]*len(data.label_names)
    cooccurrence=None
    if data.task in {'multilabel','multi_label'}:
        cooccurrence=np.zeros((len(counts),len(counts)),dtype=np.int64)
    for row in rows:
        for c in row.labels:
            counts[c]+=1
        if cooccurrence is not None:
            for c in row.labels:
                for d in row.labels:
                    cooccurrence[c,d]+=1
    sample_hash=hashlib.sha256('\n'.join(ids).encode()).hexdigest()
    natural=_support(data)
    tail_size=max(1,len(counts)//3)
    threshold=sorted(natural)[tail_size-1]
    natural_tail=None if len(set(natural))==1 else [c for c,n in enumerate(natural) if n<=threshold]
    meta={**metadata,'pool_manifest_hash':data.manifest_hash,'sample_hash':sample_hash,
          'rows':len(ids),'label_support':counts,'natural_label_support':natural,
          'natural_rare_labels':natural_tail,'natural_rare_definition':'bottom_third_including_ties_minimum_one; NA_if_all_equal',
          'missing_training_labels':[c for c,n in enumerate(counts) if n==0]}
    if cooccurrence is not None:
        meta['cooccurrence_counts']=cooccurrence.tolist()
        meta['label_marginals']=[n/len(rows) if rows else 0.0 for n in counts]
    identity={'dataset':data.dataset_id,'experiment':experiment,'seed':seed,'sample_hash':sample_hash,
              'pool_hash':data.manifest_hash,'parameters':metadata}
    return Condition(canonical_hash(identity),experiment,data.dataset_id,ids,seed,meta)


def sample_scale(data: PreparedDataset,n: int|str,seed: int) -> Condition:
    full=n=='full' or n==len(data.partitions['train'])
    count=len(data.partitions['train']) if full else int(n)
    if count<1 or count>len(data.partitions['train']):
        raise ValueError(f'infeasible scale capacity: {count}')
    if full:
        seed=0
        rows=tuple(sorted(data.partitions['train'],key=lambda r:r.sample_id))
    else:
        rows=_ordered(data,seed)[:count]
        if data.task in {'multilabel','multi_label'} and set(c for r in rows for c in r.labels)!=set(c for c,n in enumerate(_support(data)) if n):
            raise ValueError('infeasible multilabel coverage within requested budget')
    return _condition(data,rows,'E1',seed,{'size':'full' if full else count,'full_pool':full})


def sample_kshot(data: PreparedDataset,k: int,seed: int) -> Condition:
    if data.task!='single_label' or k<1:
        raise ValueError('K-shot requires a positive shot count and single labels')
    by_label=[[] for _ in data.label_names]
    for row in _ordered(data,seed):
        by_label[row.labels[0]].append(row)
    if any(len(rows)<k for rows in by_label):
        raise ValueError(f'infeasible K-shot capacity for k={k}: {[len(r) for r in by_label]}')
    rows=tuple(row for group in by_label for row in group[:k])
    return _condition(data,rows,'E2',seed,{'shots_per_class':k,'strict_no_val':True,
        'learner_visible_label_budget':len(rows),'target_validation_labels_allowed':False,
        'anchor_hyperparameters_allowed':False,'external_target_vocabulary_allowed':False})


def allocate_long_tail(capacities: tuple[int,...],n: int,rho: int,
                       order: tuple[int,...]) -> tuple[int,...]:
    c=len(capacities)
    if c<2 or set(order)!=set(range(c)) or len(order)!=c or rho<1 or n<c:
        raise ValueError('infeasible long-tail parameters')
    weights=np.empty(c,dtype=float)
    for rank,label in enumerate(order):
        weights[label]=float(rho)**(-rank/(c-1))
    ideal=(n-c)*weights/weights.sum()
    counts=np.floor(ideal).astype(np.int64)+1
    remaining=n-int(counts.sum())
    ranked=sorted(range(c),key=lambda label:(-(ideal[label]-math.floor(ideal[label])),label))
    for label in ranked[:remaining]:
        counts[label]+=1
    if any(int(got)>capacity for got,capacity in zip(counts,capacities)):
        raise ValueError('infeasible long-tail capacity')
    return tuple(int(n) for n in counts)


def shared_tail_budget(capacities: tuple[int,...],orders: tuple[tuple[int,...],...],
                       ratios: tuple[int,...],cap: int=6000) -> int:
    for n in range(min(cap,sum(capacities)),len(capacities)-1,-1):
        try:
            for order in orders:
                for ratio in ratios:
                    allocate_long_tail(capacities,n,ratio,order)
            return n
        except ValueError:
            continue
    raise ValueError('infeasible common long-tail capacity, even at one example per class')


def sample_long_tail(data: PreparedDataset,n: int,rho: int,order: tuple[int,...],seed: int) -> Condition:
    capacities=tuple(_support(data))
    quotas=allocate_long_tail(capacities,n,rho,order)
    by_label=[[] for _ in data.label_names]
    for row in _ordered(data,seed):
        by_label[row.labels[0]].append(row)
    rows=tuple(row for c,group in enumerate(by_label) for row in group[:quotas[c]])
    effective_order=tuple(range(len(order))) if rho==1 else order
    return _condition(data,rows,'E3',seed,{'common_budget':n,'requested_ratio':rho,
        'actual_max_min_ratio':max(quotas)/min(quotas),'head_order':list(effective_order),
        'current_tail_labels':list(order[-max(1,len(order)//3):]) if rho!=1 else None,
        'tail_definition':'bottom_max_1_floor_C_over_3_of_head_permutation; NA_for_balanced'})


def sample_multilabel(data: PreparedDataset,n: int,alpha: float,seed: int) -> Condition:
    if data.task not in {'multilabel','multi_label'} or n<1 or n>len(data.partitions['train']):
        raise ValueError('infeasible multilabel capacity or task')
    if alpha==0:
        return sample_scale(data,n,seed)
    rows=_ordered(data,seed,alpha)[:n]
    if set(c for r in rows for c in r.labels)!=set(c for c,n in enumerate(_support(data)) if n):
        raise ValueError('infeasible multilabel label coverage')
    return _condition(data,rows,'E4',seed,{'size':n,'alpha':alpha,
        'selection':'fixed_greedy_coverage_then_shared_hash_uniform_exponential_keys',
        'changes_label_cooccurrence':True})


def validate_condition_pool(condition: Condition,pool_manifest_hash: str) -> None:
    if condition.metadata['pool_manifest_hash']!=pool_manifest_hash:
        raise ValueError('Condition pool manifest changed')
