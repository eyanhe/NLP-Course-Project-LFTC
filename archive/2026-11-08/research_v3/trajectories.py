"""Descriptive, paired failure paths for design section8.4; no causal inference.

``trace_failures(predictions, paths, task, label_count)`` consumes saved rows only.
Paths are caller-declared, never discovered by sorting conditions or scores::

    {path_id: {model_id, data_seed, training_seed, steps: [
        {run_id, condition_id, condition_name?, data_seed?, training_seed?, model_id?},
        ...]}}

Each path has at least two ordered, distinct real runs/conditions, for one model,
data draw and training seed. ``training_seed=None`` describes deterministic fits.
A full-pool step may explicitly declare data_seed0; its shared physical run may
appear in multiple data-seed paths without becoming another fit. It must retain
the same model and training seed wherever referenced. Callers resolve logical
aliases to actual run IDs before calling. Distinct paths remain independent
descriptions: no ensemble, vote, cross-seed averaging or inflated fit count.

Every run in a path must have exactly matching sample/gold/base-group identities.
Multilabel gold and predictions use sorted unique sparse indices; an empty
predicted tuple is legal. Full scores/decisions and exact FN/FP label IDs remain
available for each run. A sample's main correctness is exact label-set match;
per-label FN/FP sequences preserve changes hidden by that aggregate correctness.

Patterns: always wrong=persistent_error; always right=persistent_correct;
one wrong-to-right change=corrected; one right-to-wrong change=new_error;
two or more changes=unstable. Transitions use the declared step order.
Each label's FN/FP classification applies these same patterns to presence or
absence of that error (persistent_correct therefore means that error is absent).

All causes remain unresolved. Phenomenon tags or unverified ``cause`` metadata
cannot establish mechanisms. Human annotations, raw examples, paired intervention
effects and reviewed causal evidence can be joined externally by stable IDs.
Sample counts are descriptive coverage, not weighted population estimates,
confidence intervals, independent case evidence or counts of unique model fits.
"""
from __future__ import annotations

from collections import defaultdict
from math import isfinite
from numbers import Integral
from typing import Any

from .types import PredictionRow


_PATTERNS=('persistent_error','persistent_correct','corrected','new_error','unstable')


def _pattern(correctness: list[bool]) -> str:
    changes=sum(a!=b for a,b in zip(correctness,correctness[1:]))
    if changes>=2: return 'unstable'
    if not changes: return 'persistent_correct' if correctness[0] else 'persistent_error'
    return 'corrected' if correctness[-1] else 'new_error'


def _seed(value: Any) -> bool:
    return isinstance(value,Integral) and not isinstance(value,bool)


def _validate_row(row: PredictionRow, multi: bool, label_count: int) -> None:
    if not row.sample_id or not row.group_id or not row.input_hash:
        raise ValueError('Missing sample/group/input identity in prediction')
    if not row.labels:
        raise ValueError('Gold labels must be nonempty canonical indices')
    for indices in (row.labels,row.prediction):
        if (any(not _seed(i) or not 0<=i<label_count for i in indices)
                or tuple(indices)!=tuple(sorted(set(indices)))):
            raise ValueError('Gold/predicted labels must be unique sorted in-range indices')
    if not multi and (len(row.labels)!=1 or len(row.prediction)!=1):
        raise ValueError('Single-label rows require exactly one gold and predicted label')
    if (row.score_kind not in {'probability','logit','svm_margin'}
            or len(row.scores)!=label_count or not all(isfinite(x) for x in row.scores)):
        raise ValueError('Prediction scores require a known scale and finite full-label vector')
    if row.score_kind=='probability' and any(not 0<=x<=1 for x in row.scores):
        raise ValueError('Probability scores must lie in [0,1]')


def _paths(paths: dict) -> tuple[dict, set[str]]:
    if not isinstance(paths,dict) or not paths:
        raise ValueError('Declare at least one ordered path')
    prepared={};run_identity={}
    for path_id,path in paths.items():
        if not path_id or not isinstance(path,dict) or not path.get('model_id'):
            raise ValueError('Each path needs an ID and model identity')
        if not _seed(path.get('data_seed')) or ('training_seed' not in path
                or path['training_seed'] is not None and not _seed(path['training_seed'])):
            raise ValueError('Each path needs valid data/training seed identity')
        steps=path.get('steps')
        if not isinstance(steps,(list,tuple)) or len(steps)<2:
            raise ValueError('Each path needs at least two ordered steps')
        seen_runs=set();seen_conditions=set();ordered=[]
        for step in steps:
            if not isinstance(step,dict) or not step.get('run_id') or not step.get('condition_id'):
                raise ValueError('Each path step needs real run and condition IDs')
            run=step['run_id'];condition=step['condition_id']
            if run in seen_runs or condition in seen_conditions:
                raise ValueError('Duplicate real run or condition in path; resolve aliases first')
            seen_runs.add(run);seen_conditions.add(condition)
            model=step.get('model_id',path['model_id'])
            train=step.get('training_seed',path['training_seed'])
            data=step.get('data_seed',path['data_seed'])
            if model!=path['model_id'] or train!=path['training_seed']:
                raise ValueError('A path cannot mix model/training seed identities')
            if not _seed(data) or data not in {0,path['data_seed']}:
                raise ValueError('A path cannot mix data seeds except an explicit full-pool seed0')
            identity=(model,data,train)
            if run in run_identity and run_identity[run]!=identity:
                raise ValueError('Shared real run cannot change model or data/training seed identity')
            run_identity[run]=identity
            ordered.append({'run_id':run,'condition_id':condition,
                'condition_name':step.get('condition_name',condition),'model_id':model,
                'data_seed':int(data),'training_seed':int(train) if train is not None else None})
        prepared[path_id]={'model_id':path['model_id'],'data_seed':int(path['data_seed']),
                          'training_seed':int(path['training_seed']) if path['training_seed'] is not None else None,
                          'steps':ordered}
    return prepared,set(run_identity)


def trace_failures(predictions: tuple[PredictionRow,...], paths: dict,
                   task: str, label_count: int) -> dict:
    """Return deterministic failure content for explicit paired run paths."""
    if not _seed(label_count) or label_count<1:
        raise ValueError('Declared label count must be positive')
    multi=task in {'multilabel','multi_label'}
    if not multi and task!='single_label': raise ValueError('Unknown task')
    declared,requested=_paths(paths)
    indexed=defaultdict(dict)
    for row in predictions:
        if row.run_id not in requested: continue
        _validate_row(row,multi,label_count)
        if row.sample_id in indexed[row.run_id]:
            raise ValueError('Duplicate run/sample prediction row')
        indexed[row.run_id][row.sample_id]=row
    for run in requested:
        if not indexed[run]: raise ValueError(f'Missing predictions for real run {run}')
    result={}
    for path_id,path in declared.items():
        steps=path['steps'];first=indexed[steps[0]['run_id']]
        identity={key:(row.labels,row.group_id) for key,row in first.items()}
        for step in steps[1:]:
            candidate=indexed[step['run_id']]
            if {key:(row.labels,row.group_id) for key,row in candidate.items()}!=identity:
                raise ValueError('All path conditions require identical paired sample/gold/group identity')
        samples=[];counts=dict.fromkeys(_PATTERNS,0)
        for sample_id in sorted(first):
            gold=set(first[sample_id].labels);runs=[];correctness=[]
            for step in steps:
                row=indexed[step['run_id']][sample_id];decision=set(row.prediction)
                correct=gold==decision;correctness.append(correct)
                runs.append({**step,'labels':list(row.labels),'prediction':list(row.prediction),
                    'scores':list(row.scores),'score_kind':row.score_kind,'input_hash':row.input_hash,
                    'correct':correct,'false_negative_label_ids':sorted(gold-decision),
                    'false_positive_label_ids':sorted(decision-gold)})
            classification=_pattern(correctness);counts[classification]+=1
            transitions=[]
            for i,(before,after) in enumerate(zip(correctness,correctness[1:])):
                transition=('remained_correct' if before else 'remained_error') if before==after else ('corrected' if after else 'new_error')
                transitions.append({'from_run_id':steps[i]['run_id'],'to_run_id':steps[i+1]['run_id'],
                    'from_condition_id':steps[i]['condition_id'],'to_condition_id':steps[i+1]['condition_id'],
                    'from_correct':before,'to_correct':after,'transition':transition})
            sample={'sample_id':sample_id,'group_id':first[sample_id].group_id,
                    'labels':list(first[sample_id].labels),'classification':classification,
                    'correctness':correctness,'change_count':sum(a!=b for a,b in zip(correctness,correctness[1:])),
                    'runs':runs,'transitions':transitions,'cause_status':'unresolved','cause_evidence':[]}
            if multi:
                per_label=[]
                for label in range(label_count):
                    fn=[label in run['false_negative_label_ids'] for run in runs]
                    fp=[label in run['false_positive_label_ids'] for run in runs]
                    per_label.append({'label_id':label,'gold_positive':label in gold,
                        'false_negative':fn,'false_positive':fp,
                        'fn_classification':_pattern([not x for x in fn]),
                        'fp_classification':_pattern([not x for x in fp]),
                        'classification':_pattern([not(a or b) for a,b in zip(fn,fp)])})
                sample['per_label']=per_label
            samples.append(sample)
        result[path_id]={**path,'samples':samples,'sample_count':len(samples),
                         'counts':counts,'cause_status':'unresolved'}
    return {'task':task,'label_count':int(label_count),'paths':result,
            'unique_run_ids':sorted(requested),'unique_run_count':len(requested),
            'cause_status':'unresolved','scope':'descriptive_paired_failure_trajectories'}
