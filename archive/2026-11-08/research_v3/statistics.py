"""Frozen-run contrasts and synchronized cluster bootstrap (design section 9).

Manifest schema (JSON-compatible; extra provenance fields are retained by callers)::

    datasets = {dataset_id: {task, label_count, partition, ap_mask?}}
    runs = {physical_or_view_run_id: {dataset_id, data_seed, training_seed,
            model_id?, evaluation_id?, checkpoint_run_id?}}
    estimates = [{estimate_id, metric, label_ids?, draws: [
        {data_seed, left_runs: [run_id, ...], right_runs: [run_id, ...]}]}]
    contrasts = [{contrast_id, family, kind?, components: [
        {estimate_id, coefficient}]}]

An estimate averages metrics over training runs within an arm, subtracts arms,
then averages actual data draws. A deterministic arm has one physical run;
logical aliases must be resolved before calling. Full pool has one data_seed=0
draw. Components form linear contrasts (including adjacent-condition interactions).
No probabilities or different datasets' metrics are pooled. F1--F6 and auxiliary
families use the same engine. ``family`` optionally selects one family from a
registry; bootstrap refuses an unselected mixture of families.

Run evaluation_id defaults to dataset_id. Runs with the same evaluation_id must
contain the same sample IDs, targets and base groups. Edited evaluation views use
distinct evaluation_ids but retain original group_ids and checkpoint_run_id.
Within each estimate/data draw, all arms and training repeats must additionally
use identical base sample_id/gold/group identities, even across evaluation views.
Unrelated estimates may evaluate distinct subsets. Multilabel PredictionRow
labels and decisions are canonical sparse label-index tuples, including an empty
predicted tuple; they are encoded to full-width binary matrices internally.
For interactions involving a full-pool component, its single seed0 delta may be
broadcast as a fixed constant for seed-sign sensitivity; this never adds fits.
Other mismatched data seeds produce an explicit unavailable seed sensitivity.

Macro-AP requires the partition's independently frozen ap_mask. Its excluded
columns remain excluded in every bootstrap. Missing-positive mask columns use
AP=0, missing-negative columns AP=1; combined degeneracy rate above .05 is flagged.
Predictions supply the frozen decisions for F1/accuracy; scores supply AP.
All draws use global group multiplicities as row weights, including repeated
and edited members. An empty dataset draw is retained as undefined and blocks
intervals rather than silently deleting or redrawing a replicate.

Intervals are conditional on saved fitted models. Training SD/data seed deltas
are separate descriptive summaries; no future-training claim or nested seed
sensitivity is implied by these intervals. Three seeds provide limited evidence.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from .types import PredictionRow


def exact_mcnemar(intervention_correct,control_correct,group_ids) -> dict:
    """Exact two-sided paired-binomial sensitivity; one independent base group/row."""
    from scipy.stats import binomtest
    a=np.asarray(intervention_correct);b=np.asarray(control_correct);groups=list(group_ids)
    if (a.ndim!=1 or not len(a) or b.shape!=a.shape or len(groups)!=len(a)
            or not np.isin(a,[0,1]).all() or not np.isin(b,[0,1]).all()):
        raise ValueError('McNemar needs aligned nonempty binary paired correctness')
    if any(not isinstance(g,str) or not g for g in groups) or len(set(groups))!=len(groups):
        raise ValueError('Exact McNemar requires one independent response per base group')
    a=a.astype(bool);b=b.astype(bool)
    wins=int((a & ~b).sum());losses=int((~a & b).sum());discordant=wins+losses
    return {'n':len(a),'intervention_only_correct':wins,'control_only_correct':losses,
        'both_correct':int((a & b).sum()),'both_wrong':int((~a & ~b).sum()),
        'discordant':discordant,'correctness_difference':(wins-losses)/len(a),
        'p_value':float(binomtest(wins,discordant,.5,alternative='two-sided').pvalue) if discordant else 1.,
        'role':'sensitivity_only','unit':'independent_base_group',
        'primary_inference':'paired_group_bootstrap_with_F6_simultaneous_bands'}


def classify_effect(estimate: float, lower: float, upper: float,
                    seed_deltas: tuple[float, ...], margin: float = .01) -> str:
    """Classify only a simultaneous interval, with strict advantage boundaries."""
    values = np.asarray((estimate, lower, upper, margin, *seed_deltas), dtype=float)
    if not np.isfinite(values).all() or lower > upper or margin < 0:
        raise ValueError('Effect bounds, estimates and seed deltas must be finite and ordered')
    if lower >= -margin and upper <= margin:
        return 'practically_close'
    if lower > margin:
        return 'positive_advantage' if seed_deltas and all(x > 0 for x in seed_deltas) else 'condition_sensitive'
    if upper < -margin:
        return 'negative_advantage' if seed_deltas and all(x < 0 for x in seed_deltas) else 'condition_sensitive'
    return 'uncertain'


def classify_reversal(first: str, second: str, interaction_lower: float,
                      interaction_upper: float) -> bool:
    """Opposite supported advantages additionally require an interaction band excluding 0."""
    if not np.isfinite([interaction_lower, interaction_upper]).all() or interaction_lower > interaction_upper:
        raise ValueError('Invalid interaction interval')
    return ({first, second} == {'positive_advantage', 'negative_advantage'}
            and (interaction_lower > 0 or interaction_upper < 0))


def tail_group(*, head_order: list[int] | None = None,
               frequencies: list[int] | None = None) -> dict[str, Any]:
    """Current E3/M3 order or natural lowest third with ties; uniform natural is NA."""
    if (head_order is None) == (frequencies is None):
        raise ValueError('Supply exactly one tail definition')
    if head_order is not None:
        if not head_order or set(head_order) != set(range(len(head_order))):
            raise ValueError('Head order must be a complete label permutation')
        return {'tail_definition': 'current_head_permutation',
                'label_ids': list(head_order[-max(1, len(head_order)//3):])}
    f = np.asarray(frequencies, dtype=float)
    if f.ndim != 1 or not len(f) or not np.isfinite(f).all() or (f < 0).any():
        raise ValueError('Training frequencies must be a finite nonnegative vector')
    labels = None if np.all(f == f[0]) else np.flatnonzero(f <= np.sort(f)[max(1,len(f)//3)-1]).tolist()
    return {'tail_definition': 'natural_training_frequency_with_boundary_ties', 'label_ids': labels}


def compute_simultaneous_bands(point_estimates: dict[str, float],
                               bootstrap_estimates: Any,
                               confidence: float = .95) -> dict[str, Any]:
    """Unstudentized max-centered-error bands and separate percentile/Holm sensitivity.

    Matrix columns follow point_estimates insertion order; rows are synchronized
    replicates. Quantiles use NumPy's explicit linear empirical interpolation.
    """
    ids = list(point_estimates)
    points = np.asarray(list(point_estimates.values()), dtype=float)
    matrix = np.asarray(bootstrap_estimates, dtype=float)
    if (not ids or matrix.ndim != 2 or matrix.shape[1] != len(ids) or not len(matrix)
            or not np.isfinite(points).all() or not np.isfinite(matrix).all()
            or not 0 < confidence < 1):
        raise ValueError('Bands require finite points and a nonempty B by contrast matrix')
    error = matrix-points
    critical = float(np.quantile(np.max(np.abs(error), axis=1), confidence, method='linear'))
    p = (1+(np.abs(error) >= np.abs(points)).sum(axis=0))/(len(matrix)+1)
    order = np.argsort(p, kind='stable')
    holm = np.empty(len(ids))
    holm[order] = np.minimum(1, np.maximum.accumulate(p[order]*(len(ids)-np.arange(len(ids)))))
    lo, hi = np.quantile(matrix, [(1-confidence)/2, (1+confidence)/2], axis=0, method='linear')
    return {'critical': critical, 'confidence': confidence, 'quantile_method': 'linear',
            'contrasts': {key: {'estimate': float(points[j]),
                'simultaneous_lower': float(points[j]-critical),
                'simultaneous_upper': float(points[j]+critical),
                'pointwise_lower': float(lo[j]), 'pointwise_upper': float(hi[j]),
                'centered_p': float(p[j]), 'holm_p': float(holm[j])}
                for j,key in enumerate(ids)}}


def _unique_records(records: list[dict], key: str) -> dict[str, dict]:
    result = {}
    for item in records:
        name = item[key]
        if name in result:
            raise ValueError(f'Duplicate {key}: {name}')
        result[name] = item
    return result


def _prepare(predictions: tuple[PredictionRow, ...], manifest: dict) -> dict:
    contrasts = _unique_records(manifest['contrasts'], 'contrast_id')
    if manifest.get('family') is not None:
        contrasts = {k:v for k,v in contrasts.items() if v['family'] == manifest['family']}
    if not contrasts:
        raise ValueError('No contrasts in requested family')
    all_estimates = _unique_records(manifest['estimates'], 'estimate_id')
    requested = set()
    for contrast in contrasts.values():
        if not contrast['components']:
            raise ValueError('Contrast needs components')
        for part in contrast['components']:
            if part['estimate_id'] not in all_estimates or not np.isfinite(part['coefficient']):
                raise ValueError('Unknown estimate or nonfinite contrast coefficient')
            requested.add(part['estimate_id'])
    estimates = {k:v for k,v in all_estimates.items() if k in requested}
    run_ids = set()
    for estimate in estimates.values():
        seen = set()
        if not estimate['draws']:
            raise ValueError('Estimate needs actual data draws')
        for draw in estimate['draws']:
            if draw['data_seed'] in seen:
                raise ValueError('Duplicate actual data seed draw')
            seen.add(draw['data_seed'])
            for arm in ('left_runs','right_runs'):
                ids = draw[arm]
                if not ids or len(set(ids)) != len(ids):
                    raise ValueError('Empty arm or duplicate physical run')
                seeds=[]; physical=[]
                for run in ids:
                    if run not in manifest['runs']:
                        raise ValueError(f'Missing run metadata: {run}')
                    meta=manifest['runs'][run]
                    if meta['data_seed'] != draw['data_seed']:
                        raise ValueError('Run data seed disagrees with actual draw')
                    seeds.append(meta.get('training_seed'))
                    physical.append(meta.get('checkpoint_run_id',run))
                if len(set(physical)) != len(physical):
                    raise ValueError('Logical aliases cannot duplicate a physical fit within an arm')
                if len(set(seeds)) != len(seeds):
                    raise ValueError('Duplicate training seed or deterministic fits in arm')
                run_ids.update(ids)
    rows=defaultdict(list)
    for row in predictions:
        if row.run_id in run_ids:
            rows[row.run_id].append(row)
    compiled={}
    views={}
    for run in sorted(run_ids):
        rr=sorted(rows[run],key=lambda r:r.sample_id)
        if not rr:
            raise ValueError(f'Missing predictions for run {run}')
        meta=manifest['runs'][run]
        ds=manifest['datasets'][meta['dataset_id']]
        c=ds['label_count']
        if not isinstance(c,int) or c < 1:
            raise ValueError('Declared label count must be positive')
        task=ds['task']
        multi=task in {'multilabel','multi_label'}
        if not multi and task != 'single_label':
            raise ValueError('Unknown task')
        if len({r.sample_id for r in rr}) != len(rr):
            raise ValueError('Duplicate sample in run')
        if any(not r.group_id or r.score_kind not in {'probability','logit','svm_margin'} for r in rr):
            raise ValueError('Missing base group or unknown score kind')
        scores=np.asarray([r.scores for r in rr],dtype=float)
        if scores.shape != (len(rr),c) or not np.isfinite(scores).all():
            raise ValueError('Missing or nonfinite full score columns')
        if any(r.score_kind=='probability' and (min(r.scores)<0 or max(r.scores)>1) for r in rr):
            raise ValueError('Probability scores outside [0,1]')
        for record in rr:
            if not record.labels:
                raise ValueError('Gold labels must be nonempty canonical indices')
            for indices in (record.labels,record.prediction):
                if (any(not isinstance(i,(int,np.integer)) or isinstance(i,(bool,np.bool_))
                        or not 0 <= i < c for i in indices)
                        or tuple(indices)!=tuple(sorted(set(indices)))):
                    raise ValueError('Gold/predicted labels must be unique canonical in-range indices')
            if not multi and (len(record.labels)!=1 or len(record.prediction)!=1):
                raise ValueError('Single-label targets/decisions require one valid class')
        if multi:
            y=np.zeros((len(rr),c),dtype=bool)
            pred=np.zeros_like(y)
            for i,record in enumerate(rr):
                y[i,list(record.labels)]=True
                pred[i,list(record.prediction)]=True
        else:
            y=np.asarray([record.labels[0] for record in rr])
            pred=np.asarray([record.prediction[0] for record in rr])
        view=(meta['dataset_id'],meta.get('evaluation_id',meta['dataset_id']))
        identity=tuple((r.sample_id,r.labels,r.group_id) for r in rr)
        if view in views and views[view] != identity:
            raise ValueError(f'Missing or mismatched paired evaluation rows: {run}')
        views[view]=identity
        compiled[run]={'rows':rr,'y':y,'pred':pred,'scores':scores,'dataset_id':meta['dataset_id'],
                       'multi':multi,'label_count':c,'groups':tuple(r.group_id for r in rr),
                       'identity':identity}
    for estimate in estimates.values():
        for draw in estimate['draws']:
            paired_runs=draw['left_runs']+draw['right_runs']
            expected=compiled[paired_runs[0]]['identity']
            if any(compiled[run]['identity']!=expected for run in paired_runs[1:]):
                raise ValueError('Contrasted arms require matching paired base sample/gold/group identity')
        used={r for d in estimate['draws'] for a in ('left_runs','right_runs') for r in d[a]}
        if len({compiled[r]['dataset_id'] for r in used}) != 1:
            raise ValueError('Cannot average metrics across datasets in one estimate')
        ds=manifest['datasets'][compiled[next(iter(used))]['dataset_id']]
        if estimate['metric'] not in {'macro_f1','macro_ap','accuracy','correctness'}:
            raise ValueError('Unknown metric')
        ids=estimate.get('label_ids')
        if ids is not None and (not ids or len(set(ids))!=len(ids) or not set(ids)<=set(range(ds['label_count']))):
            raise ValueError('Invalid fixed label subgroup')
        if estimate['metric']=='macro_ap':
            mask=ds.get('ap_mask')
            if not mask or len(set(mask))!=len(mask) or not set(mask)<=set(range(ds['label_count'])):
                raise ValueError('Macro AP needs an independently frozen valid ap_mask')
            if ids is not None and not set(ids)<=set(mask):
                raise ValueError('AP subgroup must be within frozen mask')
            for run in used:
                data=compiled[run]
                if not data['multi']:
                    raise ValueError('Macro AP requires multilabel task')
                support=data['y'].sum(axis=0)
                if any(not 0 < support[i] < len(data['y']) for i in mask):
                    raise ValueError('Frozen AP mask requires positive and negative original targets')
    for contrast in contrasts.values():
        datasets={manifest['runs'][run]['dataset_id'] for part in contrast['components']
                  for draw in estimates[part['estimate_id']]['draws']
                  for arm in ('left_runs','right_runs') for run in draw[arm]}
        if len(datasets)!=1:
            raise ValueError('Cannot combine different datasets/tasks in a contrast')
    return {'contrasts':contrasts,'estimates':estimates,'runs':compiled,'manifest':manifest}


def _metric(data: dict, metric: str, label_ids: list | None, ds: dict,
            weights: np.ndarray) -> float:
    if not weights.sum():
        raise ValueError('Empty dataset in synchronized bootstrap replicate')
    y,pred=data['y'],data['pred']
    if metric in {'accuracy','correctness'}:
        correct=np.all(y==pred,axis=1) if data['multi'] else y==pred
        return float(np.dot(weights,correct)/weights.sum())
    ids=label_ids if label_ids is not None else (ds['ap_mask'] if metric=='macro_ap' else range(data['label_count']))
    values=[]
    for i in ids:
        target=y[:,i] if data['multi'] else y==i
        if metric=='macro_ap':
            positive=float(np.dot(weights,target))
            if positive==0:
                values.append(0.); continue
            if positive==weights.sum():
                values.append(1.); continue
            order=np.argsort(-data['scores'][:,i],kind='stable')
            score=data['scores'][order,i]
            ends=np.r_[np.flatnonzero(np.diff(score)),len(score)-1]
            tp=np.cumsum(weights[order]*target[order])[ends]
            total=np.cumsum(weights[order])[ends]
            precision=np.divide(tp,total,out=np.zeros_like(tp),where=total!=0)
            values.append(float(np.dot(np.diff(np.r_[0.,tp]),precision)/positive))
        else:
            decision=pred[:,i] if data['multi'] else pred==i
            tp=np.dot(weights,target & decision)
            denom=np.dot(weights,target)+np.dot(weights,decision)
            values.append(float(2*tp/denom) if denom else 0.)
    return float(np.mean(values))


def _evaluate(prepared: dict, group_weights: dict[str,int] | None = None) -> dict:
    estimates={}
    cache={}
    for key,estimate in prepared['estimates'].items():
        run_metrics={}; seed_deltas={}; training_sd={}
        for draw in estimate['draws']:
            arm_values={}
            for arm in ('left_runs','right_runs'):
                values=[]
                for run in draw[arm]:
                    cache_key=(run,estimate['metric'],tuple(estimate.get('label_ids',[])))
                    if cache_key not in cache:
                        data=prepared['runs'][run]
                        weights=np.ones(len(data['y'])) if group_weights is None else np.asarray([group_weights[g] for g in data['groups']],dtype=float)
                        ds=prepared['manifest']['datasets'][data['dataset_id']]
                        cache[cache_key]=_metric(data,estimate['metric'],estimate.get('label_ids'),ds,weights)
                    values.append(cache[cache_key]);run_metrics[run]=cache[cache_key]
                arm_values[arm]=values
            seed=str(draw['data_seed'])
            seed_deltas[seed]=float(np.mean(arm_values['left_runs'])-np.mean(arm_values['right_runs']))
            training_sd[seed]={arm.removesuffix('_runs'):float(np.std(v,ddof=1)) if len(v)>1 else None
                               for arm,v in arm_values.items()}
        estimates[key]={'estimate':float(np.mean(list(seed_deltas.values()))),'seed_deltas':seed_deltas,
                        'training_sd':training_sd,'run_metrics':run_metrics}
    contrasts={}
    for key,contrast in prepared['contrasts'].items():
        parts=contrast['components']
        delta=float(sum(p['coefficient']*estimates[p['estimate_id']]['estimate'] for p in parts))
        seed_sets=[set(estimates[p['estimate_id']]['seed_deltas']) for p in parts]
        variable=[s for s in seed_sets if s!={'0'}]
        aligned=not variable or all(s==variable[0] for s in variable)
        seeds=sorted(variable[0] if variable else {'0'}) if aligned else []
        seed_deltas={s:float(sum(p['coefficient']*estimates[p['estimate_id']]['seed_deltas'][
            '0' if seed_sets[i]=={'0'} else s] for i,p in enumerate(parts))) for s in seeds}
        contrasts[key]={'estimate':delta,'seed_deltas':seed_deltas,'family':contrast['family'],
                        'kind':contrast.get('kind','difference'),
                        'seed_sensitivity_status':'aligned_with_fixed_full_pool' if aligned and {'0'} in seed_sets and variable else ('aligned' if aligned else 'unavailable_mismatched_seeds')}
    return {'estimates':estimates,'contrasts':contrasts}


def estimate_contrasts(predictions: tuple[PredictionRow,...], contrast_manifest: dict) -> dict:
    """Recompute saved per-run metrics; never ensemble scores or create fits."""
    return _evaluate(_prepare(predictions,contrast_manifest))


def bootstrap_family(predictions: tuple[PredictionRow,...], contrast_manifest: dict,
                     replicates: int = 2000, seed: int = 7301) -> dict:
    """One synchronized global union cluster draw for every replicate of one family."""
    if not isinstance(replicates,int) or isinstance(replicates,bool) or replicates < 1:
        raise ValueError('Replicates must be a positive integer')
    prepared=_prepare(predictions,contrast_manifest)
    families={c['family'] for c in prepared['contrasts'].values()}
    if len(families)!=1:
        raise ValueError('Select exactly one family before bootstrap')
    result=_evaluate(prepared)
    groups=sorted({g for data in prepared['runs'].values() for g in data['groups']})
    rng=np.random.default_rng(seed)
    values={key:[] for key in result['contrasts']}
    invalid=[]
    # Distinct evaluation views are tracked separately, without counting model copies.
    ap_views={}
    for estimate in prepared['estimates'].values():
        if estimate['metric']!='macro_ap': continue
        for draw in estimate['draws']:
            for arm in ('left_runs','right_runs'):
                for run in draw[arm]:
                    meta=contrast_manifest['runs'][run]; dsid=meta['dataset_id']
                    view=meta.get('evaluation_id',dsid)
                    ap_views[(dsid,view)]=prepared['runs'][run]
    degenerate={key:{i:[0,0] for i in contrast_manifest['datasets'][key[0]]['ap_mask']} for key in ap_views}
    for b in range(replicates):
        counts=np.bincount(rng.integers(0,len(groups),size=len(groups)),minlength=len(groups))
        multiplicities=dict(zip(groups,counts))
        for key,data in ap_views.items():
            weights=np.asarray([multiplicities[g] for g in data['groups']])
            for i,counter in degenerate[key].items():
                positive=float(np.dot(weights,data['y'][:,i]))
                counter[0]+=int(positive==0)
                counter[1]+=int(weights.sum()>0 and positive==weights.sum())
        try:
            boot=_evaluate(prepared,multiplicities)
        except ValueError as exc:
            if 'Empty dataset' not in str(exc): raise
            invalid.append({'replicate':b,'status':'empty_dataset','reason':str(exc)})
            for key in values: values[key].append(None)
            continue
        for key in values: values[key].append(boot['contrasts'][key]['estimate'])
    result.update({'family':next(iter(families)),'replicates':replicates,'seed':seed,
                   'group_ids':groups,'bootstrap_estimates':values,'invalid_replicates':invalid,
                   'uncertainty_scope':'test_sampling_conditional_on_fitted_models',
                   'seed_limitation':'Few data/training seeds; no future-training inference',
                   'ap_masks':{dsid:list(contrast_manifest['datasets'][dsid]['ap_mask']) for dsid,_ in ap_views}})
    ap_diagnostics={}
    for (dsid,view),labels in degenerate.items():
        view_diag={str(i):{'no_positive_rate':n[0]/replicates,'no_negative_rate':n[1]/replicates,
                          'degenerate_rate':sum(n)/replicates,
                          'unstable':sum(n)/replicates>.05} for i,n in labels.items()}
        if view==dsid: ap_diagnostics[dsid]=view_diag
        else: ap_diagnostics.setdefault(dsid,{})[view]=view_diag
    result['ap_degeneracy']=ap_diagnostics
    if invalid:
        result.update(status='undefined_bootstrap',critical=None)
        for contrast in result['contrasts'].values():
            contrast.update(simultaneous_lower=None,simultaneous_upper=None,pointwise_lower=None,
                            pointwise_upper=None,centered_p=None,holm_p=None,classification='uncertain')
        return result
    matrix=np.asarray([values[k] for k in result['contrasts']]).T
    bands=compute_simultaneous_bands({k:v['estimate'] for k,v in result['contrasts'].items()},matrix)
    result.update(status='ok',critical=bands['critical'],confidence=bands['confidence'],quantile_method=bands['quantile_method'])
    for key,contrast in result['contrasts'].items():
        contrast.update(bands['contrasts'][key])
        contrast['classification']=classify_effect(contrast['estimate'],contrast['simultaneous_lower'],
            contrast['simultaneous_upper'],tuple(contrast['seed_deltas'].values()))
    return result
