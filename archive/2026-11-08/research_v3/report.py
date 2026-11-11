"""Read-only, evidence-driven G0--G5 verification and staged bilingual reports.

``verify_research(root, 'training'|'full_research')`` never fits, predicts, changes
locks or accepts a previous completion.json as proof. Training requires real E0,
all fixed anchor searches/D pilots, both global freezes and every planned model's
byte-bound reload receipt. Saved official predictions are opened only after
verify_freeze('evaluation') succeeds; metrics are reconstructed from scores and
bound to frozen source sample/gold/group IDs and visible-input hashes.

Future human/statistical outputs are located by analysis/research-evidence.json.
This is a routing index, NOT an approval receipt. References are {path,sha256}
under root, and every referenced record is checked/recomputed. Index schema::

    {audit: {frame:ref, reviews:ref, guide:ref, guide_freeze:ref},
     statistics:[{family,path,sha256,contrast_file_sha256}],
     trajectories:{paths:ref,results:[{dataset_id,path_id,path,sha256}]},
     related_work:ref, claims:ref, claim_reviews:ref}

Audit frame JSON has records (draw_audit schema), seeds by dataset, created_at_utc,
evaluation_freeze_hash and guide_freeze_sha256. It contains no model outputs.
Guide-freeze JSON binds guide_sha256 and records frozen_at_utc before frame
creation. Review JSON has records (the existing audit schema) and an envelope
phase='blind_first_pass'; existing annotation records receive no extra fields.
Both distinct original human annotations must exist for every sampled case; a
nonempty reviewer ID is record evidence, not independent identity authentication.
Disagreements require an additional audit.adjudications byte-bound JSON reference.
Its records bind dataset_id/sample_id/group_id, nonempty adjudicator_id/rationale,
gold_status/phenomena and review_record_hashes (canonical hashes of both preserved
original annotations). Extra or stale adjudications are rejected.

Statistics references point to exact bootstrap_family result JSON for F1--F6 and
all declared AUX_M2/M3/M5 families; replicas/seed are recomputed (2000/7301).
Confirmation is authenticated through confirmation.verify_confirmation and
verify_confirmation_audit_frame using the SAME global A envelope. Authenticated
underpowered results remain reportable, but their separate sufficiency check
keeps full_research pending. Neither gate accepts human_review/complete booleans.
Trajectory paths JSON has datasets[dataset].paths;
each result reference is trace_failures output for ONE declared path. These are
recomputed and released one at a time. Related-work CSV requires
citation,url,relevance,difference columns.
Claims JSON contains claims with claim_id,kind,status,text,evidence references;
supported causal claims additionally need exact dataset_id,contrast_id,direction,
scope and cases:{path,sha256}. Cases JSON contains dataset_id and the complete
sample_id/group_id list for that exact effect. Direction requires its reconstructed
SIMULTANEOUS band to exclude0. Scope is local_paired_intervention for F6 or
specified_training_comparison for AUX_M2/M3/M5; arbitrary cross-task cases or
general model-ability claims cannot borrow a significant contrast.
claim_reviews JSON has records with claim_id,reviewer_id,decision='supported',
rationale,claim_content_hash,evidence_hash,dataset_id,contrast_id,direction,scope,
case_set_hash,case_count. At least two distinct actual interpretation records must
bind all fields. The evidence_hash is canonical_hash of contrast_file_sha256,
statistics_file_sha256,contrast (exact reconstructed cell),dataset_id,contrast_id,
direction,scope,cases_file_sha256,case_set_hash,prediction_files (run_id to SHA).
case_set_hash is canonical_hash({dataset_id,cases}) with cases sorted by sample ID.
claim_content_hash binds the entire original claim, including its text. Invalid
or missing interpretations remain unresolved and leave G5 pending. These are
preserved human record identities, not independent proof of a person's identity.

Every trajectory path additionally declares experiment_id. Its ordered steps
must bind actual resolved run/condition/data/training identities, joined through
the design's E1--E5 logical contexts. Representative selection uses data2026,
train42 and E3head9101 when declared, with the same full-pool data0 constant and
balanced E3 baseline. Each applicable dataset/experiment/model with two or more
distinct feasible physical conditions needs a genuine ordered path. Logical E4
alpha0 and E5 baseline aliases remain valid; duplicate physical fits do not create
extra steps. Explicit design infeasibility/one-level exceptions are preserved in
the generated _coverage_ledger; caller-invented exceptions are never accepted.
Missing producer records leave pending gates; no synthetic positive full-research
claim is made.

build_report writes only its explicit output directory: Chinese/English Markdown,
evidence-map.csv and completion.json. It does not redistribute raw text or models.
Evidence maps contain observed SHA256s, gate checks and missing paths. Reports
separate dataset/task metrics, historical-Q exploratory evidence, auxiliary
mechanisms, failure content and limitations; no mixed task metric total is made.
Only authenticated prediction file references are retained across physical runs.
Statistics stream one family's required runs, trajectories stream one path, and
audit sampling uses scalar frozen source identity maps instead of saved scores.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import io
import json
from pathlib import Path
from typing import Any

import numpy as np

from .config import canonical_hash
from .storage import atomic_write_json,bounded_path,hash_file,read_json


class _Evidence:
    def __init__(self,root):
        self.root=root;self.files={};self.missing=[]
        self.gates={f'G{i}':{'complete':False,'checks':{},'failures':[]} for i in range(6)}

    def path(self,relative,expected=None):
        target=bounded_path(self.root/relative,self.root)
        if not target.is_file(): raise FileNotFoundError(f'Required evidence missing: {relative}')
        sha=hash_file(target)
        if expected is not None and sha!=expected: raise ValueError(f'Evidence bytes changed: {relative}')
        self.files[target.relative_to(self.root).as_posix()]={'path':target.relative_to(self.root).as_posix(),
            'sha256':sha,'bytes':target.stat().st_size,'status':'bytes_observed'}
        return target

    def read(self,relative,expected=None): return read_json(self.path(relative,expected))

    def ref(self,record):
        if not isinstance(record,dict) or not record.get('path') or not record.get('sha256'):
            raise ValueError('A byte-bound evidence reference needs path and sha256')
        return self.path(record['path'],record['sha256'])

    def check(self,gate,name,path,action):
        try:
            value=action();self.gates[gate]['checks'][name]={'status':'verified','path':path}
            return value
        except (OSError,ValueError,PermissionError,KeyError,TypeError,ImportError,AttributeError) as exc:
            status='pending' if isinstance(exc,(FileNotFoundError,PermissionError,ImportError)) else 'invalid'
            failure={'check':name,'path':path,'status':status,'reason':str(exc),'error_type':type(exc).__name__}
            self.gates[gate]['failures'].append(failure);self.missing.append({'gate':gate,**failure})
            return None

    def bind_records(self,records):
        for record in records.values(): self.path(record['path'],record['sha256'])


def _runtime(e):
    from .verification import validate_forward_evidence
    runtime=e.read('environment.json');forward=e.read('forward_checks.json')
    if (runtime.get('runtime_ready') is not True or runtime.get('import_errors')
            or not runtime.get('packages') or not runtime.get('gpu',{}).get('arithmetic_verified')):
        raise ValueError('Runtime/locked dependency/GPU arithmetic evidence is incomplete')
    validate_forward_evidence(forward)
    return {'python':runtime.get('python_version'),'device':forward.get('device')}


def _engineering(e,prepared,design):
    from .evaluate import _read_predictions,_metrics,validate_prediction_rows
    from .freeze import _checkpoint
    smoke=e.read('training/engineering-smoke.json');cases=smoke.get('cases',[])
    expected={(d,m) for d in ('go_emotions','banking77','yahoo_answers_topics')
              for m in ('tfidf_lr','tfidf_svm','w2v_svm','bilstm','distilbert')}
    identities=[(c['dataset_id'],c['model_id']) for c in cases]
    if len(identities)!=15 or set(identities)!=expected or smoke.get('design_hash')!=design.get('design_hash'):
        raise PermissionError('All15 actual E0 combinations with saved scores are required')
    for case in cases:
        if (case['training_rows'],case['validation_rows'],case['evaluation_rows'])!=(256,128,128) or case.get('evaluation_source')!='discovery':
            raise ValueError('E0 must use actual256/128/128 T/V/D rows')
        e.path(case['run_status'],case['run_status_sha256'])
        files={};status=_checkpoint(e.root,case['run_id'],files);e.bind_records(files)
        if status.get('training_unique_rows')!=256 or not np.isfinite(status.get('reload_max_absolute_difference',np.nan)):
            raise ValueError('E0 actual fit/reload numerical evidence is missing')
        rows=_read_predictions(e.ref(case['score_rows']));meta=prepared['datasets'][case['dataset_id']]
        validate_prediction_rows(rows,len(meta['label_names']),meta['task'])
        if len(rows)!=128 or any(r.run_id!=case['run_id'] for r in rows): raise ValueError('E0 saved score coverage changed')
        # E0's support mask is computed from its D engineering rows, not Q.
        emeta={**meta,'partitions':{'test':{}}}
        if canonical_hash(_metrics(rows,emeta))!=canonical_hash(case['engineering_metrics']):
            raise ValueError('E0 metrics do not reconstruct from saved scores')
    return len(cases)


def _prepared(e,lock):
    prepared=e.read('data/prepared_manifest.json')
    if not prepared.get('complete') or not prepared.get('datasets'):
        raise PermissionError('Prepared data coverage is incomplete')
    if canonical_hash({k:v for k,v in prepared.items() if k!='prepared_hash'})!=prepared.get('prepared_hash'):
        raise ValueError('Prepared manifest canonical identity changed')
    if lock is None or lock['bindings'].get('prepared_hash')!=prepared['prepared_hash']:
        raise ValueError('Prepared artifacts are not authenticated by the design lock')
    for meta in prepared['datasets'].values():
        for record in meta['partitions'].values(): e.path(record['file'],record['sha256'])
        e.path(meta['inputs']['file'],meta['inputs']['file_sha256'])
    return prepared


def _prediction(e,run,meta,lock,source_cache):
    from .evaluate import _read_predictions,_metrics,validate_prediction_rows,test_exposure_status
    import pyarrow.parquet as pq
    rid=run.get('artifact_id',run['run_id']);ds=run['dataset_id']
    info=e.read(f'predictions/{rid}.manifest.json')
    path=e.path(f'predictions/{rid}.parquet',info['file_sha256']);rows=_read_predictions(path)
    validate_prediction_rows(rows,len(meta['label_names']),meta['task'])
    status=e.read(f'runs/{rid}/status.json');model_request=status['run']
    if any(row.score_kind!=status.get('score_kind') for row in rows):
        raise ValueError('Saved row score_kind differs from the authenticated checkpoint score scale')
    if (info.get('run_id')!=rid or info.get('dataset_id')!=ds or info.get('evaluation_freeze_hash')!=lock['lock_hash']
            or info.get('score_dtype')!='float64' or len(rows)!=meta['partitions']['test']['rows']
            or info.get('rows')!=len(rows) or not rows or any(r.run_id!=rid or r.input_hash!=meta['inputs']['input_hash'] for r in rows)):
        raise ValueError('Saved PredictionRow run/input/freeze/full-Q identity changed')
    if ds not in source_cache:
        source=pq.read_table(e.path(meta['partitions']['test']['file']),columns=['sample_id','labels','group_id']).to_pylist()
        visible=pq.read_table(e.path(meta['inputs']['file']),columns=['sample_id','partition','text_hash'],filters=[('partition','=','test')]).to_pylist()
        source_cache[ds]=({r['sample_id']:(tuple(r['labels']),r['group_id']) for r in source},
                          {r['sample_id']:r['text_hash'] for r in visible})
        if len(source_cache[ds][0])!=len(source) or len(source_cache[ds][1])!=len(source):
            raise ValueError('Frozen Q source/input IDs are duplicated or incomplete')
    source,visible=source_cache[ds]
    if {r.sample_id:(r.labels,r.group_id) for r in rows}!=source: raise ValueError('Prediction gold/group IDs differ from frozen Q')
    ordered=sorted(rows,key=lambda r:r.sample_id)
    request=canonical_hash({'run':model_request,'evaluation_freeze_hash':lock['lock_hash'],
        'rows':[(r.sample_id,r.group_id,r.labels,visible[r.sample_id]) for r in ordered]})
    if request!=info.get('request_hash'): raise ValueError('Saved prediction request identity changed')
    for row in rows:
        scores=np.asarray(row.scores)
        decision=(int(scores.argmax()),) if meta['task']=='single_label' else tuple(int(i) for i in np.flatnonzero(scores>=(.5 if row.score_kind=='probability' else 0)))
        if row.prediction!=decision: raise ValueError('Saved decisions disagree with scores/declared operational threshold')
    metrics=_metrics(rows,meta)
    if canonical_hash(metrics)!=info.get('metrics_hash') or canonical_hash(metrics)!=canonical_hash(info.get('metrics')):
        raise ValueError('Saved metrics do not reconstruct from full PredictionRow scores')
    return {'run_id':rid,'dataset_id':ds,'model_id':run.get('model_id',model_request.get('model_id')),
        'condition_id':run.get('condition_id',model_request.get('condition_id')),
        'metrics':metrics,'test_exposure_status':test_exposure_status(ds),'source':f'predictions/{rid}.parquet',
        'checkpoint_to_scores_seconds':info.get('checkpoint_to_scores_seconds')},rows


def _audit(e,index,prepared,predictions,lock):
    from .audit import validate_reviews
    from collections import defaultdict
    refs=index['audit'];frame=read_json(e.ref(refs['frame']));reviews=read_json(e.ref(refs['reviews']))
    e.ref(refs['guide']);frozen=read_json(e.ref(refs['guide_freeze']))
    if (frozen.get('guide_sha256')!=refs['guide']['sha256'] or frame.get('guide_freeze_sha256')!=refs['guide_freeze']['sha256']
            or datetime.fromisoformat(frozen['frozen_at_utc'])>=datetime.fromisoformat(frame['created_at_utc'])
            or frame.get('evaluation_freeze_hash')!=lock['lock_hash']):
        raise ValueError('Annotation guide/frame freeze identity or chronology is invalid')
    if reviews.get('phase')!='blind_first_pass':raise PermissionError('Original blind first-pass human review evidence is pending')
    records=frame['records'];people=reviews['records'];datasets=('yelp','go_emotions','banking77','yahoo_answers_topics')
    adjudications=read_json(e.ref(refs['adjudications']))['records'] if refs.get('adjudications') else []
    adjudicated={(r['dataset_id'],r['sample_id']):r for r in adjudications};used_adjudications=set()
    if len(adjudicated)!=len(adjudications):raise ValueError('Duplicate human adjudication records')
    frames={};summaries=[];reviewed_ids=set()
    for ds in datasets:
        rows=predictions[ds]
        source=rows if isinstance(rows,dict) else {r.sample_id:(r.labels,r.group_id) for r in rows}
        groups=defaultdict(list)
        for sid,(_,group) in source.items():groups[group].append(sid)
        group_ids=sorted(groups);take=min(200,len(group_ids));rng=np.random.default_rng(frame['seeds'][ds])
        expected=[]
        for i in rng.choice(len(group_ids),size=take,replace=False):
            group=group_ids[int(i)];candidates=sorted(groups[group]);sid=candidates[int(rng.integers(len(candidates)))]
            probability=(take/len(group_ids))/len(candidates)
            expected.append({'sample_id':sid,'group_id':group,'labels':list(source[sid][0]),
                'inclusion_probability':probability,'analysis_weight':1/probability,
                'group_population_size':len(candidates),'sampling_frame_groups':len(group_ids),'sampled_groups':take})
        actual=[r for r in records if r['dataset_id']==ds]
        if len(actual)!=len(expected): raise PermissionError('All declared A audit cases are required (200 per available frame)')
        lookup={r['sample_id']:r for r in expected}
        if len({r['sample_id'] for r in actual})!=len(actual) or {r['sample_id'] for r in actual}!=set(lookup):
            raise ValueError('A sample frame is not the fixed random group/row sample')
        for row in actual:
            want=lookup[row['sample_id']]
            for key in ('group_id','labels','inclusion_probability','analysis_weight','group_population_size','sampling_frame_groups','sampled_groups'):
                if row[key]!=want[key]: raise ValueError('A sampling identity/probability changed')
            if any(key in row for key in ('scores','prediction','model_id','model_outputs')):
                raise ValueError('First-pass A packet contains revealed model outputs')
            frames[(ds,row['sample_id'])]=row
        ds_reviews=[r for r in people if r['dataset_id']==ds]
        summary=validate_reviews(tuple(ds_reviews))
        if not summary['double_review_complete'] or summary['samples']!=len(actual):
            raise PermissionError('Every A case requires two distinct preserved human review records')
        for review in ds_reviews:
            key=(ds,review['sample_id'])
            if key not in frames or review['group_id']!=frames[key]['group_id'] or review.get('partition')!='audit':
                raise ValueError('Review does not match the blinded original A frame')
            if review.get('gold_status')=='not_checked':raise PermissionError('Named review receipts still have unchecked gold labels')
            if review.get('gold_status') not in {'supported','ambiguous','insufficient_information','suspected_error'}:
                raise ValueError('Human gold judgment is outside the actual annotation schema')
            if any(review.get(k) is not None for k in ('checkpoint_id','condition_id','anonymous_model_id','model_prediction','score_kind')):
                raise ValueError('Original first-pass human annotations must remain blinded to model outputs/identity')
            reviewed_ids.add(f'{ds}:{review["sample_id"]}')
        for disagreement in summary['samples_summary']:
            if not disagreement['disagreement']:continue
            key=(ds,disagreement['sample_id']);decision=adjudicated.get(key)
            if decision is None:raise PermissionError('Human disagreement requires an authenticated adjudication')
            originals=[r for r in ds_reviews if r['sample_id']==key[1]]
            if (decision.get('group_id')!=frames[key]['group_id']
                    or sorted(decision.get('review_record_hashes',[]))!=sorted(canonical_hash(r) for r in originals)
                    or not isinstance(decision.get('adjudicator_id'),str) or not decision['adjudicator_id'].strip()
                    or not isinstance(decision.get('rationale'),str) or not decision['rationale'].strip()
                    or not decision.get('gold_status') or not isinstance(decision.get('phenomena'),list)):
                raise ValueError('Human adjudication must bind both preserved original review records and decision')
            used_adjudications.add(key)
        summaries.append({'dataset_id':ds,'samples':len(actual),'independent_review_records':len(ds_reviews)})
    if len(records)!=sum(r['samples'] for r in summaries): raise ValueError('Audit frame includes undeclared/history cases')
    if used_adjudications!=set(adjudicated):raise ValueError('Extra adjudications do not match actual review disagreements')
    return {'datasets':summaries,'reviewed_sample_ids':sorted(reviewed_ids),
            'adjudicated_cases':len(used_adjudications),'cause_status':'unresolved'}


def _prediction_stream(e,refs,run_ids):
    """Read only requested authenticated files, one Arrow batch at a time."""
    from .types import PredictionRow
    import pyarrow.parquet as pq
    for run in sorted(run_ids):
        if run not in refs:raise PermissionError(f'Authenticated prediction evidence is pending: {run}')
        path=e.ref(refs[run])
        for batch in pq.ParquetFile(path).iter_batches(batch_size=4096):
            for record in batch.to_pylist():
                if record['run_id']!=run:raise ValueError('Prediction reference contains a different run identity')
                yield PredictionRow(record['run_id'],record['sample_id'],record['group_id'],tuple(record['labels']),
                    tuple(record['prediction']),tuple(record['scores']),record['score_kind'],record['input_hash'])


def _family_run_ids(manifest,family):
    estimate_ids={part['estimate_id'] for contrast in manifest['contrasts'] if contrast['family']==family for part in contrast['components']}
    return {run for estimate in manifest['estimates'] if estimate['estimate_id'] in estimate_ids
            for draw in estimate['draws'] for arm in ('left_runs','right_runs') for run in draw[arm]}


def _confirmation_sufficiency(receipt):
    if (receipt.get('underpowered') is not False or receipt.get('mechanism_evidence_sufficient') is not True
            or receipt.get('n_actual',0)<receipt.get('n_planned',1)):
        raise PermissionError('Authenticated confirmation remains underpowered; mechanism evidence sufficiency is pending')
    return {'n_actual':receipt['n_actual'],'n_planned':receipt['n_planned'],'sufficient':True}


def _confirmation_evidence(e,index,node,prediction_refs):
    from .confirmation import verify_confirmation,verify_confirmation_audit_frame
    frame=read_json(e.ref(index['audit']['frame']))
    binding=verify_confirmation_audit_frame(e.root,node['pilot_id'],frame)
    verified=verify_confirmation(e.root,node['pilot_id'])
    receipt=e.read(f'confirmation/{node["pilot_id"]}/result-manifest.json')
    if canonical_hash(receipt)!=canonical_hash(verified):raise ValueError('Confirmation authentication/result bytes disagree')
    for ref in binding['confirmation_audit_frames'].values():e.path(ref['file'],ref['sha256'])
    for view,ref in receipt['views'].items():
        e.path(ref['file'],ref['sha256'])
        prediction_refs[view]={'path':ref['file'],'sha256':ref['sha256']}
    for path in (e.root/'confirmation'/node['pilot_id']).iterdir():
        if path.is_file():e.path(path.relative_to(e.root).as_posix())
    return {**verified,'global_audit_binding':binding}


def _completion(e,training_complete,scope):
    for gate in e.gates.values():gate['complete']=bool(gate['checks']) and not gate['failures']
    full_complete=bool(training_complete and all(g['complete'] for g in e.gates.values()))
    return (bool(training_complete) if scope=='training' else full_complete),full_complete


def _statistics(e,index,manifest,prediction_refs):
    from .statistics import bootstrap_family
    families={c['family'] for c in manifest['contrasts']}
    required={'F1','F2','F3','F4','F5','F6','AUX_M2','AUX_M3','AUX_M5'}
    if not required<=families: raise PermissionError('Actual contrast manifest lacks required F1--F6/M2/M3/M5 families')
    refs=index['statistics'];lookup={r['family']:r for r in refs}
    if len(lookup)!=len(refs) or set(lookup)!=families: raise PermissionError('Statistics receipts must cover every actual declared family exactly once')
    results=[]
    for family in sorted(families):
        ref=lookup[family]
        if ref['contrast_file_sha256']!=hash_file(e.path('contrasts.json')): raise ValueError('Statistics contrast manifest bytes changed')
        saved=read_json(e.ref(ref));selected={**manifest,'family':family}
        computed=bootstrap_family(_prediction_stream(e,prediction_refs,_family_run_ids(manifest,family)),
                                  selected,replicates=2000,seed=7301)
        if canonical_hash(saved)!=canonical_hash(computed) or computed['status']!='ok':
            raise ValueError('Statistics do not reconstruct from saved predictions and actual contrasts')
        results.append({'family':family,'role':'auxiliary_mechanism' if family.startswith('AUX_') else 'predeclared_primary_family',
                        'critical':computed['critical'],'contrasts':computed['contrasts']})
    return results


def _trajectory_catalogue(design,manifest):
    """Derive representative coverage from frozen logical contexts, including aliases."""
    from collections import defaultdict
    conditions={c['condition_id']:c for c in design['conditions']}
    physical={r['run_id']:r for r in design['physical_runs']}
    entries={r.get('planned_run_id',r['run_id']):r for r in manifest['runs']}
    if len(entries)!=len(manifest['runs']):raise ValueError('Duplicate resolved trajectory run identities')
    grouped=defaultdict(list)
    for logical in design['logical_runs']:
        experiment=logical['experiment_id']
        if experiment not in {'E1','E2','E3','E4','E5'}:continue
        planned=logical['run_id']
        if planned not in entries or planned not in physical:raise PermissionError('Resolved trajectory run coverage is pending')
        entry=entries[planned];condition=conditions[logical['condition_id']]
        ds=logical['dataset_id'];model=logical['model_id'];context=logical.get('context',{});meta=condition['metadata']
        if (entry['dataset_id']!=ds or condition['dataset_id']!=ds or entry['model_id']!=model
                or entry['training_seed']!=physical[planned]['training_seed']):
            raise ValueError('Trajectory logical/physical dataset/model/seed identity differs')
        actual_condition=conditions[entry['condition_id']]
        if actual_condition['dataset_id']!=ds or actual_condition['data_seed']!=entry['data_seed']:
            raise ValueError('Trajectory resolved condition/data seed differs')
        level=(meta['shots_per_class'] if experiment=='E2' else context.get('ratio',meta.get('requested_ratio')) if experiment=='E3'
               else context.get('alpha',meta.get('alpha')) if experiment=='E4' else meta['size'])
        if level is None:raise ValueError('Trajectory logical condition level is missing')
        grouped[(ds,experiment,model)].append({'run_id':entry.get('artifact_id',entry['run_id']),
            'condition_id':entry['condition_id'],'logical_condition_id':logical['condition_id'],
            'data_seed':entry['data_seed'],'training_seed':entry['training_seed'],'level':level,
            'head_seed':context.get('head_seed')})
    if not grouped:raise PermissionError('Design has no authenticated E1--E5 trajectory coverage')
    for exception in design.get('infeasible_conditions',[]):
        ds=exception['dataset_id'];experiment=exception['experiment_id']
        if experiment not in {'E1','E2','E3','E4','E5'}:continue
        models=({'modernbert','distilbert'} if experiment=='E5' else
                {model for d,ex,model in grouped if d==ds and ex=='E1'})
        for model in models:grouped.setdefault((ds,experiment,model),[])
    catalogue={}
    for key,rows in sorted(grouped.items()):
        exceptions=[r for r in design.get('infeasible_conditions',[]) if r['dataset_id']==key[0] and r['experiment_id']==key[1]]
        if not rows:
            catalogue[key]={'dataset_id':key[0],'experiment_id':key[1],'model_id':key[2],
                'data_seed':None,'training_seed':None,'head_seed':None,'nodes':[],'required':False,
                'reason':'all_conditions_declared_infeasible','design_infeasible_conditions':exceptions}
            continue
        nonfull={r['data_seed'] for r in rows if r['data_seed']!=0}
        data_seed=2026 if 2026 in nonfull else min(nonfull) if nonfull else 0
        training={r['training_seed'] for r in rows}
        train_seed=42 if 42 in training else None if None in training else min(training)
        heads={r['head_seed'] for r in rows if r['head_seed'] is not None}
        head=9101 if 9101 in heads else min(heads) if heads else None
        selected=[r for r in rows if r['data_seed'] in {0,data_seed} and r['training_seed']==train_seed
                  and (key[1]!='E3' or r['head_seed'] is None or r['head_seed']==head)]
        selected.sort(key=lambda r:(float('inf') if r['level']=='full' else float(r['level']),r['run_id'],r['condition_id']))
        unique=[];seen_runs=set();seen_conditions=set()
        for row in selected:
            if row['run_id'] in seen_runs or row['condition_id'] in seen_conditions:continue
            seen_runs.add(row['run_id']);seen_conditions.add(row['condition_id']);unique.append(row)
        catalogue[key]={'dataset_id':key[0],'experiment_id':key[1],'model_id':key[2],'data_seed':data_seed,
            'training_seed':train_seed,'head_seed':head,'nodes':unique,'required':len(unique)>=2,
            'reason':None if len(unique)>=2 else 'fewer_than_two_distinct_feasible_physical_conditions',
            'design_infeasible_conditions':exceptions}
    return catalogue


def _trajectory_summaries(e,index,prepared,prediction_refs):
    from .trajectories import trace_failures
    refs=index['trajectories'];declared=read_json(e.ref(refs['paths']))
    catalogue=_trajectory_catalogue(e.read('matrix/design_manifest.json'),e.read('run_manifest.json'))
    if not declared.get('datasets'):raise PermissionError('Representative trajectory coverage cannot be empty')
    results={(r['dataset_id'],r['path_id']):r for r in refs['results']}
    requested={(ds,path_id) for ds,spec in declared['datasets'].items() for path_id in spec['paths']}
    if len(results)!=len(refs['results']) or set(results)!=requested:
        raise PermissionError('Every declared trajectory path needs its own byte-bound result')
    summaries={};covered=set()
    for ds,spec in declared['datasets'].items():
        for path_id,path in spec['paths'].items():
            key=(ds,path.get('experiment_id'),path['model_id'])
            if key not in catalogue or not catalogue[key]['required']:raise ValueError('Trajectory has no applicable representative logical path')
            expected=catalogue[key]
            if path['data_seed']!=expected['data_seed'] or path['training_seed']!=expected['training_seed']:
                raise ValueError('Trajectory is not the declared representative data/training seed')
            nodes={(n['run_id'],n['condition_id']):(i,n) for i,n in enumerate(expected['nodes'])};order=[]
            for step in path['steps']:
                identity=(step['run_id'],step['condition_id'])
                if identity not in nodes:raise ValueError('Trajectory run/condition/model differs from actual logical coverage')
                position,node=nodes[identity];order.append(position)
                if (step.get('data_seed',path['data_seed'])!=node['data_seed']
                        or step.get('training_seed',path['training_seed'])!=node['training_seed']
                        or step.get('model_id',path['model_id'])!=key[2]
                        or step.get('logical_condition_id',node['logical_condition_id'])!=node['logical_condition_id']):
                    raise ValueError('Trajectory step seed/model/condition is not the actual resolved run')
            if len(order)<2 or order!=sorted(set(order)):raise ValueError('Trajectory conditions must have genuine increasing declared level order')
            covered.add(key)
    missing={key for key,value in catalogue.items() if value['required']}-covered
    if missing:raise PermissionError(f'Representative trajectory coverage is incomplete: {sorted(missing)}')
    for ds,spec in declared['datasets'].items():
        meta=prepared['datasets'][ds];summaries[ds]={}
        for path_id,path in spec['paths'].items():
            run_ids={step['run_id'] for step in path['steps']}
            saved=read_json(e.ref(results[(ds,path_id)]))
            computed=trace_failures(_prediction_stream(e,prediction_refs,run_ids),{path_id:path},meta['task'],len(meta['label_names']))
            if canonical_hash(computed)!=canonical_hash(saved):raise ValueError('Failure trajectory does not reconstruct from saved predictions')
            value=computed['paths'][path_id]
            summaries[ds][path_id]={'counts':value['counts'],'sample_count':value['sample_count'],
                                   'source':results[(ds,path_id)]['path']}
            del saved,computed,value
    summaries['_coverage_ledger']=list(catalogue.values())
    return summaries


def _related_work(e,index):
    path=e.ref(index['related_work'])
    with path.open(encoding='utf-8-sig',newline='') as stream:
        reader=csv.DictReader(stream);rows=list(reader)
        if not {'citation','url','relevance','difference'}<=set(reader.fieldnames or []):
            raise ValueError('Related-work matrix requires citation,url,relevance,difference')
    if len(rows)<5 or any(not all(row.get(k,'').strip() for k in ('citation','url','relevance','difference')) for row in rows):
        raise PermissionError('Related-work comparison is incomplete')
    return rows


def _verify_interventional_claim(e,index,claim,manifest,statistics,prediction_refs,audit,confirmations):
    """Bind reviewed interpretation to one exact reconstructed effect and case set."""
    if audit is None or not manifest or not statistics:raise PermissionError('Reviewed claim lacks actual audit/contrast/statistical evidence')
    cid=claim.get('contrast_id');matches=[c for c in manifest['contrasts'] if c['contrast_id']==cid]
    if len(matches)!=1:raise PermissionError('Supported claim needs one actual declared intervention contrast')
    contrast=matches[0];family=contrast['family']
    allowed_scope='local_paired_intervention' if family=='F6' else 'specified_training_comparison' if family in {'AUX_M2','AUX_M3','AUX_M5'} else None
    if allowed_scope is None or claim.get('scope')!=allowed_scope:raise PermissionError('Claim scope exceeds its actual intervention contrast')
    estimates={r['estimate_id']:r for r in manifest['estimates']}
    run_ids={run for part in contrast['components'] for draw in estimates[part['estimate_id']]['draws']
             for arm in ('left_runs','right_runs') for run in draw[arm]}
    datasets={manifest['runs'][run]['dataset_id'] for run in run_ids}
    ds=claim.get('dataset_id')
    if datasets!={ds} or contrast.get('dataset_id',ds)!=ds:raise ValueError('Claim dataset is not the exact contrast dataset')
    actual={}
    for run in sorted(run_ids):
        current={}
        for row in _prediction_stream(e,prediction_refs,{run}):
            if row.sample_id in current:raise ValueError('Duplicate actual claim case identity')
            current[row.sample_id]=(row.group_id,row.labels)
        if not current:raise PermissionError('Actual contrast case predictions are pending')
        if actual and actual!=current:raise ValueError('Claim contrast arms do not share exact base case/gold/group identities')
        actual=current
    cases=read_json(e.ref(claim['cases']))
    expected=[{'sample_id':sid,'group_id':actual[sid][0]} for sid in sorted(actual)]
    declared=cases.get('cases',[])
    if (cases.get('dataset_id')!=ds or len(declared)!=len(expected)
            or sorted(declared,key=lambda r:(r['sample_id'],r['group_id']))!=expected):
        raise ValueError('Claim cases must equal the actual contrast dataset/base identities')
    summaries=[r for r in statistics if r['family']==family]
    if len(summaries)!=1 or cid not in summaries[0]['contrasts']:raise PermissionError('Reconstructed statistical effect for claim is pending')
    effect=summaries[0]['contrasts'][cid]
    refs=[r for r in index['statistics'] if r['family']==family]
    if len(refs)!=1:raise PermissionError('Claim needs the exact statistical result receipt')
    ref=refs[0];saved=read_json(e.ref(ref))
    contrast_sha=hash_file(e.path('contrasts.json'))
    if (ref.get('contrast_file_sha256')!=contrast_sha or canonical_hash(manifest)!=canonical_hash(e.read('contrasts.json'))
            or canonical_hash(saved['contrasts'][cid])!=canonical_hash(effect)):
        raise ValueError('Claim statistical/contrast evidence differs from the reconstructed effect')
    point,lower,upper=np.asarray([effect.get('estimate'),effect.get('simultaneous_lower'),effect.get('simultaneous_upper')],dtype=float)
    direction=claim.get('direction')
    if (not np.isfinite([point,lower,upper]).all() or lower>upper
            or direction not in {'positive','negative'}
            or direction=='positive' and not(point>0 and lower>0)
            or direction=='negative' and not(point<0 and upper<0)):
        raise PermissionError('Claim direction is not supported by the reconstructed simultaneous interval')
    if family=='F6':
        receipts=[r for r in confirmations if r['pilot_id']==contrast.get('pilot_id')]
        if len(receipts)!=1:raise PermissionError('Exact claim intervention confirmation is pending')
        receipt=receipts[0];_confirmation_sufficiency(receipt)
        if set(receipt['views'])!=run_ids or any(r['dataset_id']!=ds for r in receipt['views'].values()):
            raise ValueError('Claim borrows cases from a different confirmation intervention/dataset')
    case_hash=canonical_hash({'dataset_id':ds,'cases':expected})
    binding={'contrast_file_sha256':contrast_sha,'statistics_file_sha256':ref['sha256'],'contrast':effect,
        'dataset_id':ds,'contrast_id':cid,'direction':direction,'scope':allowed_scope,
        'cases_file_sha256':claim['cases']['sha256'],'case_set_hash':case_hash,
        'prediction_files':{r:prediction_refs[r]['sha256'] for r in sorted(run_ids)}}
    if not index.get('claim_reviews'):raise PermissionError('Two actual human claim-interpretation review records are pending')
    records=read_json(e.ref(index['claim_reviews']))['records']
    reviews=[r for r in records if r.get('claim_id')==claim['claim_id']]
    people=[r.get('reviewer_id') for r in reviews]
    if len(people)<2 or any(not isinstance(p,str) or not p.strip() for p in people) or len(set(people))!=len(people):
        raise PermissionError('Supported claim requires two distinct preserved human interpretation reviewers')
    required={'claim_content_hash':canonical_hash(claim),'evidence_hash':canonical_hash(binding),'dataset_id':ds,
        'contrast_id':cid,'direction':direction,'scope':allowed_scope,'case_set_hash':case_hash,'case_count':len(expected),'decision':'supported'}
    if any(any(r.get(k)!=v for k,v in required.items()) or not isinstance(r.get('rationale'),str) or not r['rationale'].strip() for r in reviews):
        raise ValueError('Human claim review must bind the exact text/evidence/dataset/base cases/direction/scope')
    return {**claim,'interpretation_reviewers':people,'verified_evidence_hash':canonical_hash(binding)}


def _claims(e,index,manifest,statistics,prediction_refs,audit,confirmations):
    payload=read_json(e.ref(index['claims']));verified=[];seen=set()
    for claim in payload['claims']:
        if (claim.get('kind') not in {'observational','associational','interventional'}
                or not claim.get('claim_id') or not claim.get('text') or claim['claim_id'] in seen):
            raise ValueError('Claim needs a unique declared identity, scope and text')
        seen.add(claim['claim_id'])
        for ref in claim.get('evidence',[]):e.ref(ref)
        if claim.get('status')=='supported' and claim['kind']=='interventional':
            checked=e.check('G5','reviewed_claim:'+claim['claim_id'],index['claims']['path'],
                lambda claim=claim:_verify_interventional_claim(e,index,claim,manifest,statistics,prediction_refs,audit,confirmations))
            verified.append(checked if checked is not None else {**claim,'status':'unresolved','claimed_status':'supported',
                'verification_status':'pending_or_invalid_interpretation_evidence'})
        else:
            if claim.get('status')=='supported' and not claim.get('evidence'):
                raise ValueError('Supported descriptive claim has no authenticated evidence')
            verified.append(claim)
    return verified


def verify_research(root: Path,required_scope: str) -> dict[str,Any]:
    """Verify saved evidence; missing evidence is pending, never mean-filled."""
    if required_scope not in {'training','full_research'}: raise ValueError('Unknown research verification scope')
    from .freeze import verify_freeze,_checkpoint,assert_resolved_coverage,verify_anchor_choice,_verify_pilot
    from .matrix import verify_design_lock
    root=Path(root).resolve();e=_Evidence(root);context={};results=[];prediction_refs={};by_dataset={};statistics=[];audit=None;claims=[]
    related_work=[];trajectory_summary={};confirmation_summary=[]
    def load(name,path,default):
        value=e.check('G2' if name=='design' else 'G1',name,path,lambda:e.read(path))
        context[name]=value or default
    load('design','matrix/design_manifest.json',{})
    load('prepared','data/prepared_manifest.json',{'datasets':{}})
    design=context['design'];prepared=context['prepared']
    def design_check():
        lock=verify_design_lock(root);e.path('design.lock.json');e.bind_records(lock['files']);return lock
    dlock=e.check('G2','design_lock','design.lock.json',design_check)
    e.check('G1','prepared_bytes','data/prepared_manifest.json',lambda:_prepared(e,dlock))
    e.check('G0','environment_and_forward','environment.json',lambda:_runtime(e))
    e.check('G0','real_E0_15','training/engineering-smoke.json',lambda:_engineering(e,prepared,design))
    selections=e.check('G2','selection_records','selections.json',lambda:e.read('selections.json')) or {}
    fixed=selections.get('selection_policy')=='fixed_defaults'
    anchors={'verified_choices':0,'required_choices':len(design.get('selection_nodes',[])),
             'verified_candidates':0,'required_candidates':0 if fixed else 6*len(design.get('selection_nodes',[])),
             'mode':'fixed_defaults' if fixed else 'grid_search',
             'baseline_checkpoints_available':sum(c.get('checkpoint_status')=='complete' for c in selections.get('choices',{}).values())}
    if not design.get('selection_nodes'):
        e.check('G2','anchor_coverage','matrix/design_manifest.json',lambda:(_ for _ in ()).throw(PermissionError('All declared anchor search nodes are required')))
    for node in design.get('selection_nodes',[]):
        def anchor(node=node):
            choice=selections.get('choices',{}).get(node['node_id'])
            if choice is None:raise PermissionError(f'Anchor search remains pending: {node["node_id"]}')
            files={};record=verify_anchor_choice(root,node,choice,files);e.bind_records(files);return record
        if e.check('G2','anchor:'+node['node_id'],'selections.json',anchor) is not None:
            anchors['verified_choices']+=1;anchors['verified_candidates']+=0 if fixed else 6
    pilots=design.get('discovery_pilot_nodes',[])
    if len(pilots)!=3:
        e.check('G2','three_D_pilots','selections.json',lambda:(_ for _ in ()).throw(PermissionError('All three genuine50-case D pilots are required')))
    for node in pilots:
        files={}
        e.check('G2','pilot:'+node['pilot_id'],f'pilots/{node["pilot_id"]}/results.json',
            lambda node=node,files=files:(_verify_pilot(root,node,selections['discovery_pilots'][node['pilot_id']],selections['choices'],files),e.bind_records(files)))
    def freeze(stage):
        lock=verify_freeze(stage,root);e.path('selection.lock.json' if stage=='selection' else 'evaluation.freeze.json');e.bind_records(lock['files']);return lock
    selection=e.check('G2','global_selection','selection.lock.json',lambda:freeze('selection'))
    evaluation=e.check('G3','global_evaluation','evaluation.freeze.json',lambda:freeze('evaluation'))
    manifest=e.check('G3','run_manifest','run_manifest.json',lambda:e.read('run_manifest.json')) or {}
    runs=e.check('G3','all_planned_training_coverage','run_manifest.json',lambda:assert_resolved_coverage(design,manifest))
    required_runs=len(design.get('physical_runs',[]));verified_runs=0;costs=[]
    if not required_runs:
        e.check('G3','nonempty_training_scope','matrix/design_manifest.json',lambda:(_ for _ in ()).throw(PermissionError('Required physical fit scope is empty or missing')))
    for run in runs or []:
        rid=run.get('artifact_id',run['run_id'])
        def checkpoint(run=run,rid=rid):
            files={};status=_checkpoint(root,rid,files,run.get('expected_request_hash'),run.get('expected_model_request_hash'));e.bind_records(files)
            if not np.isfinite(status.get('reload_max_absolute_difference',np.nan)):
                raise ValueError('Numerical saved reload verification is missing')
            return status
        status=e.check('G3','checkpoint:'+rid,f'runs/{rid}/status.json',checkpoint)
        if status is not None:
            verified_runs+=1
            cost={'run_id':rid,**{k:status.get(k) for k in ('fit_seconds','feature_seconds','total_seconds','training_unique_rows','micro_batch','effective_batch')}}
            if run.get('model_id')=='modernbert':
                def token_budget(run=run):
                    from .freeze import verify_token_budget
                    files={};value=verify_token_budget(root,run,files);e.bind_records(files);return value
                cost['own_token_budget']=e.check('G3','modern_token_budget:'+rid,f'training/token-budgets/{rid}.json',token_budget)
            costs.append(cost)
    # Preserve training result before prediction-specific checks are added to G3.
    training_complete=all(not e.gates[g]['failures'] and e.gates[g]['checks'] for g in ('G0','G1','G2','G3'))
    prediction_access='not_requested_for_training_scope' if required_scope=='training' else 'blocked_until_evaluation_freeze'
    if evaluation is not None and required_scope=='full_research':
        prediction_access='verified_saved_artifacts_only';source_cache={}
        for run in runs or []:
            rid=run.get('artifact_id',run['run_id'])
            value=e.check('G3','predictions:'+rid,f'predictions/{rid}.parquet',
                lambda run=run:_prediction(e,run,prepared['datasets'][run['dataset_id']],evaluation,source_cache))
            if value:
                summary,rows=value;results.append(summary)
                prediction_refs[rid]={'path':summary['source'],'sha256':e.files[summary['source']]['sha256']}
                by_dataset.setdefault(run['dataset_id'],source_cache[run['dataset_id']][0])
                del rows,value
    index=e.check('G4','evidence_routing_index','analysis/research-evidence.json',lambda:e.read('analysis/research-evidence.json')) if required_scope=='full_research' else None
    if required_scope=='full_research':
        if evaluation is None or index is None:
            for gate,name,path in [('G4','real_A_double_review','audit/reviews.json'),('G4','paired_confirmation','confirmation/'),
                                   ('G4','all_statistical_families','analysis/statistics/'),('G5','report_evidence','analysis/research-evidence.json')]:
                e.check(gate,name,path,lambda:(_ for _ in ()).throw(PermissionError('Authenticated evaluation freeze and byte-bound evidence index are prerequisites')))
        else:
            audit=e.check('G4','real_A_double_review',index.get('audit',{}).get('reviews',{}).get('path','audit/reviews.json'),
                lambda:_audit(e,index,prepared,by_dataset,evaluation))
            for node in pilots:
                checked=e.check('G4','confirmation:'+node['pilot_id'],f'confirmation/{node["pilot_id"]}/result-manifest.json',
                    lambda node=node:_confirmation_evidence(e,index,node,prediction_refs))
                if checked is not None:
                    confirmation_summary.append({'pilot_id':node['pilot_id'],**checked})
                    e.check('G4','confirmation_sufficiency:'+node['pilot_id'],f'confirmation/{node["pilot_id"]}/result-manifest.json',
                        lambda checked=checked:_confirmation_sufficiency(checked))
            contrasts=e.check('G4','actual_contrast_manifest','contrasts.json',lambda:e.read('contrasts.json'))
            if contrasts:
                statistics=e.check('G4','all_statistical_families','analysis/statistics/',lambda:_statistics(e,index,contrasts,prediction_refs)) or []
            trajectory_summary=e.check('G5','failure_trajectories','analysis/trajectories.json',
                lambda:_trajectory_summaries(e,index,prepared,prediction_refs))
            related_work=e.check('G5','related_work',index.get('related_work',{}).get('path','analysis/related-work.csv'),lambda:_related_work(e,index)) or []
            claims=e.check('G5','evidence_linked_claims','analysis/claims.json',
                lambda:_claims(e,index,contrasts,statistics,prediction_refs,audit,confirmation_summary)) or []
    complete,full_complete=_completion(e,training_complete,required_scope)
    return {'schema_version':1,'scope':required_scope,'complete':bool(complete),'training_complete':bool(training_complete),
        'achieved_scope':'full_research' if full_complete else ('training' if training_complete else 'partial_preparation_or_training'),
        'recorded_at_utc':datetime.now(timezone.utc).isoformat(),'gates':e.gates,'missing':e.missing,
        'evidence':list(e.files.values()),'anchors':anchors,
        'training':{'verified_physical_runs':verified_runs,'required_physical_runs':required_runs,'costs':costs},
        'prediction_access':prediction_access,'results':results,'statistics':statistics,'audit':audit,'claims':claims,
        'failure_trajectories':trajectory_summary or {},'related_work':related_work,'confirmation':confirmation_summary,
        'archive_inventory':[record for record in e.files.values() if not record['path'].startswith(('data/','inputs/','assets/'))],
        'archive_exclusions':['raw corpora/visible texts, including Yahoo with unknown redistribution permission','unrelated caches'],
        'datasets':{ds:{'task':meta['task'],'label_count':len(meta['label_names']),
            'partitions':{p:{k:r.get(k) for k in ('rows','label_support','ap_mask')} for p,r in meta.get('partitions',{}).items()}}
            for ds,meta in prepared['datasets'].items()},
        'cause_status':'unresolved','cross_task_metric_average':None,
        'limitations':['Conditional test-sampling intervals describe saved fitted models; few seeds limit generalization.',
            'Task type is confounded with corpus/style/annotation; external pretraining exposure is not fully auditable.',
            'Historical Q remains previously seen exploratory evidence; automated records do not replace genuine human review.',
            'Unsupported explanations remain unresolved; auxiliary mechanism results do not replace primary families.']}


def _render(result,language):
    zh=language=='zh';lines=['# 文本分类研究证据报告' if zh else '# Text classification research evidence report',
        '',(('状态：完成。' if result['complete'] else '状态：未完成；缺失门和证据保留为待完成。') if zh else 'Status: complete.' if result['complete'] else 'Status: pending; missing gates and evidence remain incomplete.'),
        f"Scope: {result['achieved_scope']}",'',
        '原因状态：unresolved。软件完成不等于真实研究完成。' if zh else 'Cause status: unresolved. Software readiness does not establish completed research.',
        '', '不跨任务平均F1/AP；各数据集分别报告。' if zh else 'No cross-task metric average; each dataset retains its own endpoint.',
        '', '| Gate | Status |', '| --- | --- |']
    lines.extend(f"| {name} | {'verified' if gate['complete'] else 'pending / invalid'} |" for name,gate in result['gates'].items())
    anchor=result['anchors']
    configuration_line=(f"固定训练配置：{anchor['verified_choices']}/{anchor['required_choices']}；已具备的基线检查点：{anchor.get('baseline_checkpoints_available',0)}。" if zh else
        f"Fixed configurations: {anchor['verified_choices']}/{anchor['required_choices']}; available baseline checkpoints: {anchor.get('baseline_checkpoints_available',0)}.") if anchor.get('mode')=='fixed_defaults' else f"Anchors: {anchor['verified_choices']}/{anchor['required_choices']}; candidates: {anchor['verified_candidates']}/{anchor['required_candidates']}."
    lines.extend(['',configuration_line,
        (f"主实验运行清单的检查点核验：{result['training']['verified_physical_runs']}/{result['training']['required_physical_runs']}。" if zh else f"Required-run manifest checkpoint verification: {result['training']['verified_physical_runs']}/{result['training']['required_physical_runs']}."),
        '', '## 数据与支持数' if zh else '## Dataset sizes and support'])
    for ds,meta in sorted(result['datasets'].items()):
        lines.extend(['',f"### {ds} ({meta['task']})",'```json',json.dumps(meta,ensure_ascii=False,indent=2),'```'])
    lines.extend(['','## 经重建的模型结果' if zh else '## Reconstructed model results',
        '| Dataset | Model | Condition | Endpoint | Value | Exposure |','| --- | --- | --- | --- | --- | --- |'])
    for row in result['results']:
        metric='macro_ap' if row['metrics']['task'] in {'multilabel','multi_label'} else 'macro_f1'
        lines.append(f"| {row['dataset_id']} | {row['model_id']} | {row['condition_id']} | {metric} | {row['metrics'][metric]} | {row['test_exposure_status']} |")
    if not result['results']:lines.append('| pending | — | — | — | — | — |')
    lines.extend(['','各run的完整类别/标签支持数、F1/AP与混淆矩阵保存在completion.json的results中。' if zh else
                  'Full per-run class/label support, F1/AP and confusion matrices are preserved in completion.json results.'])
    lines.extend(['','## 统计与失败/机制证据' if zh else '## Statistics and failure/mechanism evidence'])
    for family in result['statistics']:
        lines.extend(['',f"### {family['family']} ({family['role']})",'```json',json.dumps(family,ensure_ascii=False,indent=2),'```'])
    if not result['statistics']:lines.append('pending；没有经过预测重建的统计结论。' if zh else 'pending; no statistical conclusions without reconstructed predictions.')
    lines.extend(['','### 失败迁移与人工干预' if zh else '### Failure transitions and reviewed interventions','```json',
                  json.dumps({'failure_trajectories':result['failure_trajectories'],'audit':result['audit'],
                              'confirmation':result['confirmation'],'claims':result['claims']},ensure_ascii=False,indent=2),'```'])
    lines.extend(['','## 证据缺口' if zh else '## Evidence gaps'])
    lines.extend(f"- {x['gate']} / {x['check']}: `{x['path']}` — {x['reason']}" for x in result['missing'])
    lines.extend(['','## 资源成本' if zh else '## Resource costs',
        '```json',json.dumps(result['training']['costs'],ensure_ascii=False,indent=2),'```',
        '', '## 限制与相关工作' if zh else '## Limitations and related work'])
    lines.extend('- '+x for x in result['limitations'])
    if result['related_work']:
        lines.extend(['','| Citation | URL | Relevance | Difference |','| --- | --- | --- | --- |'])
        for row in result['related_work']:
            lines.append('| '+' | '.join(str(row[k]).replace('|','\\|').replace('\n',' ') for k in ('citation','url','relevance','difference'))+' |')
    else:lines.append('相关工作比较须由已绑定的矩阵给出；缺失时保持pending。' if zh else 'Related-work comparisons require the byte-bound matrix; absent comparisons remain pending.')
    return '\n'.join(lines)+'\n'


def build_report(root: Path,output: Path) -> dict[str,Any]:
    """Write a truthful staged/full report solely inside output; root is read-only."""
    root=Path(root).resolve();output=Path(output).resolve()
    result=verify_research(root,'full_research');output.mkdir(parents=True,exist_ok=True)
    for lang in ('zh','en'):
        (output/f'research-report-{lang}.md').write_text(_render(result,lang),encoding='utf-8')
    stream=io.StringIO(newline='');writer=csv.DictWriter(stream,fieldnames=['gate','check','path','sha256','bytes','status','reason']);writer.writeheader()
    for record in result['evidence']:writer.writerow({'gate':'artifact',**record})
    for record in result['missing']:
        writer.writerow({k:record.get(k,'') for k in ('gate','check','path','status','reason')})
    (output/'evidence-map.csv').write_text(stream.getvalue(),encoding='utf-8-sig')
    atomic_write_json(output/'completion.json',result,output)
    return result
