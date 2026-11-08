"""Additive Q scope for the 24 already-fitted S1/S3 checkpoints; never trains."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import os
from pathlib import Path

import numpy as np

from .config import canonical_hash
from .storage import atomic_write_json, bounded_path, hash_file, read_json
from .types import PredictionRow
from .scoped_evaluation import _load_Q, _asset_files, _versions, _sources as prior_sources
from .evaluate import _metrics, _read_predictions, validate_prediction_rows

REL = Path('training/supplement-q-v1')
AUTH = REL/'authorization.json'
SUPPLEMENT = Path('reports/overnight/supplement-1-4/manifest.json')
COUNTS = {'S1':16, 'S3':8}


def folder(stage):
    if stage not in COUNTS:raise ValueError('Expected S1 or S3')
    return Path('reports/overnight/supplement-q-v1')/stage


def _sources():
    return {**prior_sources(), 'supplement_q.py':hash_file(Path(__file__))}


def _entries(manifest,stage):
    entries=[row for row in manifest['fits'] if row['supplement_id']==stage]
    if len(entries)!=COUNTS[stage] or len({e['entry']['run_id'] for e in entries})!=len(entries):
        raise ValueError('Supplement checkpoint coverage changed')
    return entries


def authorize(root):
    """Record explicit user scope using manifests/hashes only; never loads Q."""
    root=Path(root).resolve();manifest=read_json(root/SUPPLEMENT)
    prepared=read_json(root/'data/prepared_manifest.json')
    body={'schema_version':1,'user_instruction':'你先启动S3和S1的独立Q测试',
        'created_at_utc':datetime.now(timezone.utc).isoformat(),'official_test_predictions_allowed':True,
        'new_training_allowed':False,'checkpoint_selection':'all previously fitted S1/S3 cells; no Q selection',
        'supplement_manifest_sha256':hash_file(root/SUPPLEMENT),
        'prepared_sha256':hash_file(root/'data/prepared_manifest.json'),
        'stages':{stage:_entries(manifest,stage) for stage in COUNTS},
        'metadata':{ds:prepared['datasets'][ds] for ds in ['banking77','go_emotions']},
        'threshold_rule':'single_label_argmax; multilabel_logit_or_margin_zero',
        'primary_metrics':{'banking77':'macro_f1','go_emotions':'macro_ap'},
        'contrasts':{'S1':'within dataset/model: LR and total-budget recipe contrasts; seed42 only',
            'S3':'within representation: full vs tail-reduced support at fixed risk; uniform vs skew at fixed support'},
        'Q_exposure_note':'Same official partitions as prior convergence Q, evaluated on checkpoints fitted before release; no refitting or tuning.'}
    body['authorization_hash']=canonical_hash(body)
    path=root/AUTH
    if path.exists():
        existing=_authorization(root)
        if existing['stages']!=body['stages']:raise ValueError('Existing supplemental Q authorization differs')
        return existing
    atomic_write_json(path,body,root);return body


def _authorization(root):
    auth=read_json(root/AUTH)
    if (auth.get('authorization_hash')!=canonical_hash({k:v for k,v in auth.items() if k!='authorization_hash'})
        or auth.get('official_test_predictions_allowed') is not True or auth.get('new_training_allowed') is not False):
        raise ValueError('Supplement Q authorization changed')
    if (hash_file(root/SUPPLEMENT)!=auth['supplement_manifest_sha256']
        or hash_file(root/'data/prepared_manifest.json')!=auth['prepared_sha256']):
        raise ValueError('Supplement/data manifest changed')
    manifest=read_json(root/SUPPLEMENT)
    if auth['stages']!={stage:_entries(manifest,stage) for stage in COUNTS}:
        raise ValueError('Authorized models differ from the original 24 fitted cells')
    prepared=read_json(root/'data/prepared_manifest.json')['datasets']
    if (auth['metadata']!={ds:prepared[ds] for ds in ['banking77','go_emotions']}
        or auth['threshold_rule']!='single_label_argmax; multilabel_logit_or_margin_zero'
        or auth['primary_metrics']!={'banking77':'macro_f1','go_emotions':'macro_ap'}):
        raise ValueError('Authorized labels, metrics or threshold rule differ')
    return auth


def _pin(root,files,path,expected=None):
    path=bounded_path(path,root)
    if not path.is_file():raise PermissionError('Required evidence is missing: '+str(path))
    digest=hash_file(path)
    if expected is not None and digest!=expected:raise ValueError('Evidence bytes changed: '+str(path))
    files[path.relative_to(root).as_posix()]=digest


def _request_valid(status,entry):
    run=entry['run'];model_hash=canonical_hash(run)
    if entry['engine']=='core':return status.get('request_hash')==model_hash
    if entry['engine']!='diagnostics':return False
    keys=['model_request_hash','diagnostics_code_hash','runtime_resource_fingerprint']
    if any(key not in status for key in keys):return False
    identity={key:status[key] for key in keys};fingerprint=identity['runtime_resource_fingerprint']
    return (identity['model_request_hash']==model_hash and identity['diagnostics_code_hash']==run['code_hash']
        and isinstance(fingerprint,str) and len(fingerprint)==64 and all(c in '0123456789abcdef' for c in fingerprint)
        and status.get('request_hash')==canonical_hash(identity))


def _material(root,stage):
    auth=_authorization(root);files={};models=[]
    design=read_json(root/'matrix/design_manifest.json')
    conditions={c['condition_id']:c for c in design.get('conditions',[])}
    for item in auth['stages'][stage]:
        entry=item['entry'];run=entry['run'];out=root/'runs'/entry['run_id'];status_path=out/'status.json'
        if not status_path.exists():raise PermissionError('Missing saved supplemental checkpoint')
        status=read_json(status_path)
        if (status.get('status')!='complete' or not status.get('reload_verified')
            or not status.get('actual_model_training_run') or status.get('run')!=run
            or not _request_valid(status,entry)
            or status.get('training_ids_sha256')!=run['hparams']['role_ids_hash']['train']
            or run['input_hash']!=auth['metadata'][item['dataset_id']]['inputs']['input_hash']):
            raise ValueError('Saved checkpoint identity or completion differs')
        if 'bundle.joblib' not in status.get('files',{}):raise ValueError('Missing authenticated model bundle')
        _pin(root,files,status_path)
        for name,digest in status['files'].items():_pin(root,files,bounded_path(out/name,out),digest)
        tail=[];factors={}
        if stage=='S3':
            meta=conditions.get(run['condition_id'],{}).get('metadata',{})
            tail=meta.get('current_tail_labels',[])
            if not tail or not set(tail)<set(range(run['hparams']['label_count'])):
                raise ValueError('Missing original training-condition tail labels')
            factors={k:meta[k] for k in ['support','risk','reference_n']}
        models.append({**entry,'stage':stage,'role':'supplemental_Q','training_seed':run['training_seed'],
            'score_kind':status['score_kind'],'tail_labels':tail,'factors':factors})
    metadata={ds:auth['metadata'][ds] for ds in sorted({m['dataset_id'] for m in models})}
    for meta in metadata.values():
        for purpose in ['train','select','test']:
            record=meta['partitions'][purpose];_pin(root,files,root/record['file'],record['sha256'])
        _pin(root,files,root/meta['inputs']['file'],meta['inputs']['file_sha256'])
    for path in [AUTH,SUPPLEMENT,Path('data/prepared_manifest.json'),Path('matrix/design_manifest.json'),Path('assets/assets.lock.json')]:
        _pin(root,files,root/path)
    body={'schema_version':1,'stage':stage,'models':models,'metadata':metadata,'files':files,
        'source_inventory':_sources(),'versions':_versions(),'asset_files':_asset_files(root),
        'authorization_hash':auth['authorization_hash'],'threshold_rule':auth['threshold_rule'],
        'primary_metrics':auth['primary_metrics'],'contrasts':auth['contrasts'][stage],
        'official_test_predictions_allowed':True}
    return body


def freeze(root,stage):
    root=Path(root).resolve();dest=root/folder(stage)/'evaluation.freeze.json'
    if dest.exists():return verify(root,stage)
    body=_material(root,stage);body['created_at_utc']=datetime.now(timezone.utc).isoformat()
    body['lock_hash']=canonical_hash(body);atomic_write_json(dest,body,root);return body


def verify(root,stage):
    root=Path(root).resolve()
    lock=read_json(root/folder(stage)/'evaluation.freeze.json')
    if lock.get('lock_hash')!=canonical_hash({k:v for k,v in lock.items() if k!='lock_hash'}):
        raise ValueError('Supplement scope hash changed')
    expected=_material(root,stage)
    if {k:v for k,v in lock.items() if k not in {'created_at_utc','lock_hash'}}!=expected:
        raise ValueError('Frozen scope differs from authorized models, labels, rules or authenticated evidence')
    return lock


def _diagnostics(metrics,entry):
    tail=entry['tail_labels']
    if not tail:return {}
    per={r['label_id']:r for r in metrics['per_label']};head=sorted(set(per)-set(tail))
    result={'tail_labels':tail,'factors':entry['factors'],'label_groups_source':'original training condition, fixed before Q'}
    for label,group in [('tail',tail),('head',head)]:
        for metric in ['f1','recall']:result[label+'_macro_'+metric]=float(np.mean([per[i][metric] for i in group]))
        result[label+'_Q_support']=sum(per[i]['support'] for i in group)
    return result


def _one(root,lock,entry,examples,inputs,scorer):
    import pyarrow as pa
    import pyarrow.parquet as pq
    run=entry['run'];meta=lock['metadata'][entry['dataset_id']];c=len(meta['label_names']);multi=meta['task']!='single_label'
    identity=canonical_hash({'freeze':lock['lock_hash'],'run':run,
        'rows':[(r.sample_id,r.group_id,r.labels,v.text_hash) for r,v in zip(examples,inputs)]})
    path=root/folder(lock['stage'])/'predictions'/(entry['run_id']+'.parquet');info_path=path.with_suffix('.manifest.json')
    def predicted(score):
        return tuple(int(i) for i in np.flatnonzero(np.asarray(score)>=(.5 if entry['score_kind']=='probability' else 0.))) if multi else (int(np.argmax(score)),)
    if info_path.exists():
        info=read_json(info_path)
        if info['request_hash']!=identity or hash_file(path)!=info['sha256']:raise ValueError('Cached prediction identity changed')
        saved=_read_predictions(path);validate_prediction_rows(saved,c,meta['task'])
        if len(saved)!=len(examples):raise ValueError('Cached Q coverage changed')
        for row,example in zip(saved,examples):
            if (row.run_id!=entry['run_id'] or row.sample_id!=example.sample_id or row.group_id!=example.group_id
                or row.labels!=example.labels or row.input_hash!=run['input_hash'] or row.score_kind!=entry['score_kind']
                or row.prediction!=predicted(row.scores)):raise ValueError('Cached sample/threshold provenance changed')
        metrics=_metrics(saved,meta);diagnostics=_diagnostics(metrics,entry)
        for key,value in [('metrics',metrics),('diagnostics',diagnostics),('timing',info['timing'])]:
            if canonical_hash(value)!=info[key+'_hash'] or canonical_hash(info[key])!=info[key+'_hash']:
                raise ValueError('Cached '+key+' changed')
        for key,value in [('run_id',entry['run_id']),('stage',lock['stage']),('dataset_id',entry['dataset_id']),
                          ('model_id',entry['model_id']),('freeze_hash',lock['lock_hash']),('rows',len(examples)),
                          ('training_seed',entry['training_seed']),('role','supplemental_Q'),
                          ('path',path.relative_to(root).as_posix())]:
            if info[key]!=value:raise ValueError('Cached result fields changed')
        return info
    scores,status,timing=scorer(root/'runs'/entry['run_id'],inputs);scores=np.asarray(scores,dtype=np.float64)
    if status['run']!=run or scores.shape!=(len(inputs),c):raise ValueError('Scorer model/score space differs')
    rows=tuple(PredictionRow(entry['run_id'],r.sample_id,r.group_id,tuple(r.labels),predicted(s),
        tuple(float(x) for x in s),entry['score_kind'],run['input_hash']) for r,s in zip(examples,scores))
    validate_prediction_rows(rows,c,meta['task']);path.parent.mkdir(parents=True,exist_ok=True)
    schema=pa.schema([('run_id',pa.string()),('sample_id',pa.string()),('group_id',pa.string()),
        ('labels',pa.list_(pa.int32())),('prediction',pa.list_(pa.int32())),('scores',pa.list_(pa.float64())),
        ('score_kind',pa.string()),('input_hash',pa.string())])
    temporary=path.with_suffix('.parquet.part');pq.write_table(pa.Table.from_pylist([asdict(r) for r in rows],schema=schema),temporary,compression='zstd');temporary.replace(path)
    saved=_read_predictions(path)
    if saved!=rows:raise ValueError('Stored score precision changed')
    metrics=_metrics(saved,meta);diagnostics=_diagnostics(metrics,entry)
    info={'run_id':entry['run_id'],'stage':lock['stage'],'dataset_id':entry['dataset_id'],'model_id':entry['model_id'],
        'training_seed':entry['training_seed'],'role':'supplemental_Q','request_hash':identity,'freeze_hash':lock['lock_hash'],
        'rows':len(rows),'path':path.relative_to(root).as_posix(),'sha256':hash_file(path),
        'metrics':metrics,'metrics_hash':canonical_hash(metrics),'timing':timing,'timing_hash':canonical_hash(timing),
        'diagnostics':diagnostics,'diagnostics_hash':canonical_hash(diagnostics)}
    atomic_write_json(info_path,info,root);return info


def predict(root,stage,*,scorer=None):
    root=Path(root).resolve();lock=verify(root,stage)
    if scorer is None:
        from .timed_inference import score_checkpoint_cpu_timed
        scorer=score_checkpoint_cpu_timed
    journal=root/folder(stage)/'evaluation-started.json'
    if journal.exists():
        if read_json(journal)['freeze_hash']!=lock['lock_hash']:raise ValueError('Prior Q exposure identity differs')
    else:atomic_write_json(journal,{'freeze_hash':lock['lock_hash'],'Q_access_started':True,
        'started_at_utc':datetime.now(timezone.utc).isoformat(),'datasets':list(lock['metadata'])},root)
    frames={ds:_load_Q(root,meta) for ds,meta in lock['metadata'].items()};results=[]
    for entry in lock['models']:
        examples,inputs=frames[entry['dataset_id']];results.append(_one(root,lock,entry,examples,inputs,scorer))
        atomic_write_json(root/folder(stage)/'progress.json',{'stage':stage,'completed':len(results),
            'total':len(lock['models']),'latest_run_id':entry['run_id'],'checked_at_utc':datetime.now(timezone.utc).isoformat()},root)
        print({'event':'Q_scored','stage':stage,'complete':len(results),'total':len(lock['models'])},flush=True)
    result={'status':'complete','stage':stage,'models':results,'freeze_hash':lock['lock_hash'],
        'official_test_predictions_generated':True,'new_training_performed':False,
        'completed_at_utc':datetime.now(timezone.utc).isoformat()}
    atomic_write_json(root/folder(stage)/'test-results.json',result,root);return result


def run(root,stage,*,scorer=None):
    if not os.environ.get('SLURM_JOB_ID'):raise PermissionError('S1/S3 Q requires Slurm compute allocation')
    root=Path(root).resolve()
    try:
        freeze(root,stage);return predict(root,stage,scorer=scorer)
    except Exception as exc:
        atomic_write_json(root/folder(stage)/'evaluation-failed.json',{'status':'failed','stage':stage,
            'error':repr(exc),'checked_at_utc':datetime.now(timezone.utc).isoformat(),
            'Q_access_started':(root/folder(stage)/'evaluation-started.json').exists()},root)
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=list(COUNTS));parser.add_argument('--authorize',action='store_true')
    args=parser.parse_args();root=Path('research-v3')
    if args.authorize:print({'authorization_hash':authorize(root)['authorization_hash']})
    else:
        result=run(root,args.stage);print({'status':result['status'],'stage':args.stage,'models':len(result['models'])},flush=True)
