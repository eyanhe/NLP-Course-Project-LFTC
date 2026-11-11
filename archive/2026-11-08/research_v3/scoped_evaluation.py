"""One-time Q evaluation of explicitly scoped, fully converged E1 models."""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime,timezone
from importlib.metadata import version
import os
import time
from pathlib import Path

import numpy as np

from . import convergence as driver
from .budgeted_finish import completed_neural,choose_completed,update_completed_selections
from .config import canonical_hash
from .storage import atomic_write_json,bounded_path,hash_file,read_json
from .types import Example,PredictionRow,VisibleInput

FREEZE=driver.REL/'scoped-evaluation.freeze.json'


def _sources():
    names=['scoped_evaluation.py','budgeted_finish.py','timed_inference.py','evaluate.py','thresholds.py']
    return {**driver.driver_sources(),**{n:hash_file(Path(__file__).parent/n) for n in names}}


def _versions():
    return {name:version(name) for name in ['torch','transformers','numpy','scikit-learn','pyarrow']}


def _asset_files(root):
    assets=read_json(root/'assets/assets.lock.json')
    rows=[r for r in assets['files'] if r.get('model_id')=='distilbert']
    if {r['source_file'] for r in rows}!=set(assets['models']['distilbert']['files']):
        raise ValueError('DistilBERT asset coverage changed')
    checked={}
    for record in rows:
        path=bounded_path(root.parent/record['path'],root.parent)
        if not path.is_file() or hash_file(path)!=record['sha256']:raise ValueError('Actual model/tokenizer asset changed')
        checked[record['path']]=record['sha256']
    return checked


def _pin(root,files,path,expected=None):
    path=bounded_path(path,root)
    if not path.is_file():raise PermissionError('Required evidence is missing: '+str(path))
    actual=hash_file(path)
    if expected is not None and actual!=expected:raise ValueError('Evidence bytes changed: '+str(path))
    files[path.relative_to(root).as_posix()]=actual


def _checkpoint(root,entry,files,role):
    status=driver._status(root,entry)
    if not status or status.get('status')!='complete' or not status.get('reload_verified'):
        raise PermissionError('Checkpoint is not complete and verified')
    if (status.get('request_hash')!=canonical_hash(entry['run']) or not status.get('actual_model_training_run')
            or status.get('training_ids_sha256')!=entry['run']['hparams']['role_ids_hash']['train']):
        raise ValueError('Checkpoint request/training identity differs')
    if role=='primary' and entry['model_id']!='tfidf_svm' and not completed_neural(status):
        raise PermissionError('Primary neural training did not complete normally')
    if role=='primary' and entry['model_id']=='tfidf_svm' and not status.get('solver_verified'):
        raise PermissionError('SVM solver is unresolved')
    output=root/'runs'/entry['run_id']
    _pin(root,files,output/'status.json')
    if 'bundle.joblib' not in status.get('files',{}):raise ValueError('No authenticated model bundle')
    for relative,expected in status['files'].items():_pin(root,files,bounded_path(output/relative,output),expected)
    return {**entry,'role':role,'training_seed':entry['run']['training_seed'],'score_kind':status['score_kind'],
        'stop_reason':status.get('stop_reason','legacy_fixed_budget'),
        'convergence_verified':status.get('convergence_verified',False)}


def freeze_scope(root: Path,selected=None,references=None):
    root=Path(root).resolve()
    if (root/FREEZE).exists():return verify_scope(root)
    manifest=read_json(root/driver.REL/'manifest.json');driver.verify_manifest(root,manifest)
    if len(manifest['candidates'])!=18:raise ValueError('Expected the complete 18-candidate grid')
    update_completed_selections(root,manifest)
    files={};models=[];selection_records={}
    groups={entry['group'] for entry in manifest['candidates']}
    if len(groups)!=6:raise ValueError('Expected six declared dataset/model groups')
    for group in sorted(groups):
        path=root/driver.REL/'selections'/(group.replace('/','--')+'.json')
        if not path.exists():raise PermissionError('Selection is incomplete: '+group)
        selection=read_json(path)
        if (selection['selection_hash']!=canonical_hash({k:v for k,v in selection.items() if k!='selection_hash'})
                or selection['manifest_hash']!=manifest['manifest_hash']):raise ValueError('Selection identity changed')
        entries=[e for e in manifest['candidates'] if e['group']==group]
        statuses=[driver._status(root,e) for e in entries]
        if any(s is None for s in statuses):raise PermissionError('Candidate is missing')
        declared=next(spec['selection_metric'] for spec in manifest['scope_metadata'] if spec['dataset_id']==entries[0]['dataset_id'])
        if selection['metric']!=declared:raise ValueError('Selection metric differs from declared V metric')
        expected=choose_completed(statuses,3,declared)
        chosen=next(e for e in entries if e['run_id']==expected['run_id'])
        if selection['chosen']!=chosen:raise ValueError('Stored choice differs from V-only selection')
        if set(selection['candidate_status_sha256'])!={e['run_id'] for e in entries}:raise ValueError('Candidate evidence coverage changed')
        for entry in entries:
            _pin(root,files,root/'runs'/entry['run_id']/'status.json',selection['candidate_status_sha256'][entry['run_id']])
            if entry['model_id']=='tfidf_svm':_pin(root,files,root/driver.REL/'solver-audits'/(entry['run_id']+'.json'))
        confirmation=selection['confirmation_entries']
        if chosen['model_id']=='tfidf_svm':
            if confirmation:raise ValueError('SVM cannot have synthetic training replicates')
        else:
            expected_confirm=[]
            for seed in [43,44]:
                e=driver._entry(chosen['run'],chosen['dataset_id'],chosen['model_id'],chosen['training_rows'],{},seed=seed)
                e.update(phase='confirmation',source_run_id=chosen['source_run_id'],recipe_source_run_id=chosen['run_id'])
                expected_confirm.append(e)
            if confirmation!=expected_confirm:raise ValueError('Confirmation recipes or training seeds changed')
        for entry in [chosen,*confirmation]:models.append(_checkpoint(root,entry,files,'primary'))
        _pin(root,files,path);selection_records[group]=selection['selection_hash']
    if len(models)!=14:raise PermissionError('Fourteen primary physical models are required')
    proposal=read_json(root/driver.REL/'proposal.json')
    source_bindings={b['source_run_id']:b for b in proposal['source_bindings']}
    for ref in manifest['fixed_budget_references']:
        source=source_bindings[ref['run_id']]['source_run']
        entry={'run_id':ref['run_id'],'run':source,'dataset_id':ref['dataset_id'],'model_id':ref['model_id'],
            'group':ref['dataset_id']+'/'+ref['model_id'],'data_seed':2026,'phase':'reference',
            'training_rows':source_bindings[ref['run_id']]['training_rows']}
        models.append(_checkpoint(root,entry,files,'fixed_budget_reference'))
    if len(models)!=18 or len({m['run_id'] for m in models})!=18:raise ValueError('Physical test model coverage changed')
    prepared=read_json(root/'data/prepared_manifest.json');metadata={}
    for spec in manifest['scope_metadata']:
        ds=spec['dataset_id'];meta=prepared['datasets'][ds];metadata[ds]=meta
        if meta['partitions']['test']['rows']!=spec['test_rows'] or len(meta['label_names'])!=spec['label_count']:
            raise ValueError('Frozen test metadata scope changed')
        for purpose in ['train','select','test']:
            record=meta['partitions'][purpose];_pin(root,files,root/record['file'],record['sha256'])
        _pin(root,files,root/meta['inputs']['file'],meta['inputs']['file_sha256'])
    for path in [driver.REL/'manifest.json',driver.REL/'proposal.json',Path('data/prepared_manifest.json'),
                 Path('matrix/design_manifest.json'),Path('assets/assets.lock.json')]:_pin(root,files,root/path)
    amendment=root/driver.REL/'budget-policy-amendment.json'
    if amendment.exists():_pin(root,files,amendment)
    body={'schema_version':1,'scope':'convergence_v1_two_E1_conditions','original_global_scope_complete':False,
        'created_at_utc':datetime.now(timezone.utc).isoformat(),'manifest_hash':manifest['manifest_hash'],
        'models':models,'primary_models':14,'reference_models':4,'metadata':metadata,'selection_hashes':selection_records,
        'files':files,'asset_files':_asset_files(root),'source_inventory':_sources(),'versions':_versions(),'official_test_predictions_allowed':True,
        'threshold_rule':'single_label_argmax; multilabel_logit_or_margin_zero',
        'selection_policy':'completed_budget_V_best',
        'optimization_interpretation':'V-best checkpoint within declared training budgets; max_epochs is recorded and does not assert convergence',
        'primary_metrics':{'banking77':'macro_f1','go_emotions':'macro_ap'},
        'contrasts':['per_dataset_bilstm_vs_svm','per_dataset_distilbert_vs_svm',
            'per_dataset_distilbert_vs_bilstm','per_dataset_neural_new_recipe_seed42_vs_fixed_budget_reference'],
        'seed_reporting':'all 42/43/44, arithmetic mean and sample SD; deterministic SVM once',
        'M1_review_identity':'Gemini AI-reviewed exploratory intervention; independent of this E1 test scope'}
    body['lock_hash']=canonical_hash(body);atomic_write_json(root/FREEZE,body,root);return body


def verify_scope(root: Path,scope=None):
    root=Path(root).resolve();path=root/FREEZE if scope is None else bounded_path(scope,root)
    if not path.exists():raise PermissionError('A complete scoped freeze is required before Q')
    lock=read_json(path)
    if lock.get('lock_hash')!=canonical_hash({k:v for k,v in lock.items() if k!='lock_hash'}):raise ValueError('Scoped freeze changed')
    if lock.get('official_test_predictions_allowed') is not True or lock.get('primary_models')!=14 or len(lock.get('models',[]))!=18:
        raise PermissionError('Scope is not authorized and complete')
    if lock['source_inventory']!=_sources() or lock['versions']!=_versions():raise ValueError('Frozen evaluator/environment changed')
    for relative,expected in lock['files'].items():
        path=bounded_path(root/relative,root)
        if not path.is_file() or hash_file(path)!=expected:raise ValueError('Frozen evidence changed: '+relative)
    if lock['asset_files']!=_asset_files(root):raise ValueError('Frozen model/tokenizer assets changed')
    return lock


def _load_Q(root,meta):
    import pyarrow.parquet as pq
    import hashlib
    rows=pq.read_table(root/meta['partitions']['test']['file'],columns=['sample_id','labels','group_id','official_split']).to_pylist()
    examples=tuple(sorted((Example(meta['dataset_id'],r['sample_id'],'',tuple(r['labels']),r['official_split'],r['group_id']) for r in rows),key=lambda r:r.sample_id))
    visible=pq.read_table(root/meta['inputs']['file'],columns=['sample_id','text','start_char','end_char','content_tokens','full_tokens','text_hash'],
        filters=[('partition','=','test')]).to_pylist()
    byid={r['sample_id']:VisibleInput(**r) for r in visible}
    if (len(examples)!=meta['partitions']['test']['rows'] or len({r.sample_id for r in examples})!=len(examples)
            or len(byid)!=len(visible) or set(byid)!={r.sample_id for r in examples}):raise ValueError('Q coverage/unique IDs changed')
    inputs=tuple(byid[r.sample_id] for r in examples)
    if any(r.official_split!='test' or not r.group_id for r in examples):raise ValueError('Invalid Q role or group')
    if any(v.text_hash!=hashlib.sha256(v.text.encode()).hexdigest() for v in inputs):raise ValueError('Q visible input hash changed')
    return examples,inputs


def _predict_one(root,lock,entry,examples,inputs,scorer):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from .evaluate import validate_prediction_rows,_metrics,_read_predictions
    ds=entry['dataset_id'];meta=lock['metadata'][ds];run=entry['run'];c=len(meta['label_names'])
    identity=canonical_hash({'freeze':lock['lock_hash'],'run':run,
        'rows':[(r.sample_id,r.group_id,r.labels,v.text_hash) for r,v in zip(examples,inputs)]})
    path=root/driver.REPORT/'predictions'/(entry['run_id']+'.parquet');info_path=path.with_suffix('.manifest.json')
    if info_path.exists():
        info=read_json(info_path)
        if info['request_hash']!=identity or hash_file(path)!=info['sha256']:raise ValueError('Prediction cache identity/bytes changed')
        saved=_read_predictions(path);validate_prediction_rows(saved,c,meta['task'])
        metrics=_metrics(saved,meta)
        if (len(saved)!=len(examples) or canonical_hash(metrics)!=info['metrics_hash']
                or canonical_hash(info['metrics'])!=info['metrics_hash']):raise ValueError('Cached metrics do not reconstruct')
        for row,example in zip(saved,examples):
            score=np.asarray(row.scores);kind=entry['score_kind'];multi=meta['task'] in {'multilabel','multi_label'}
            predicted=tuple(int(i) for i in np.flatnonzero(score>=(.5 if kind=='probability' else 0.))) if multi else (int(score.argmax()),)
            if (row.run_id!=entry['run_id'] or row.sample_id!=example.sample_id or row.group_id!=example.group_id
                    or row.labels!=example.labels or row.input_hash!=run['input_hash'] or row.score_kind!=kind
                    or row.prediction!=predicted):raise ValueError('Cached prediction provenance or threshold changed')
        for key,expected in [('run_id',entry['run_id']),('dataset_id',ds),('model_id',entry['model_id']),
                             ('role',entry['role']),('training_seed',entry['training_seed']),('rows',len(examples))]:
            if info[key]!=expected:raise ValueError('Cached result identity changed')
        if 'timing' in info and canonical_hash(info['timing'])!=info.get('timing_hash'):
            raise ValueError('Cached timing record changed')
        return info
    inference_started=time.perf_counter()
    scored=scorer(root/'runs'/entry['run_id'],inputs)
    if len(scored)==3:scores,status,timing=scored
    else:
        scores,status=scored
        timing={'measurement':'injected_scorer_total_only','predict_seconds':None,
            'authenticated_inference_seconds':time.perf_counter()-inference_started,'rows':len(inputs)}
    scores=np.asarray(scores,dtype=np.float64)
    if scores.shape!=(len(inputs),c) or status['run']!=run:raise ValueError('Scorer model/full-label identity changed')
    kind=entry['score_kind'];cutoff=.5 if kind=='probability' else 0.;multi=meta['task'] in {'multilabel','multi_label'}
    rows=tuple(PredictionRow(entry['run_id'],r.sample_id,r.group_id,tuple(r.labels),
        tuple(int(x) for x in np.flatnonzero(s>=cutoff)) if multi else (int(s.argmax()),),
        tuple(float(x) for x in s),kind,run['input_hash']) for r,s in zip(examples,scores))
    validate_prediction_rows(rows,c,meta['task']);path.parent.mkdir(parents=True,exist_ok=True)
    schema=pa.schema([('run_id',pa.string()),('sample_id',pa.string()),('group_id',pa.string()),
        ('labels',pa.list_(pa.int32())),('prediction',pa.list_(pa.int32())),('scores',pa.list_(pa.float64())),
        ('score_kind',pa.string()),('input_hash',pa.string())])
    temporary=path.with_suffix('.parquet.part');pq.write_table(pa.Table.from_pylist([asdict(r) for r in rows],schema=schema),temporary,compression='zstd');temporary.replace(path)
    saved=_read_predictions(path)
    if saved!=rows:raise ValueError('Saved prediction precision changed')
    metrics=_metrics(saved,meta)
    info={'run_id':entry['run_id'],'dataset_id':ds,'model_id':entry['model_id'],'training_seed':entry['training_seed'],
        'role':entry['role'],'request_hash':identity,'freeze_hash':lock['lock_hash'],'rows':len(rows),
        'path':path.relative_to(root).as_posix(),'sha256':hash_file(path),'metrics':metrics,'metrics_hash':canonical_hash(metrics),
        'timing':timing,'timing_hash':canonical_hash(timing)}
    atomic_write_json(info_path,info,root);return info


def predict_scope(root: Path,scope=None,*,scorer=None):
    root=Path(root).resolve();lock=verify_scope(root,scope)
    if scorer is None:
        from .timed_inference import score_checkpoint_cpu_timed
        scorer=score_checkpoint_cpu_timed
    journal=root/driver.REPORT/'evaluation-started.json'
    if journal.exists():
        if read_json(journal)['freeze_hash']!=lock['lock_hash']:raise ValueError('Q was exposed under another scope')
    else:atomic_write_json(journal,{'freeze_hash':lock['lock_hash'],'started_at_utc':datetime.now(timezone.utc).isoformat(),
        'datasets':list(lock['metadata']),'Q_access_started':True},root)
    frames={ds:_load_Q(root,meta) for ds,meta in lock['metadata'].items()}
    results=[]
    for entry in lock['models']:
        examples,inputs=frames[entry['dataset_id']];results.append(_predict_one(root,lock,entry,examples,inputs,scorer))
    result={'status':'complete','scope':lock['scope'],'freeze_hash':lock['lock_hash'],'models':results,
        'completed_at_utc':datetime.now(timezone.utc).isoformat(),'official_test_predictions_generated':True}
    atomic_write_json(root/driver.REPORT/'test-results.json',result,root);return result


def run_scoped_evaluation(root):
    if not os.environ.get('SLURM_JOB_ID'):raise PermissionError('Formal inference requires Slurm compute allocation')
    root=Path(root)
    try:freeze_scope(root)
    except (PermissionError,ValueError) as exc:
        result={'status':'blocked','reason':str(exc),'official_test_predictions_generated':False,
            'checked_at_utc':datetime.now(timezone.utc).isoformat()}
        atomic_write_json(root/driver.REPORT/'evaluation-blocked.json',result,root)
        print(result,flush=True);return result
    return predict_scope(root)
