"""Serial post-selection dispatch; no official-test loading or prediction."""
from __future__ import annotations

from dataclasses import asdict,replace
from datetime import datetime,timezone
import gc
import hashlib
import os
from pathlib import Path

import joblib

from .config import canonical_hash
from .diagnostics import GenericCLSCache,diagnostics_code_hash,fit_mechanism_run
from .execution import authorize_phase,run_lease
from .features import fit_features
from .freeze import assert_resolved_coverage
from .orchestrator import _condition_ids,load_case
from .storage import atomic_write_json,bounded_path,hash_file,read_json
from .train import _joblib_atomic,fit_run,learner_code_hash
from .types import RunSpec,VisibleInput


def _ids_hash(ids):
    return hashlib.sha256('\n'.join(sorted(ids)).encode()).hexdigest()


def _run(entry):
    run=RunSpec(**entry['run'])
    expected=canonical_hash(asdict(replace(run,run_id='')))
    if (run.run_id!=expected or entry['run_id']!=run.run_id or entry.get('artifact_id',run.run_id)!=run.run_id
            or entry.get('expected_model_request_hash')!=canonical_hash(asdict(run))):
        raise ValueError('Resolved model request identity changed')
    if any(entry.get(key)!=getattr(run,key) for key in ('condition_id','model_id')):
        raise ValueError('Resolved model/condition identity changed')
    engine=entry['engine']
    if engine not in {'core','diagnostics'}:raise ValueError('Unknown fitting engine')
    expected_code=learner_code_hash() if engine=='core' else diagnostics_code_hash()
    if run.code_hash!=expected_code:raise ValueError('Resolved learner code identity changed')
    return run


def unique_fit_entries(manifest):
    ready=manifest.get('status')=='resolved' or (manifest.get('status')=='training_resolved' and manifest.get('training_only') is True)
    if not ready or manifest.get('unresolved_nodes'):
        raise PermissionError('Post-selection run manifest is unresolved')
    entries={}
    for entry in manifest['runs']:
        run=_run(entry)
        if run.run_id in entries:
            previous=entries[run.run_id]
            if previous['run']!=entry['run'] or previous['engine']!=entry['engine']:
                raise ValueError('Conflicting requests for a checkpoint alias')
        else:entries[run.run_id]=entry
    return tuple(entries.values())


def shared_tfidf_features(root: Path,meta: dict):
    """Fit once on all unlabeled T visible inputs; never read the T label column."""
    import pyarrow.parquet as pq
    root=Path(root).resolve()
    hp={'max_features':30000,'ngram_range':[1,2],'sublinear_tf':True,'min_df':1}
    source={'dataset_id':meta['dataset_id'],'pool_manifest_hash':meta['manifest_hash'],
        'common_input_hash':meta['inputs']['input_hash'],'rows':meta['partitions']['train']['rows'],
        'feature_code_hash':hash_file(Path(__file__).parent/'features.py'),'hparams':hp}
    directory=root/'features/shared-tfidf'/canonical_hash(source)
    path=directory/'features.joblib';info_path=directory/'manifest.json'
    if info_path.exists():
        info=read_json(info_path)
        if info.get('source')!=source or not path.is_file() or hash_file(path)!=info['features_sha256']:
            raise ValueError('Shared full-T feature cache bytes/hash changed')
        features=joblib.load(path)
        if len(features.fitted_sample_ids)!=source['rows'] or _ids_hash(features.fitted_sample_ids)!=info['training_ids_hash']:
            raise ValueError('Shared full-T representation coverage changed')
        return features
    ids=pq.read_table(bounded_path(root/meta['partitions']['train']['file'],root),columns=['sample_id']).column(0).to_pylist()
    records=pq.read_table(bounded_path(root/meta['inputs']['file'],root),
        columns=['sample_id','text','start_char','end_char','content_tokens','full_tokens','text_hash'],
        filters=[('partition','=','train')]).to_pylist()
    inputs=tuple(sorted((VisibleInput(**row) for row in records),key=lambda row:row.sample_id))
    if (len(ids)!=source['rows'] or len(set(ids))!=len(ids) or len(inputs)!=len(ids)
            or {row.sample_id for row in inputs}!=set(ids)):
        raise ValueError('Shared representation must cover exactly all unique T rows')
    if any(row.text_hash!=hashlib.sha256(row.text.encode()).hexdigest() for row in inputs):
        raise ValueError('Shared representation input bytes changed')
    features=fit_features('fixed_tfidf_svm',inputs,hp)
    directory.mkdir(parents=True,exist_ok=True)
    _joblib_atomic(features,path)
    atomic_write_json(info_path,{'source':source,'features_sha256':hash_file(path),
        'training_ids_hash':_ids_hash(ids),'labels_accessed':False,'source_partition':'train'},root)
    return features


def fit_declared_run(entry: dict,root: Path,prepared: dict,assets: dict):
    """Internal exact-fit primitive; the batch entrypoint owns phase authorization."""
    root=Path(root).resolve();run=_run(entry)
    metadata=prepared.get('datasets',prepared)
    meta=metadata[entry['dataset_id']]
    if run.asset_hash!=assets['asset_hash'] or run.input_hash!=meta['inputs']['input_hash']:
        raise ValueError('Resolved asset/common-input identity changed')
    design=read_json(root/'matrix/design_manifest.json')
    matches=[row for row in design['conditions'] if row['condition_id']==run.condition_id]
    if len(matches)!=1 or matches[0]['dataset_id']!=meta['dataset_id']:
        raise ValueError('Resolved condition/dataset identity changed')
    condition=matches[0];ids=_condition_ids(condition,root)
    roles=run.hparams.get('role_ids_hash',{})
    if roles.get('train')!=_ids_hash(ids) or condition['training_set_hash']!=_ids_hash(ids):
        raise ValueError('Resolved exact training IDs changed')
    if roles.get('discovery')!=_ids_hash(()):raise ValueError('Post-selection fitting cannot read D labels')
    if run.strict_no_val and roles.get('select')!=_ids_hash(()):
        raise ValueError('Strict E2 fitting cannot bind V labels')
    data,inputs=load_case(root,meta,ids,selection_ids=() if run.strict_no_val else None,discovery_ids=())
    if data.manifest_hash!=condition['metadata']['pool_manifest_hash']:
        raise ValueError('Resolved training pool identity changed')
    if not run.strict_no_val and roles.get('select')!=_ids_hash(row.sample_id for row in data.partitions['select']):
        raise ValueError('Resolved selection IDs changed')
    output=bounded_path(root/'runs'/run.run_id,root/'runs')
    if entry['engine']=='core':
        result=fit_run(run,data,inputs,assets,output)
        if run.model_id=='modernbert':
            from .costs import record_token_budget
            record_token_budget(run,tuple(inputs[sid] for sid in sorted(ids)),
                tuple(inputs[row.sample_id] for row in data.partitions['select']),assets,root)
        return result
    extra={}
    if run.model_id=='fixed_tfidf_svm':extra['shared_features']=shared_tfidf_features(root,meta)
    if run.model_id=='frozen_generic_distilbert_head':
        extra['cls_cache']=GenericCLSCache(root/'features/generic-cls')
    return fit_mechanism_run(run,data,inputs,assets,output,**extra)


def run_resolved_batch(p: dict,root: Path,authorization: Path):
    """Run every unique required checkpoint only after the real global G2b lock."""
    root=Path(root).resolve()
    permit=authorize_phase(p,root,authorization,'main')
    authorize_phase(p,root,authorization,'mechanisms')
    manifest=read_json(root/'run_manifest.json');design=read_json(root/'matrix/design_manifest.json')
    if canonical_hash({k:v for k,v in manifest.items() if k!='manifest_hash'})!=manifest['manifest_hash']:
        raise ValueError('Resolved manifest hash changed')
    if manifest['design_hash']!=design['design_hash'] or manifest['protocol_hash']!=p['_protocol_hash']:
        raise ValueError('Resolved batch protocol/design identity changed')
    assert_resolved_coverage(design,manifest)
    entries=unique_fit_entries(manifest)
    prepared=read_json(root/'data/prepared_manifest.json');assets=read_json(root/'assets/assets.lock.json')
    if assets['asset_hash']!=manifest['asset_hash']:raise ValueError('Resolved batch assets changed')
    request=canonical_hash({'permit':permit,'manifest_hash':manifest['manifest_hash']})
    completed=[]
    with run_lease(root/'training/workers/post-selection',request):
        for entry in entries:
            event={'phase':'post_selection','status':'fitting','pid':os.getpid(),
                'recorded_at_utc':datetime.now(timezone.utc).isoformat(),'run_id':entry['run_id'],
                'dataset_id':entry['dataset_id'],'model_id':entry['model_id'],
                'completed_fit_count':len(completed),'total_fit_count':len(entries)}
            atomic_write_json(root/'training/post-selection-progress.json',event,root)
            result=fit_declared_run(entry,root,prepared,assets)
            if result['status']!='complete' or not result.get('reload_verified'):
                raise ValueError('A required fit did not complete checkpoint verification')
            completed.append(entry['run_id']);gc.collect()
            import torch
            if torch.cuda.is_available():torch.cuda.empty_cache()
        summary={'phase':'post_selection','status':'complete','complete':True,
            'manifest_hash':manifest['manifest_hash'],'completed_fit_count':len(completed),
            'total_fit_count':len(entries),'planned_physical_entries':len(manifest['runs']),
            'checkpoint_run_ids':completed,'official_test_predictions_generated':False,
            'recorded_at_utc':datetime.now(timezone.utc).isoformat()}
        atomic_write_json(root/'training/post-selection-completion.json',summary,root)
        atomic_write_json(root/'training/post-selection-progress.json',summary,root)
        return summary
