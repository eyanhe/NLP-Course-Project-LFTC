"""Explicitly authorized main training while genuine human pilots stay pending."""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import gc
from pathlib import Path
import time

from .batch import fit_declared_run,unique_fit_entries
from .config import PROJECT_ROOT,canonical_hash,load_protocol,output_root
from .diagnostics import diagnostics_code_hash
from .execution import authorize_phase,run_lease
from .fixed_policy import POLICY_FILE,verify_fixed_choice
from .registry import build_run_manifest
from .storage import atomic_write_json as _atomic_write_json,hash_file,read_json
from .train import learner_code_hash

MANIFEST='training/fixed-main-run-manifest.json'
LOCK='training/fixed-main.lock.json'


def atomic_write_json(path,payload,root):
    """A brief Windows progress reader must not terminate the training queue."""
    for attempt in range(20):
        try:
            return _atomic_write_json(path,payload,root)
        except PermissionError as exc:
            if getattr(exc,'winerror',None) not in {5,32,33} or attempt==19:
                raise
            time.sleep(.05)


def authorize_training(p,root,authorization):
    permit=read_json(authorization)
    if permit.get('user_approved_training_before_pilots') is not True or not permit.get('training_order_user_instruction'):
        raise PermissionError('Explicit user order amendment is required before training ahead of human pilots')
    return authorize_phase(p,root,authorization,'fixed_main')


def _coverage(design,manifest):
    if manifest.get('status')!='training_resolved' or manifest.get('training_only') is not True:
        raise ValueError('An explicitly training-only manifest is required')
    expected={r['run_id'] for r in design['physical_runs']};actual=[r['planned_run_id'] for r in manifest['runs']]
    if len(actual)!=len(set(actual)) or set(actual)!=expected:raise ValueError('Required training scope is missing or duplicated')


def order_entries(entries,design):
    """Finish small primary conditions across tasks before expensive full pools."""
    conditions={c['condition_id']:c for c in design['conditions']}
    datasets={d:i for i,d in enumerate(['banking77','go_emotions','yahoo_answers_topics','yelp','imdb','ag_news','rotten_tomatoes'])}
    models={m:i for i,m in enumerate(['tfidf_lr','tfidf_svm','w2v_svm','bilstm','distilbert','modernbert'])}
    def key(row):
        c=conditions[row['condition_id']];n=c['metadata']['rows'];exp=c['experiment_id']
        phase=(0 if exp=='E1' and n<=6000 else 1 if exp=='E2' else
               2 if exp in {'E3','E4','M2','M3'} else 3 if exp=='E5' else 4 if exp=='E1' and n<=100000 else 5 if exp=='E1' else 6)
        return phase,n,datasets.get(row['dataset_id'],99),row.get('data_seed',0),models.get(row['model_id'],99),row.get('training_seed') or 0,row['run_id']
    return tuple(sorted(entries,key=key))


def verify_training_lock(root):
    root=Path(root);lock=read_json(root/LOCK)
    if lock.get('scope')!='training_only' or lock.get('official_test_predictions_allowed') is not False:
        raise ValueError('Training lock cannot authorize official-test predictions')
    if lock.get('lock_hash')!=canonical_hash({k:v for k,v in lock.items() if k!='lock_hash'}):raise ValueError('Training lock hash changed')
    if lock.get('learner_code_hash')!=learner_code_hash() or lock.get('diagnostics_code_hash')!=diagnostics_code_hash():
        raise ValueError('Locked learner implementation changed')
    for record in lock['files'].values():
        if hash_file(root/record['path'])!=record['sha256']:raise ValueError('Training lock artifact bytes changed')
    manifest=read_json(root/MANIFEST);_coverage(read_json(root/'matrix/design_manifest.json'),manifest)
    return lock,manifest


def prepare(p,root,authorization):
    import pyarrow.parquet as pq
    root=Path(root);permit=authorize_training(p,root,authorization)
    if (root/LOCK).exists():return verify_training_lock(root)[1]
    design=read_json(root/'matrix/design_manifest.json');selections=read_json(root/'selections.json')
    policy=read_json(root/POLICY_FILE)
    if policy['policy_hash']!=permit.get('fixed_hyperparameter_policy_hash'):raise ValueError('Authorized fixed parameters changed')
    for node in design['selection_nodes']:verify_fixed_choice(root,node,selections['choices'][node['node_id']],{})
    prepared=read_json(root/'data/prepared_manifest.json');assets=read_json(root/'assets/assets.lock.json')
    hashes={ds:__import__('hashlib').sha256('\n'.join(sorted(pq.read_table(root/meta['partitions']['select']['file'],columns=['sample_id']).column(0).to_pylist())).encode()).hexdigest()
            for ds,meta in prepared['datasets'].items()}
    manifest=build_run_manifest(design,selections,prepared,p,learner_code_hash(),diagnostics_code_hash(),hashes,
        asset_hash=assets['asset_hash'],training_only=True)
    _coverage(design,manifest);unique_fit_entries(manifest)
    atomic_write_json(root/MANIFEST,manifest,root)
    paths=['design.lock.json','matrix/design_manifest.json','data/prepared_manifest.json','assets/assets.lock.json',POLICY_FILE,MANIFEST]
    lock={'scope':'training_only','user_instruction':permit['training_order_user_instruction'],
        'learner_code_hash':learner_code_hash(),'diagnostics_code_hash':diagnostics_code_hash(),
        'official_test_predictions_allowed':False,'human_pilots_verified':False,
        'required_physical_runs':len(manifest['runs']),
        'files':{path:{'path':path,'sha256':hash_file(root/path)} for path in paths}}
    lock['lock_hash']=canonical_hash(lock);atomic_write_json(root/LOCK,lock,root)
    return manifest


def run(p,root,authorization):
    root=Path(root);manifest=prepare(p,root,authorization);lock,_=verify_training_lock(root)
    design=read_json(root/'matrix/design_manifest.json');entries=order_entries(unique_fit_entries(manifest),design)
    prepared=read_json(root/'data/prepared_manifest.json');assets=read_json(root/'assets/assets.lock.json')
    conditions={c['condition_id']:c for c in design['conditions']}
    completed=[];reused=0
    with run_lease(root/'training/workers/fixed-main',lock['lock_hash']) as owner:
        for entry in entries:
            event={'phase':'fixed_main','status':'fitting','pid':owner['pid'],'process_created':owner['process_created'],
                'run_id':entry['run_id'],'dataset':entry['dataset_id'],'model_id':entry['model_id'],
                'experiment_id':conditions[entry['condition_id']]['experiment_id'],
                'training_rows':conditions[entry['condition_id']]['metadata']['rows'],
                'data_seed':entry['data_seed'],'training_seed':entry['training_seed'],
                'completed_fits':len(completed),'total_fits':len(entries),'reused_checkpoints':reused,
                'grid_search_enabled':False,'official_test_predictions_generated':False,
                'recorded_at_utc':datetime.now(timezone.utc).isoformat()}
            atomic_write_json(root/'training/progress.json',event,root)
            try:result=fit_declared_run(entry,root,prepared,assets)
            except Exception as exc:
                event.update(status='failed',error_type=type(exc).__name__,error=str(exc))
                atomic_write_json(root/'training/progress.json',event,root);raise
            if result.get('status')!='complete' or not result.get('reload_verified'):raise ValueError('Actual model fit/reload verification did not complete')
            reused+=int(result.get('reused_complete',False));completed.append(entry['run_id'])
            gc.collect()
            import torch
            if torch.cuda.is_available():torch.cuda.empty_cache()
        summary={'phase':'fixed_main','status':'complete','complete':True,'completed_fits':len(completed),
            'total_fits':len(entries),'checkpoint_run_ids':completed,'reused_checkpoints':reused,
            'training_lock_hash':lock['lock_hash'],'human_pilots_verified':False,
            'official_test_predictions_generated':False,'recorded_at_utc':datetime.now(timezone.utc).isoformat()}
        atomic_write_json(root/'training/main-training-completion.json',summary,root)
        atomic_write_json(root/'training/progress.json',summary,root)
        return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--prepare-only',action='store_true');args=parser.parse_args()
    p=load_protocol(PROJECT_ROOT/'configs/research-v3.json');root=output_root(p)
    authorization=root/'training/authorization.json'
    result=prepare(p,root,authorization) if args.prepare_only else run(p,root,authorization)
    print({'status':result['status'],'runs':len(result.get('runs',[])),'completed_fits':result.get('completed_fits')})


if __name__=='__main__':main()
