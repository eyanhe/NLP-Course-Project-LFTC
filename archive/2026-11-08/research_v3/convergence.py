"""Explicit convergence requests and an atomic shared Slurm work queue."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict,replace
from datetime import datetime,timezone
import gc
import json
import math
import os
from pathlib import Path
import sys
import time
import subprocess

from .config import canonical_hash
from .convergence_training import convergence_code_hash,convergence_sources,fit_convergence,ids_hash
from .storage import atomic_write_json,bounded_path,hash_file,read_json
from .types import RunSpec

REL=Path('training/convergence-v1')
REPORT=Path('reports/overnight/convergence-v1')


def driver_sources():
    names=['convergence.py','orchestrator.py','batch.py','supplemental.py','overnight_training.py']
    return {**convergence_sources(),**{n:hash_file(Path(__file__).parent/n) for n in names}}


def _entry(source,dataset,model,n,changes,*,seed=42,engine='convergence'):
    from .train import learner_code_hash
    original=RunSpec(**source)
    hp={**original.hparams,**changes}
    code=convergence_code_hash() if engine=='convergence' else learner_code_hash()
    run=replace(original,run_id='',training_seed=seed if engine=='convergence' else original.training_seed,
                hparams=hp,code_hash=code)
    run=replace(run,run_id=canonical_hash(asdict(run)))
    return {'dataset_id':dataset,'model_id':model,'training_rows':n,'data_seed':2026,
        'condition_id':run.condition_id,'run_id':run.run_id,'artifact_id':run.run_id,'run':asdict(run),
        'expected_model_request_hash':canonical_hash(asdict(run)),'engine':engine,
        'source_run_id':original.run_id,'group':dataset+'/'+model,'phase':'grid'}


def build_candidates(proposal,distil_device='cpu'):
    if distil_device not in {'cpu','cuda'}:raise ValueError('Unknown DistilBERT execution device')
    sources={b['source_run_id']:b['source_run'] for b in proposal['source_bindings']};result=[]
    for row in proposal['grid']:
        device=distil_device if row['model_id']=='distilbert' else 'cpu'
        hp={k:row[k] for k in ['lr','min_epochs','patience','max_epochs','min_delta','effective_batch','warmup_updates']}
        hp.update(device=device,cpu_threads=4 if device=='cuda' else 8,micro_batch=4,
            epochs=row['max_epochs'],schedule='fixed_warmup_then_constant',precision='float32',supplement_id='convergence-v1')
        result.append(_entry(sources[row['source_run_id']],row['dataset_id'],row['model_id'],row['training_rows'],hp))
    for row in proposal['svm_candidates']:
        source=sources[row['source_run_id']]
        # An exact old candidate is retained byte-for-byte, including device metadata.
        if row['C']==source['hparams'].get('C',1):
            changes={}
        else:changes={'C':row['C'],'device':'cpu','cpu_threads':8}
        result.append(_entry(source,row['dataset_id'],'tfidf_svm',row['training_rows'],changes,engine='core'))
    return result


def prepare(root: Path,proposal: dict,distil_device='cpu'):
    root=Path(root).resolve();path=root/REL/'manifest.json'
    candidates=build_candidates(proposal,distil_device)
    body={'schema_version':1,'scope':'convergence_v1_two_E1_conditions','official_test_predictions_allowed':False,
        'user_instruction':'那现在提到计算集群上去跑吧','created_at_utc':datetime.now(timezone.utc).isoformat(),
        'proposal_sha256':hash_file(root/REL/'proposal.json'),'source_inventory':driver_sources(),
        'prepared_sha256':hash_file(root/'data/prepared_manifest.json'),
        'design_sha256':hash_file(root/'matrix/design_manifest.json'),
        'assets_sha256':hash_file(root/'assets/assets.lock.json'),
        'start_cutoff_utc':proposal['deadlines']['last_new_fit'],'hard_stop_utc':proposal['deadlines']['training_hard_stop'],
        'scope_metadata':proposal['scope'],'candidates':candidates,
        'fixed_budget_references':proposal['fixed_budget_references'],'confirmation_seeds':[43,44],
        'distil_device':distil_device,'precision':'float32','selection_tolerance':1e-12,
        'runtime_ruling':'DistilBERT uses one typed pro6000 GPU if currently available; otherwise all CPU. GPU gets 4 CPUs, no explicit mem/cpu request.'}
    if path.exists():
        old=read_json(path)
        if (old['candidates']!=candidates or old['source_inventory']!=body['source_inventory']):
            raise ValueError('Existing convergence manifest differs; cannot overwrite')
        return old
    body['manifest_hash']=canonical_hash(body);atomic_write_json(path,body,root);return body


def verify_manifest(root,manifest):
    if canonical_hash({k:v for k,v in manifest.items() if k!='manifest_hash'})!=manifest['manifest_hash']:
        raise ValueError('Convergence manifest hash changed')
    if manifest['source_inventory']!=driver_sources():raise ValueError('Convergence driver sources changed')
    for field,path in [('proposal_sha256',REL/'proposal.json'),('prepared_sha256',Path('data/prepared_manifest.json')),
                       ('design_sha256',Path('matrix/design_manifest.json')),('assets_sha256',Path('assets/assets.lock.json'))]:
        if hash_file(root/path)!=manifest[field]:raise ValueError('Pinned training source identity changed: '+field)


def select_recipes(records,expected_count=3,metric='macro_f1'):
    if len(records)!=expected_count:raise PermissionError('Candidate set is incomplete')
    for record in records:
        if record.get('status')!='complete' or not record.get('reload_verified'):
            raise PermissionError('Candidate did not complete verification')
        if record['run']['model_id'] in {'bilstm','distilbert'} and not record.get('convergence_verified'):
            raise PermissionError('Candidate has not passed convergence')
        if record['run']['model_id']=='tfidf_svm' and not record.get('solver_verified'):
            raise PermissionError('SVM solver has not passed convergence')
        value=record.get('validation',{}).get(metric)
        if value is None or not math.isfinite(value):raise ValueError('Missing V selection metric')
    best=max(r['validation'][metric] for r in records)
    tied=[r for r in records if abs(r['validation'][metric]-best)<=1e-12]
    return min(tied,key=lambda r:r['run']['hparams'].get('lr',r['run']['hparams'].get('C',1)))


@contextmanager
def _selection_lock(folder):
    lock=folder/'.selection-lock';folder.mkdir(parents=True,exist_ok=True)
    for _ in range(100):
        try:lock.mkdir();break
        except FileExistsError:time.sleep(.05)
    else:raise RuntimeError('Selection lock is busy; not overwriting another writer')
    try:yield
    finally:lock.rmdir()


def _status(root,entry):
    path=root/'runs'/entry['run_id']/'status.json'
    error_path=root/REPORT/'worker-errors'/(entry['run_id']+'.json')
    if error_path.exists():return {**read_json(error_path),'run':entry['run']}
    if not path.exists():return None
    status=read_json(path)
    if status.get('run')!=entry['run']:
        if not (status['status']=='running' and status.get('request_hash')==entry['expected_model_request_hash']):
            raise ValueError('Candidate saved request differs')
    if entry['model_id']=='tfidf_svm':
        audit_path=root/REL/'solver-audits'/(entry['run_id']+'.json')
        if audit_path.exists():
            audit=read_json(audit_path)
            if audit['status_sha256']!=hash_file(path):raise ValueError('Solver audit source changed')
            status={**status,**audit}
    return status


def _all_entries(root,manifest):
    entries=list(manifest['candidates'])
    for path in sorted((root/REL/'selections').glob('*.json')):
        selection=read_json(path)
        if selection['manifest_hash']!=manifest['manifest_hash']:raise ValueError('Selection uses a different manifest')
        entries.extend(selection['confirmation_entries'])
    return entries


def update_selections(root,manifest):
    folder=root/REL/'selections';groups=sorted({e['group'] for e in manifest['candidates']})
    with _selection_lock(folder):
        for group in groups:
            path=folder/(group.replace('/','--')+'.json')
            if path.exists():continue
            entries=[e for e in manifest['candidates'] if e['group']==group]
            statuses=[_status(root,e) for e in entries]
            if any(s is None or s['status'] in {'running','verifying'} for s in statuses):continue
            metric='macro_ap' if entries[0]['dataset_id']=='go_emotions' else 'macro_f1'
            try:winner=select_recipes(statuses,3,metric)
            except PermissionError as exc:
                atomic_write_json(root/REPORT/'blocked'/path.name,
                    {'group':group,'reason':str(exc),'official_test_predictions_allowed':False},root)
                continue
            chosen=next(e for e in entries if e['run_id']==winner['run_id']);confirm=[]
            if chosen['engine']=='convergence':
                for seed in manifest['confirmation_seeds']:
                    entry=_entry(chosen['run'],chosen['dataset_id'],chosen['model_id'],chosen['training_rows'],{},seed=seed)
                    entry.update(phase='confirmation',source_run_id=chosen['source_run_id'],recipe_source_run_id=chosen['run_id'])
                    confirm.append(entry)
            body={'manifest_hash':manifest['manifest_hash'],'group':group,'metric':metric,'chosen':chosen,
                'candidate_status_sha256':{e['run_id']:hash_file(root/'runs'/e['run_id']/'status.json') for e in entries},
                'confirmation_entries':confirm,'selected_at_utc':datetime.now(timezone.utc).isoformat()}
            body['selection_hash']=canonical_hash(body);atomic_write_json(path,body,root)


def _slurm_state(job_id):
    if not str(job_id).isdigit():return 'UNKNOWN'
    try:
        data=subprocess.check_output(['sacct','-nP','-j',str(job_id),'--format=JobIDRaw,State'],text=True,timeout=8)
        for line in data.splitlines():
            fields=line.split('|')
            if fields[0]==str(job_id):return fields[1].split()[0]
    except (OSError,subprocess.SubprocessError):pass
    return 'UNKNOWN'


def _recover_claim(root,path,entry,job_state):
    lock=path.with_suffix('.recovering')
    try:lock.mkdir()
    except FileExistsError:return False
    try:
        if not path.exists():return True
        try:owner=read_json(path)
        except (json.JSONDecodeError,FileNotFoundError):return False
        job=owner.get('job_id')
        if not job or job_state(job) not in {'FAILED','CANCELLED','TIMEOUT','NODE_FAIL','PREEMPTED','OUT_OF_MEMORY','COMPLETED','BOOT_FAIL','DEADLINE'}:
            return False
        status=_status(root,entry)
        if status and status.get('status') in {'failed','not_started'}:return False
        path.rename(path.with_name(path.stem+'.abandoned.'+str(time.time_ns())+'.json'))
        lease=root/'runs'/entry['run_id']/'.lease.json'
        if lease.exists():
            info=read_json(lease)
            if info.get('request_hash')!=entry['expected_model_request_hash']:raise ValueError('Dead lease has another request')
            lease.rename(lease.with_name('.lease.history.recovered.'+str(time.time_ns())+'.json'))
        return True
    finally:lock.rmdir()


def claim_next(root,worker_id,now,*,manifest=None,device='cpu',job_state=None):
    root=Path(root);manifest=manifest or read_json(root/REL/'manifest.json')
    if now>=datetime.fromisoformat(manifest['start_cutoff_utc']):raise PermissionError('New-fit cutoff reached')
    entries=_all_entries(root,manifest)
    def cost(e):
        return e.get('training_rows',0)*(30 if e.get('model_id')=='distilbert' else 1)
    claims=root/REL/'claims';claims.mkdir(parents=True,exist_ok=True)
    for entry in sorted(entries,key=cost,reverse=True):
        hp=entry['run']['hparams'];requested='cuda' if hp.get('device')=='cuda' and entry.get('model_id')=='distilbert' else 'cpu'
        if requested!=device:continue
        path=claims/(entry['run_id']+'.json')
        if path.exists():
            status=_status(root,entry)
            if status and status.get('status')=='complete' and (entry.get('model_id')!='tfidf_svm' or status.get('solver_verified')):continue
            if not _recover_claim(root,path,entry,job_state or _slurm_state):continue
        try:
            with path.open('x',encoding='utf-8') as stream:
                json.dump({'worker_id':worker_id,'job_id':os.environ.get('SLURM_JOB_ID'),'host':os.environ.get('HOSTNAME'),
                    'run_id':entry['run_id'],'claimed_at_utc':now.isoformat()},stream)
        except FileExistsError:continue
        return entry
    return None


def _solver_check(output):
    from .train import _verified_bundle
    status,bundle=_verified_bundle(output);model=bundle['model']
    estimators=model.estimators if model.task in {'multilabel','multi_label'} else [model.estimator]
    import numpy as np
    estimators=[e for e in estimators if e is not None]
    if not estimators or any(np.max(e.n_iter_)>=e.get_params()['max_iter'] for e in estimators):
        raise ValueError('SVM solver did not stop below iteration cap')
    return {'solver_verified':True,'max_observed_solver_iterations':int(max(np.max(e.n_iter_) for e in estimators))}


def fit_entry(root,entry,manifest):
    cutoff=datetime.fromisoformat(manifest['start_cutoff_utc'])
    if datetime.now(timezone.utc)>=cutoff:raise PermissionError('New-fit cutoff reached before data loading')
    from .batch import fit_declared_run
    from .orchestrator import _condition_ids,load_case
    from .supplemental import relocated_assets
    from .train import _verified_bundle
    run=RunSpec(**entry['run']);prepared=read_json(root/'data/prepared_manifest.json')
    assets=relocated_assets(read_json(root/'assets/assets.lock.json'),root.parent)
    meta=prepared['datasets'][entry['dataset_id']]
    if run.asset_hash!=assets['asset_hash'] or run.input_hash!=meta['inputs']['input_hash']:
        raise ValueError('Asset/common-input identity changed')
    for purpose in ['train','select']:
        p=meta['partitions'][purpose]
        if hash_file(root/p['file'])!=p['sha256']:raise ValueError('T/V source changed')
    if hash_file(root/meta['inputs']['file'])!=meta['inputs']['file_sha256']:raise ValueError('Common input source changed')
    if entry['engine']=='core':
        if datetime.now(timezone.utc)>=cutoff:raise PermissionError('New-fit cutoff reached before SVM fitting')
        result=fit_declared_run(entry,root,prepared,assets)
        # Do not mutate reused legacy status bytes; keep the solver audit beside the new manifest.
        solver=_solver_check(root/'runs'/run.run_id)
        atomic_write_json(root/REL/'solver-audits'/(run.run_id+'.json'),
            {**solver,'status_sha256':hash_file(root/'runs'/run.run_id/'status.json')},root)
        return {**result,**solver}
    design=read_json(root/'matrix/design_manifest.json')
    conditions=[c for c in design['conditions'] if c['condition_id']==run.condition_id]
    if len(conditions)!=1 or conditions[0]['dataset_id']!=entry['dataset_id']:raise ValueError('Training condition changed')
    ids=_condition_ids(conditions[0],root)
    data,inputs=load_case(root,meta,ids,selection_ids=None,discovery_ids=())
    if datetime.now(timezone.utc)>=cutoff:raise PermissionError('New-fit cutoff reached before neural fitting')
    return fit_convergence(entry,data,inputs,assets,root/'runs'/run.run_id,datetime.fromisoformat(manifest['hard_stop_utc']),start_cutoff=cutoff)


def verify_pretrained_assets(root,manifest):
    from .supplemental import relocated_assets
    locked=read_json(root/'assets/assets.lock.json');runtime=relocated_assets(locked,root.parent)
    rows=[r for r in locked['files'] if r.get('model_id')=='distilbert']
    if {r['source_file'] for r in rows}!=set(locked['models']['distilbert']['files']):
        raise ValueError('DistilBERT asset lock coverage changed')
    checked=[]
    for record in rows:
        path=bounded_path(root.parent/record['path'],root.parent)
        actual=Path(runtime['models']['distilbert']['directory'])/record['source_file']
        if actual.resolve()!=path or hash_file(actual)!=record['sha256']:raise ValueError('Pretrained asset bytes changed')
        checked.append({'path':record['path'],'sha256':record['sha256']})
    atomic_write_json(root/REPORT/('asset-audit-'+os.environ['SLURM_JOB_ID']+'.json'),
        {'manifest_hash':manifest['manifest_hash'],'files':checked,'verified_at_utc':datetime.now(timezone.utc).isoformat()},root)


def worker(root,device):
    if not os.environ.get('SLURM_JOB_ID'):raise PermissionError('Heavy work requires Slurm compute allocation')
    root=Path(root).resolve();manifest=read_json(root/REL/'manifest.json');verify_manifest(root,manifest)
    verify_pretrained_assets(root,manifest)
    worker_id=os.environ['SLURM_JOB_ID'];summary=[]
    while datetime.now(timezone.utc)<datetime.fromisoformat(manifest['hard_stop_utc']):
        update_selections(root,manifest)
        try:entry=claim_next(root,worker_id,datetime.now(timezone.utc),manifest=manifest,device=device)
        except PermissionError:break
        if entry is None:
            # GPU worker never waits for the unrelated CPU model families.
            active=[e for e in _all_entries(root,manifest) if
                ('cuda' if e['model_id']=='distilbert' and e['run']['hparams']['device']=='cuda' else 'cpu')==device
                and (_status(root,e) or {}).get('status') in {None,'running','verifying'}]
            if not active:break
            time.sleep(5);continue
        print(json.dumps({'event':'fit_start','entry':{k:entry[k] for k in ['run_id','dataset_id','model_id','phase']},
            'lr':entry['run']['hparams'].get('lr'),'C':entry['run']['hparams'].get('C'),'device':device},ensure_ascii=False),flush=True)
        try:
            result=fit_entry(root,entry,manifest)
            summary.append({'run_id':entry['run_id'],'status':result['status'],'stop_reason':result.get('stop_reason','solver')})
            print(json.dumps({'event':'fit_end',**summary[-1]},ensure_ascii=False),flush=True)
        except Exception as exc:
            saved_path=root/'runs'/entry['run_id']/'status.json'
            saved=read_json(saved_path) if saved_path.exists() else {}
            summary.append({'run_id':entry['run_id'],'status':saved.get('status','failed'),
                'stop_reason':saved.get('stop_reason','failed'),'error':repr(exc)})
            atomic_write_json(root/REPORT/'worker-errors'/(entry['run_id']+'.json'),summary[-1],root)
            print(json.dumps(summary[-1],ensure_ascii=False),flush=True)
        atomic_write_json(root/REPORT/('worker-'+worker_id+'.json'),{'device':device,'completed':summary,
            'checked_at_utc':datetime.now(timezone.utc).isoformat(),'official_test_predictions_allowed':False},root)
        gc.collect()
    update_selections(root,manifest)
    return summary


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','worker','evaluate'])
    parser.add_argument('--root',type=Path,default=Path('research-v3'));parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    args=parser.parse_args();root=args.root.resolve()
    if args.action=='prepare':
        result=prepare(root,read_json(root/REL/'proposal.json'),args.device)
        print(json.dumps({'manifest_hash':result['manifest_hash'],'candidates':len(result['candidates']),'distil_device':result['distil_device']}))
    elif args.action=='worker':worker(root,args.device)
    else:
        from .scoped_evaluation import run_scoped_evaluation
        run_scoped_evaluation(root)


if __name__=='__main__':main()
