"""V-only selection after normal training; epoch caps are annotations, not vetoes."""
from datetime import datetime,timezone
import math
import os
import json
from pathlib import Path

from . import convergence as driver
from .config import canonical_hash
from .storage import atomic_write_json,hash_file,read_json


def completed_neural(record):
    if record.get('status')!='complete' or not record.get('reload_verified'):return False
    reason=record.get('stop_reason')
    if reason=='patience':return bool(record.get('convergence_verified'))
    if reason=='max_epochs':
        maximum=record['run']['hparams']['max_epochs']
        return record.get('epochs_completed')==maximum and 0<record.get('selected_epoch',0)<=maximum
    return False


def choose_completed(records,expected_count,metric):
    if len(records)!=expected_count:raise PermissionError('Candidate set is incomplete')
    for record in records:
        if not record or record.get('status')!='complete' or not record.get('reload_verified'):
            raise PermissionError('Training did not finish normally')
        if record['run']['model_id']=='tfidf_svm':
            if not record.get('solver_verified'):raise PermissionError('SVM solver audit is incomplete')
        elif not completed_neural(record):raise PermissionError('No normally completed neural checkpoint')
        value=record.get('validation',{}).get(metric)
        if value is None or not math.isfinite(value):raise ValueError('Missing declared V metric')
    best=max(record['validation'][metric] for record in records)
    return min((r for r in records if abs(r['validation'][metric]-best)<=1e-12),
        key=lambda r:r['run']['hparams'].get('lr',r['run']['hparams'].get('C',1)))


def update_completed_selections(root,manifest):
    root=Path(root);folder=root/driver.REL/'selections'
    with driver._selection_lock(folder):
        for group in sorted({e['group'] for e in manifest['candidates']}):
            path=folder/(group.replace('/','--')+'.json')
            if path.exists():continue
            entries=[e for e in manifest['candidates'] if e['group']==group]
            records=[driver._status(root,e) for e in entries]
            if any(not r or r['status']!='complete' for r in records):continue
            metric=next(s['selection_metric'] for s in manifest['scope_metadata'] if s['dataset_id']==entries[0]['dataset_id'])
            try:winner=choose_completed(records,3,metric)
            except PermissionError:continue
            chosen=next(e for e in entries if e['run_id']==winner['run_id']);confirm=[]
            if chosen['engine']=='convergence':
                for seed in manifest['confirmation_seeds']:
                    entry=driver._entry(chosen['run'],chosen['dataset_id'],chosen['model_id'],chosen['training_rows'],{},seed=seed)
                    entry.update(phase='confirmation',source_run_id=chosen['source_run_id'],recipe_source_run_id=chosen['run_id'])
                    confirm.append(entry)
            body={'manifest_hash':manifest['manifest_hash'],'group':group,'metric':metric,'chosen':chosen,
                'candidate_status_sha256':{e['run_id']:hash_file(root/'runs'/e['run_id']/'status.json') for e in entries},
                'confirmation_entries':confirm,'selected_at_utc':datetime.now(timezone.utc).isoformat(),
                'selection_policy':'completed_budget_V_best','max_epochs_is_not_a_convergence_claim':True}
            body['selection_hash']=canonical_hash(body);atomic_write_json(path,body,root)


def pending_repeats(root,manifest):
    return [entry for entry in driver._all_entries(root,manifest)
        if entry.get('phase')=='confirmation' and entry['run']['training_seed'] in {43,44}
        and driver._status(root,entry) is None
        and not (root/driver.REL/'claims'/(entry['run_id']+'.json')).exists()]


def finish(root):
    if not os.environ.get('SLURM_JOB_ID'):raise PermissionError('Finalization requires a Slurm compute allocation')
    root=Path(root).resolve();manifest=read_json(root/driver.REL/'manifest.json');driver.verify_manifest(root,manifest)
    driver.verify_pretrained_assets(root,manifest)
    update_completed_selections(root,manifest)
    # Only finish already planned, unclaimed seed repeats; never restart a grid or a claimed run.
    all_grid_complete=True
    for group in {e['group'] for e in manifest['candidates']}:
        entries=[e for e in manifest['candidates'] if e['group']==group]
        metric=next(s['selection_metric'] for s in manifest['scope_metadata'] if s['dataset_id']==entries[0]['dataset_id'])
        try:choose_completed([driver._status(root,e) for e in entries],3,metric)
        except PermissionError:all_grid_complete=False
    if all_grid_complete:
        for entry in pending_repeats(root,manifest):
            if datetime.now(timezone.utc)>=datetime.fromisoformat(manifest['start_cutoff_utc']):break
            claim=root/driver.REL/'claims'/(entry['run_id']+'.json');claim.parent.mkdir(parents=True,exist_ok=True)
            try:
                with claim.open('x',encoding='utf-8') as stream:
                    json.dump({'worker_id':os.environ['SLURM_JOB_ID'],'job_id':os.environ['SLURM_JOB_ID'],
                        'host':os.environ.get('HOSTNAME'),'run_id':entry['run_id'],
                        'claimed_at_utc':datetime.now(timezone.utc).isoformat(),'phase':'confirmation'},stream)
            except FileExistsError:continue
            print({'event':'finish_planned_repeat','run_id':entry['run_id'],'seed':entry['run']['training_seed']},flush=True)
            try:driver.fit_entry(root,entry,manifest)
            except Exception as exc:
                print({'event':'repeat_not_complete','run_id':entry['run_id'],'error':repr(exc)},flush=True)
    update_completed_selections(root,manifest)
    from .scoped_evaluation import run_scoped_evaluation
    return run_scoped_evaluation(root)


if __name__=='__main__':finish(Path('research-v3'))
