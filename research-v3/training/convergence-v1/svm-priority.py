"""Bounded metadata-only queue priority step inside an existing CPU allocation."""
import hashlib
import json
import os
from pathlib import Path
import signal
import time
import uuid
import sys

REL=Path('training/convergence-v1')
REPORT=Path('reports/overnight/convergence-v1')


def read(path):
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else None


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix('.json.part')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');temp.replace(path)


def reserve(root,entries,nonce,job_id,owned=None):
    folder=root/REL/'claims';folder.mkdir(parents=True,exist_ok=True)
    if owned is None:owned=[]
    for entry in entries:
        if entry['model_id']=='tfidf_svm':continue
        status=read(root/'runs'/entry['run_id']/'status.json')
        if status and status.get('status')=='complete':continue
        target=folder/(entry['run_id']+'.json')
        if target.exists():continue
        temporary=folder/('.svm-reservation-'+nonce+'-'+entry['run_id']+'.tmp')
        try:
            descriptor=os.open(temporary,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
            with os.fdopen(descriptor,'w',encoding='utf-8') as stream:
                json.dump({'kind':'temporary_queue_reservation','worker_id':'svm-first-controller',
                    'job_id':job_id,'host':os.environ.get('HOSTNAME'),'run_id':entry['run_id'],
                    'nonce':nonce,'reason':'SVM candidates first after currently running fits',
                    'reserved_at_unix':time.time(),'maximum_reservation_seconds':1800},stream)
                stream.flush();os.fsync(stream.fileno())
            # Register before publication; the main finally sees this list even on interruption.
            owned.append(entry['run_id'])
            try:os.link(temporary,target)
            except FileExistsError:pass
        finally:
            if temporary.exists():temporary.unlink()
    return owned


def release(root,owned,nonce):
    for rid in owned:
        path=root/REL/'claims'/(rid+'.json')
        lock=path.with_suffix('.recovering')
        try:lock.mkdir()
        except FileExistsError:continue
        try:
            try:record=read(path)
            except (json.JSONDecodeError,FileNotFoundError):continue
            if record and record.get('kind')=='temporary_queue_reservation' and record.get('nonce')==nonce:path.unlink(missing_ok=True)
        finally:lock.rmdir()


def stale_heartbeat(now,started,heartbeat):
    return now-heartbeat>30 if heartbeat is not None else now-started>60


def entries_for(root,manifest):
    entries=list(manifest['candidates'])
    for path in (root/REL/'selections').glob('*.json'):entries.extend(read(path)['confirmation_entries'])
    return entries


def watchdog(root,manifest,nonce):
    started=time.time();svms=[e for e in manifest['candidates'] if e['model_id']=='tfidf_svm']
    while True:
        status=read(root/REPORT/'svm-priority-status.json') or {}
        heartbeat=status.get('heartbeat_unix') if status.get('nonce')==nonce else None
        if svm_ready(root,svms) or stale_heartbeat(time.time(),started,heartbeat) or time.time()-started>1800:
            release(root,[e['run_id'] for e in entries_for(root,manifest)],nonce)
            save(root/REPORT/'svm-priority-watchdog.json',{'status':'checked_and_released','nonce':nonce,
                'svm_complete':svm_ready(root,svms),'at_unix':time.time()})
            return
        time.sleep(3)


def svm_ready(root,entries):
    for entry in entries:
        path=root/'runs'/entry['run_id']/'status.json';status=read(path)
        audit=read(root/REL/'solver-audits'/(entry['run_id']+'.json'))
        if (not status or status.get('status')!='complete' or not status.get('reload_verified') or not audit
                or not audit.get('solver_verified') or audit['status_sha256']!=hashlib.sha256(path.read_bytes()).hexdigest()):return False
    return True


def comparison(root,manifest):
    results=[]
    for ds,metric in [('banking77','macro_f1'),('go_emotions','macro_ap')]:
        row={'dataset_id':ds,'partition':'V','metric':metric}
        for model in ['tfidf_svm','distilbert']:
            values=[]
            for entry in manifest['candidates']:
                if entry['dataset_id']!=ds or entry['model_id']!=model:continue
                status=read(root/'runs'/entry['run_id']/'status.json')
                if status and status.get('status')=='complete':values.append((status['validation'][metric],entry))
            if values:
                value,entry=max(values,key=lambda pair:pair[0])
                row[model]={'value':value,'run_id':entry['run_id'],'C':entry['run']['hparams'].get('C'),
                    'lr':entry['run']['hparams'].get('lr'),'completed_candidates_considered':len(values)}
        results.append(row)
    return {'scope':'V_only_SVM_vs_completed_DistilBERT_candidates','official_test_predictions_generated':False,'rows':results}


def main():
    if not os.environ.get('SLURM_JOB_ID'):raise PermissionError('Controller must run in an existing compute allocation')
    root=Path('research-v3').resolve();manifest=read(root/REL/'manifest.json')
    nonce=os.environ.get('SVM_PRIORITY_NONCE') or uuid.uuid4().hex
    if '--watchdog' in sys.argv:return watchdog(root,manifest,nonce)
    entries=entries_for(root,manifest)
    svms=[e for e in manifest['candidates'] if e['model_id']=='tfidf_svm']
    owned=[];start=time.monotonic();out=root/REPORT/'svm-priority-status.json'
    def interrupted(signum,frame):raise InterruptedError('Priority step interrupted')
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    try:
        reserve(root,entries,nonce,os.environ['SLURM_JOB_ID'],owned=owned)
        save(out,{'status':'svm_first_active','reserved_run_ids':owned,'nonce':nonce,'job_id':os.environ['SLURM_JOB_ID'],
            'heartbeat_unix':time.time()})
        print({'event':'svm_priority_active','deferred_unstarted_neural_runs':len(owned)},flush=True)
        while time.monotonic()-start<1800:
            save(out,{'status':'svm_first_active','reserved_run_ids':owned,'nonce':nonce,
                'job_id':os.environ['SLURM_JOB_ID'],'heartbeat_unix':time.time()})
            if svm_ready(root,svms):
                result=comparison(root,manifest);save(root/REPORT/'SVM-DistilBERT-validation-comparison.json',result)
                save(out,{'status':'svm_complete_priority_released','reserved_run_ids':owned,'nonce':nonce})
                print(result,flush=True);return
            if any((read(root/'runs'/e['run_id']/'status.json') or {}).get('status') in {'failed','incomplete'} for e in svms):
                save(out,{'status':'svm_error_priority_released','nonce':nonce});return
            time.sleep(3)
        save(out,{'status':'reservation_timeout_released','nonce':nonce})
    finally:
        release(root,owned,nonce)


if __name__=='__main__':main()
