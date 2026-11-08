"""Run the declared SVM grid first, then resume the unchanged original queue."""
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import runpy
import time
import uuid

from . import convergence as driver
from .storage import atomic_write_json,read_json


def claim_svm(root,entries,worker_id):
    folder=root/driver.REL/'claims';folder.mkdir(parents=True,exist_ok=True)
    for entry in entries:
        if entry['model_id']!='tfidf_svm':continue
        target=folder/(entry['run_id']+'.json')
        if target.exists():
            if 'expected_model_request_hash' not in entry:continue
            status=driver._status(root,entry)
            if status and status.get('status')=='complete' and status.get('solver_verified'):continue
            if not driver._recover_claim(root,target,entry,driver._slurm_state):continue
        temp=folder/('.svm-claim-'+uuid.uuid4().hex+'.tmp')
        try:
            descriptor=os.open(temp,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
            with os.fdopen(descriptor,'w',encoding='utf-8') as stream:
                json.dump({'worker_id':worker_id,'job_id':os.environ.get('SLURM_JOB_ID'),
                    'host':os.environ.get('HOSTNAME'),'run_id':entry['run_id'],
                    'claimed_at_utc':datetime.now(timezone.utc).isoformat(),'phase':'svm_first'},stream)
                stream.flush();os.fsync(stream.fileno())
            try:os.link(temp,target)
            except FileExistsError:continue
            return entry
        finally:temp.unlink(missing_ok=True)
    return None


def worker(root):
    if not os.environ.get('SLURM_JOB_ID'):raise PermissionError('Requires a Slurm compute allocation')
    root=Path(root).resolve();manifest=read_json(root/driver.REL/'manifest.json');driver.verify_manifest(root,manifest)
    entries=[e for e in manifest['candidates'] if e['model_id']=='tfidf_svm']
    if len(entries)!=6:raise ValueError('Expected six declared SVM C candidates including two existing models')
    helper=runpy.run_path(str(root/driver.REL/'svm-priority.py'))
    deadline=datetime.fromisoformat(manifest['start_cutoff_utc'])
    while datetime.now(timezone.utc)<datetime.fromisoformat(manifest['hard_stop_utc']):
        if helper['svm_ready'](root,entries):
            comparison=helper['comparison'](root,manifest)
            atomic_write_json(root/driver.REPORT/'SVM-DistilBERT-validation-comparison.json',comparison,root)
            atomic_write_json(root/driver.REPORT/'svm-priority-status.json',
                {'status':'svm_complete','worker_job_id':os.environ['SLURM_JOB_ID'],
                    'completed_at_utc':datetime.now(timezone.utc).isoformat()},root)
            print({'event':'svm_phase_complete','next':'resume_other_declared_training'},flush=True)
            break
        statuses=[driver._status(root,e) for e in entries]
        if any(s and s.get('status') in {'failed','incomplete'} for s in statuses):break
        entry=claim_svm(root,entries,os.environ['SLURM_JOB_ID']) if datetime.now(timezone.utc)<deadline else None
        if entry is None:
            time.sleep(2);continue
        print({'event':'svm_first_fit_start','run_id':entry['run_id'],'dataset':entry['dataset_id'],
            'C':entry['run']['hparams'].get('C')},flush=True)
        try:
            result=driver.fit_entry(root,entry,manifest)
            print({'event':'svm_first_fit_complete','run_id':entry['run_id'],
                'V_macro_f1':result['validation'].get('macro_f1'),
                'V_macro_ap':result['validation'].get('macro_ap')},flush=True)
        except Exception as exc:
            atomic_write_json(root/driver.REPORT/'worker-errors'/(entry['run_id']+'.json'),
                {'run_id':entry['run_id'],'status':'failed','error':repr(exc)},root)
            print({'event':'svm_first_error','run_id':entry['run_id'],'error':repr(exc)},flush=True)
    # Completed neural models are reused; interrupted models resume authenticated optimizer snapshots.
    driver.worker(root,'cpu')


if __name__=='__main__':worker(Path('research-v3'))
