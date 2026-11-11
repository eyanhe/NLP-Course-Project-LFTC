python3 - <<'PY'
import json, subprocess, time, os, re
from datetime import datetime, timezone
from pathlib import Path
r=Path('/projects/yiyan002humanoid/textclass-overnight-20261107/repo/research-v3')
p=json.loads((r/'training/overnight/run-plan.json').read_text())
counts={str(i):{'new_required':0,'complete':0,'failed':0,'running':0,'pending':0} for i in (0,1,2)}
finished=[]
running=[]
for row in p['runs']:
    if row['completed_checkpoint_reuse']:continue
    rec=counts[str(row['priority'])];rec['new_required']+=1
    q=r/'runs'/row['entry']['run_id']/'status.json'
    s=json.loads(q.read_text()) if q.exists() else {}
    state=s.get('status','pending')
    if state=='complete' and s.get('reload_verified'):
        rec['complete']+=1
        v=s.get('validation',{})
        finished.append({'run_id':row['entry']['run_id'],'dataset':row['dataset_id'],'model':row['model_id'],'n':row['training_rows'],'priority':row['priority'],'epochs':s.get('epochs_completed'),'seconds':s.get('total_seconds'),'macro_f1':v.get('macro_f1'),'macro_ap':v.get('macro_ap')})
    elif state in ('running','verifying'):
        rec['running']+=1
        progress=q.parent/'progress.json'
        running.append({'run_id':row['entry']['run_id'],'last_progress_age_seconds':round(time.time()-(progress.stat().st_mtime if progress.exists() else q.stat().st_mtime))})
    elif state=='failed':rec['failed']+=1
    else:rec['pending']+=1
queue=subprocess.run(['squeue','--start','--me','-h','-o','%i|%j|%T|%M|%R|%S|%V'],capture_output=True,text=True,check=True)
jobs=[]
for line in queue.stdout.splitlines():
    parts=line.split('|',6)
    if len(parts)==7:jobs.append(dict(zip(['job_id','name','state','elapsed','reason','predicted_start','submitted_at'],parts)))
extension=None
ep=r/'training/overnight/extension-run-plan.json'
if ep.exists():
    extra=json.loads(ep.read_text())
    extension={'new_required':extra['new_fits_maximum'],'complete':0,'failed':0,'running':0,'pending':0}
    for row in extra['runs']:
        if row['completed_checkpoint_reuse']:continue
        q=r/'runs'/row['entry']['run_id']/'status.json'
        s=json.loads(q.read_text()) if q.exists() else {}
        state=s.get('status','pending')
        if state=='complete' and s.get('reload_verified'):
            extension['complete']+=1
            v=s.get('validation',{})
            finished.append({'run_id':row['entry']['run_id'],'dataset':row['dataset_id'],'model':row['model_id'],'n':row['training_rows'],'priority':row['priority'],'scope':'extension','epochs':s.get('epochs_completed'),'seconds':s.get('total_seconds'),'macro_f1':v.get('macro_f1'),'macro_ap':v.get('macro_ap')})
        elif state in ('running','verifying'):extension['running']+=1
        elif state=='failed':extension['failed']+=1
        else:extension['pending']+=1
accounting=[]
ids=os.environ.get('TC_WATCHDOG_JOB_IDS','')
if re.fullmatch(r'[0-9]+(?:,[0-9]+)*',ids):
    history=subprocess.run(['sacct','-nP','-j',ids,'--format=JobIDRaw,State,ExitCode'],capture_output=True,text=True,timeout=15)
    if history.returncode==0:
        for line in history.stdout.splitlines():
            fields=line.strip().split('|')
            if len(fields)>=3 and fields[0].isdigit():accounting.append({'job_id':fields[0],'state':fields[1].split()[0],'exit_code':fields[2]})
print(json.dumps({'new_fit_counts':counts,'extension':extension,'completed':finished,'running_details':running,'jobs':jobs,'job_accounting':accounting,'recorded_at_utc':datetime.now(timezone.utc).isoformat(),'formal_Q_generated':False}))
PY
