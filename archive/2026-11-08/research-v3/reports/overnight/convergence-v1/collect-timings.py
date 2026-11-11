"""Compile logged training stages and measured Q inference without running models."""
from datetime import datetime,timezone
import csv
import hashlib
import json
from pathlib import Path

FOLDER=Path(__file__).resolve().parent
PROJECT=FOLDER.parents[3]
ROOT=PROJECT/'research-v3'
SNAPSHOT=FOLDER/'cluster-snapshot'
REL=Path('training/convergence-v1')
REPORT=Path('reports/overnight/convergence-v1')


def read(path):
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else None


def stamp(value):
    try:
        parsed=datetime.fromisoformat(value)
        return parsed.replace(tzinfo=timezone.utc).timestamp() if parsed.tzinfo is None else parsed.timestamp()
    except (TypeError,ValueError):return None


def training_wall_time(status,leases,claims,jobs):
    if status.get('status')!='complete' or status.get('total_seconds') is None:return None,None,'not_finished'
    final=status['total_seconds']
    if not leases:
        return (None,None,'resume_history_identity_missing') if claims else (final,0.,'logged_total')
    if len(leases)!=len(claims):return None,None,'resume_previous_duration_unknown'
    prior=0.;used=set()
    for lease in sorted(leases,key=lambda r:r['started_at']):
        if lease.get('request_hash')!=status.get('request_hash'):return None,None,'resume_history_identity_missing'
        available=[(abs((stamp(c.get('claimed_at_utc')) or -1e20)-lease['started_at']),i,c)
            for i,c in enumerate(claims) if i not in used]
        if not available:return None,None,'resume_previous_duration_unknown'
        distance,index,claim=min(available,key=lambda x:x[0]);job=jobs.get(str(claim.get('job_id')), {})
        ended=stamp(job.get('End'))
        if (distance>120 or ended is None or ended<lease['started_at']
                or job.get('State','').split()[0] not in {'CANCELLED','FAILED','TIMEOUT','NODE_FAIL','OUT_OF_MEMORY','PREEMPTED'}):
            return None,None,'resume_previous_duration_unknown'
        used.add(index);prior+=ended-lease['started_at']
    return prior+final,prior,'scheduler_lease_reconstruction'


def main():
    manifest=read(ROOT/REL/'manifest.json');proposal=read(ROOT/REL/'proposal.json')
    entries=list(manifest['candidates'])
    for path in sorted((SNAPSHOT/REL/'selections').glob('*.json')):entries.extend(read(path)['confirmation_entries'])
    for spec in proposal['scope']:
        for model in ['bilstm','distilbert']:
            for seed in [43,44]:
                if any(e['dataset_id']==spec['dataset_id'] and e['model_id']==model and e['run']['training_seed']==seed for e in entries):continue
                entries.append({'dataset_id':spec['dataset_id'],'model_id':model,'training_rows':spec['training_rows'],
                    'run_id':None,'phase':'confirmation','run':{'training_seed':seed,'hparams':{}}})
    for ref in proposal['fixed_budget_references']:
        source=next(b for b in proposal['source_bindings'] if b['source_run_id']==ref['run_id'])
        entries.append({**ref,'run':source['source_run'],'training_rows':source['training_rows'],'phase':'reference'})
    job_file=SNAPSHOT/REPORT/'training-job-timings.psv';jobs={}
    if job_file.exists():
        with job_file.open(encoding='utf-8',newline='') as stream:
            jobs={r['JobIDRaw']:r for r in csv.DictReader(stream,delimiter='|') if r['JobIDRaw'].isdigit()}
    result=read(SNAPSHOT/REPORT/'test-results.json') or {};inferences={r['run_id']:r for r in result.get('models',[])}
    freeze=read(SNAPSHOT/REL/'scoped-evaluation.freeze.json');tested_ids={e['run_id'] for e in freeze['models']} if freeze else None
    rows=[];pins={}
    def source(path):
        if path.exists():pins[str(path.relative_to(PROJECT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    for entry in entries:
        rid=entry['run_id'];directory=SNAPSHOT/'runs'/rid if rid else None
        path=directory/'status.json' if directory else None
        if path is not None and not path.exists():
            alternatives=[ROOT/'runs'/rid/'status.json',ROOT/'reports/overnight/analysis-handoff/cluster-evidence/runs'/rid/'status.json']
            path=next((p for p in alternatives if p.exists()),path)
        status=read(path) if path is not None else None
        status=status or {};source(path) if path is not None else None
        leases=[];claims=[]
        if directory is not None:
            for p in directory.glob('.lease.history.*.json'):leases.append(read(p));source(p)
            for p in (SNAPSHOT/REL/'claims').glob(rid+'.abandoned.*.json'):claims.append(read(p));source(p)
        total,prior,kind=training_wall_time(status,leases,claims,jobs)
        hp=entry['run']['hparams'];model=entry['model_id'];seed=entry['run'].get('training_seed')
        phase=entry.get('phase','grid');reuse=rid is not None and rid==entry.get('source_run_id')
        role='历史固定预算参考' if phase=='reference' else '稳定性复跑' if phase=='confirmation' else '旧SVM复用' if reuse else 'C值网格' if model=='tfidf_svm' else '学习率网格'
        inference=inferences.get(rid,{});timing=inference.get('timing',{})
        infer_state='measured' if timing.get('predict_seconds') is not None else 'not_recorded' if inference else 'not_in_frozen_scope' if tested_ids is not None and rid not in tested_ids else 'not_started'
        notes=[]
        if model!='tfidf_svm':notes.append('训练阶段含每轮验证与模型保存')
        if kind=='scheduler_lease_reconstruction':notes.append('累计墙钟按中断前lease与Slurm结束时间重建，含终止清理')
        if kind.startswith('resume_'):notes.append('续跑前时长不能完整核实，累计值留空')
        if reuse:notes.append('复用历史训练，未重复拟合')
        if entry['run_id'] is None:notes.append('学习率待验证集选参后确定')
        if infer_state=='not_started':notes.append('正式推理尚未执行')
        if infer_state=='not_in_frozen_scope':notes.append('未列入最终测试模型清单')
        current_claim=read(SNAPSHOT/REL/'claims'/(rid+'.json')) if rid else None
        job=jobs.get(str((current_claim or {}).get('job_id')), {})
        state=status.get('status','not_started')
        if state in {'running','verifying'} and job.get('State','').startswith(('CANCELLED','FAILED','TIMEOUT')):state='awaiting_resume'
        rows.append({'run_id':rid,'case_id':rid or f"planned/{entry['dataset_id']}/{model}/{seed}",
            'dataset_id':entry['dataset_id'],'model_id':model,'training_rows':entry['training_rows'],
            'role':role,'parameter':hp.get('C',1) if model=='tfidf_svm' else hp.get('lr'),
            'training_seed':seed,'status':state,'epochs_completed':None if model=='tfidf_svm' else status.get('epochs_completed'),
            'fit_seconds_last_segment':status.get('fit_seconds'),'feature_seconds_last_segment':status.get('feature_seconds'),
            'pipeline_seconds_last_segment':status.get('total_seconds'),'prior_interrupted_wall_seconds':prior,
            'training_pipeline_wall_seconds':total,'training_time_source':kind,
            'training_cpu_threads':hp.get('cpu_threads'),'training_device':'cpu' if model=='tfidf_svm' else hp.get('device'),
            'training_node_last_segment':None if reuse else job.get('NodeList'),'inference_status':infer_state,
            'verification_bundle_seconds':timing.get('verification_bundle_seconds'),'model_load_seconds':timing.get('model_load_seconds'),
            'predict_seconds':timing.get('predict_seconds'),'inference_cleanup_seconds':timing.get('cleanup_seconds'),
            'authenticated_inference_seconds':timing.get('authenticated_inference_seconds'),'test_rows':timing.get('rows'),
            'rows_per_second':timing.get('rows_per_second'),'milliseconds_per_row':timing.get('milliseconds_per_row'),
            'inference_cpu_threads':timing.get('cpu_threads'),'inference_node':timing.get('node'),
            'notes':'；'.join(notes)})
    rows.sort(key=lambda r:(r['dataset_id'],{'tfidf_svm':0,'bilstm':1,'distilbert':2}[r['model_id']],
        r['role']=='历史固定预算参考',r['role']=='稳定性复跑',r['parameter'] if r['parameter'] is not None else 1e20,r['training_seed'] or 0))
    source(job_file)
    for p in [ROOT/REL/'manifest.json',ROOT/REL/'proposal.json',SNAPSHOT/REPORT/'test-results.json']:source(p)
    payload={'recorded_at_utc':datetime.now(timezone.utc).isoformat(),'rows':rows,'source_sha256':pins,
        'definitions':{'training_pipeline_wall_seconds':'Feature preparation, fitting, per-epoch V, checkpoint writes, and reload verification. Resumed rows add prior interrupted process-wall segments where scheduler/lease evidence permits.',
            'fit_seconds_last_segment':'Logged fitting span only; neural includes per-epoch validation and saving, SVM includes its presave prediction probe. Not a pure forward/backward timer.',
            'predict_seconds':'Text vectorization/tokenization and forward-to-CPU-scores. Excludes model loading, file hashing, Q source loading, metrics, and prediction storage.',
            'missing':'Unmeasured or unverified timings remain null; cached predictions retain the original measured inference time.'}}
    path=FOLDER/'training-inference-timings.json';temp=path.with_suffix('.json.part');temp.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');temp.replace(path)
    print(json.dumps({'records':len(rows),'completed':sum(r['status']=='complete' for r in rows),
        'resumed_reconstructed':sum(r['training_time_source']=='scheduler_lease_reconstruction' for r in rows),
        'measured_inferences':sum(r['predict_seconds'] is not None for r in rows)},ensure_ascii=False))
    return payload


if __name__=='__main__':main()
