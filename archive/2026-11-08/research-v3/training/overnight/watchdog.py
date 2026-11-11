"""Local SSH watchdog: durable heartbeat and once-only authorized P2 continuation."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

FOLDER = Path(__file__).resolve().parent
PROJECT = FOLDER.parents[2]
SSH = PROJECT/'research-v3/training/cluster-migration/ssh_ops.py'
START_CUTOFF = datetime.fromisoformat('2026-11-08T09:00:00+08:00')
HARD_STOP = datetime.fromisoformat('2026-11-08T10:00:00+08:00')
FAILED_STATES = {'FAILED','OUT_OF_MEMORY','TIMEOUT','NODE_FAIL','BOOT_FAIL','PREEMPTED','CANCELLED','DEADLINE'}


def next_action(snapshot, p2_submitted, now, extension_required=0, extension_submitted=False):
    if now >= HARD_STOP:
        return 'deadline_stop'
    counts = snapshot['new_fit_counts']
    if any(job['state'] in FAILED_STATES for job in snapshot.get('job_accounting',[])):
        return 'failed'
    if any(row['failed'] for row in counts.values()) or (snapshot.get('extension') or {}).get('failed',0):
        return 'failed'
    if all(counts[str(i)]['complete'] == expected for i, expected in enumerate((32, 9, 16))):
        if extension_required and (snapshot.get('extension') or {}).get('complete',0) < extension_required:
            return 'launch_extension' if not extension_submitted and now < START_CUTOFF else 'monitor'
        return 'complete'
    if counts['0']['complete'] == 32 and counts['1']['complete'] == 9 and not p2_submitted and now < START_CUTOFF:
        return 'launch_p2'
    return 'monitor'


def queue_alerts(jobs,now):
    alerts=[]
    for job in jobs:
        if job['state']!='PENDING':continue
        try:
            start=datetime.fromisoformat(job['predicted_start']).replace(tzinfo=timezone.utc)
            if start>=HARD_STOP:
                alerts.append({'kind':'queue_start_after_deadline',**job})
        except (KeyError,ValueError):
            pass
        try:
            submitted=datetime.fromisoformat(job['submitted_at']).replace(tzinfo=timezone.utc)
            if (now-submitted).total_seconds()>=1800:
                alerts.append({'kind':'queued_over_30_minutes',**job})
        except (KeyError,ValueError):
            pass
    return alerts


def save_json(path, value):
    temp = path.with_suffix(path.suffix+'.part')
    for attempt in range(10):
        try:
            temp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
            temp.replace(path)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(.1)


def remote_file(path, watch_ids=None):
    if watch_ids:
        ids=sorted(set(str(value) for value in watch_ids))
        if not all(re.fullmatch('[0-9]+',value) for value in ids):
            raise ValueError('Invalid monitored job identity')
        wrapped=FOLDER/'watchdog-probe.sh'
        wrapped.write_text('export TC_WATCHDOG_JOB_IDS='+','.join(ids)+'\n'+path.read_text(encoding='utf-8'),encoding='utf-8')
        path=wrapped
    result = subprocess.run([sys.executable, '-X', 'utf8', str(SSH), '--command-file', str(path)], cwd=PROJECT, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=90)
    if result.returncode:
        message = (result.stdout+'\n'+result.stderr)[-3000:]
        raise RuntimeError(message)
    for line in reversed(result.stdout.splitlines()):
        if line.startswith('{'):
            return json.loads(line)
    raise RuntimeError('Remote response did not contain its expected JSON status')


def log_event(kind, **fields):
    record = {'time_utc':datetime.now(timezone.utc).isoformat(), 'event':kind, **fields}
    with (FOLDER/'watchdog-events.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(record, ensure_ascii=False)+'\n')
    print(json.dumps(record, ensure_ascii=False), flush=True)


def write_live_report(snapshot, state):
    counts = snapshot['new_fit_counts']
    lines = ['# 集群训练实时状态', '', '由独立看门狗每60秒更新。', '', f"最近检查UTC：{state['checked_at_utc']}", f"看门狗PID：{os.getpid()}", '', '| 阶段 | 完成 | 计划新增 | 失败 |', '| --- | ---: | ---: | ---: |']
    for i, expected in enumerate((32, 9, 16)):
        row = counts[str(i)]
        lines.append(f"| P{i} | {row['complete']} | {expected} | {row['failed']} |")
    if snapshot.get('extension'):
        row=snapshot['extension']
        lines.append(f"| 补充批次 | {row['complete']} | {row['new_required']} | {row['failed']} |")
    lines.extend(['', '正式测试集预测仍未开放，人工复核保持待完成。', '', 'GPU任务由Slurm运行；看门狗在本机后台监测并接续阶段，需要本机和VPN在线。'])
    (PROJECT/'research-v3/reports/overnight/live-progress.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def main():
    import msvcrt
    lock = (FOLDER/'watchdog.lock').open('a+b')
    if lock.tell() == 0:
        lock.write(b'1'); lock.flush()
    lock.seek(0)
    try:
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        raise RuntimeError('Another overnight watchdog already holds the process lock')
    state_path = FOLDER/'watchdog-heartbeat.json'
    previous = json.loads(state_path.read_text(encoding='utf-8')) if state_path.exists() else {}
    state = {'pid':os.getpid(), 'started_at_utc':datetime.now(timezone.utc).isoformat(), 'iteration':0, 'status':'starting', 'p2_jobs':previous.get('p2_jobs'), 'extension_jobs':previous.get('extension_jobs'), 'known_jobs':previous.get('known_jobs',['277766','277767','277768']), 'consecutive_connection_failures':0}
    log_event('watchdog_started', pid=os.getpid(), interval_seconds=60)
    while True:
        now = datetime.now(timezone.utc)
        state.update(checked_at_utc=now.isoformat(), iteration=state['iteration']+1)
        try:
            replacements=FOLDER/'p1-replacement.json'
            if replacements.exists():
                current=json.loads(replacements.read_text(encoding='utf-8'))
                state['known_jobs']=[value for value in state['known_jobs'] if value not in current['superseded_job_ids']]
                for value in current['job_ids']:
                    if value not in state['known_jobs']:state['known_jobs'].append(value)
            prep_record=FOLDER/'extension-preparation.json'
            if prep_record.exists():
                prep_info=json.loads(prep_record.read_text(encoding='utf-8'))
                old_id=str(prep_info.get('superseded_job_id',''))
                state['known_jobs']=[value for value in state['known_jobs'] if value!=old_id]
                prep_id=str(prep_info['job_id'])
                if re.fullmatch('[0-9]+',prep_id) and prep_id not in state['known_jobs']:
                    state['known_jobs'].append(prep_id)
            watch_ids=state['known_jobs']+list((state.get('p2_jobs') or {}).values())+list((state.get('extension_jobs') or {}).values())
            snapshot = remote_file(FOLDER/'read-progress.sh',watch_ids)
            save_json(FOLDER/'cluster-progress.json', snapshot)
            state['consecutive_connection_failures'] = 0
            state.pop('error', None)
            ep=FOLDER/'extension-run-plan.json'
            required=json.loads(ep.read_text(encoding='utf-8'))['new_fits_maximum'] if ep.exists() else 0
            action = next_action(snapshot, bool(state.get('p2_jobs')), now, required, bool(state.get('extension_jobs')))
            state['status'] = action
            state['counts'] = snapshot['new_fit_counts']
            state['extension'] = snapshot.get('extension')
            alerts = queue_alerts(snapshot.get('jobs',[]),now)
            for row in snapshot.get('running_details', []):
                if row['last_progress_age_seconds'] > 900:
                    alerts.append({'kind':'no_progress_for_15_minutes', **row})
            for job in snapshot.get('jobs', []):
                if 'DependencyNeverSatisfied' in job.get('reason', ''):
                    alerts.append({'kind':'failed_dependency', **job})
            if action == 'failed':
                alerts.append({'kind':'fit_or_scheduler_job_failed', 'counts':state['counts'],'job_accounting':snapshot.get('job_accounting',[])})
            if not snapshot.get('jobs') and action == 'monitor':
                alerts.append({'kind':'unfinished_scope_has_no_scheduled_jobs', 'counts':state['counts']})
            state['alerts'] = alerts
            if alerts:
                save_json(FOLDER/'watchdog-alert.json', {'checked_at_utc':now.isoformat(), 'alerts':alerts})
                log_event('attention_required', alerts=alerts)
            if action == 'launch_p2':
                log_event('p1_complete_submitting_p2')
                jobs = remote_file(FOLDER/'submit-p2-once.sh')
                if not all(re.fullmatch(r'[0-9]+', str(value)) for value in jobs.values()):
                    raise ValueError('Unexpected submitted job identifier')
                state['p2_jobs'] = jobs
                state['status'] = 'p2_submitted'
                log_event('p2_submitted', jobs=jobs)
            elif action == 'launch_extension':
                jobs=remote_file(FOLDER/'submit-extension-once.sh')
                if jobs.get('waiting_for'):
                    state['status']='waiting_for_extension_preparation'
                else:
                    if not all(re.fullmatch(r'[0-9]+',str(value)) for value in jobs.values()):
                        raise ValueError('Unexpected extension job identifier')
                    state['extension_jobs']=jobs
                    state['status']='extension_submitted'
                    log_event('extension_submitted',jobs=jobs)
            elif action == 'deadline_stop':
                known = set(state['known_jobs']) | set((state.get('p2_jobs') or {}).values()) | set((state.get('extension_jobs') or {}).values())
                live = [job['job_id'] for job in snapshot.get('jobs', []) if job['job_id'] in known and re.fullmatch('[0-9]+', job['job_id'])]
                if live:
                    command = FOLDER/'stop-expired-jobs.sh'
                    command.write_text('scancel '+' '.join(live)+'\nprintf \'{"stopped":true}\\n\'\n', encoding='utf-8')
                    remote_file(command)
                log_event('deadline_reached', stopped_jobs=live)
            save_json(state_path, state)
            write_live_report(snapshot, state)
            log_event('checked', action=state['status'], counts=state['counts'])
            if action in {'complete', 'deadline_stop'}:
                return
        except Exception as exc:
            state['consecutive_connection_failures'] += 1
            state['status'] = 'connection_or_control_error'
            state['error'] = str(exc)
            save_json(state_path, state)
            save_json(FOLDER/'watchdog-alert.json', {'checked_at_utc':now.isoformat(), 'error':str(exc)})
            log_event('check_failed', error=str(exc), failures=state['consecutive_connection_failures'])
            if 'AuthenticationException' in str(exc) or 'Authentication failed' in str(exc) or 'fingerprint mismatch' in str(exc):
                return
            if now >= HARD_STOP:
                return
        time.sleep(60)


if __name__ == '__main__':
    main()
