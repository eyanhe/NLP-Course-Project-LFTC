"""Stop only the identified local legacy auto-submit watchdog; leave Slurm jobs intact."""
from datetime import datetime,timezone
import json
from pathlib import Path
import psutil

project=Path(__file__).resolve().parents[4]
heartbeat=project/'research-v3/training/overnight/watchdog-heartbeat.json'
record=json.loads(heartbeat.read_text(encoding='utf-8'));pid=record['pid']
expected=(project/'research-v3/training/overnight/watchdog.py').resolve()
try:
    process=psutil.Process(pid)
    command=process.cmdline()
    if not any(Path(arg).resolve()==expected for arg in command[1:] if arg.endswith('watchdog.py')):
        raise RuntimeError('PID does not match the legacy project watchdog')
    process.terminate();process.wait(timeout=10)
    status='terminated_identified_legacy_auto_submitter'
except psutil.NoSuchProcess:status='already_stopped'
event=dict(pid=pid,status=status,checked_at_utc=datetime.now(timezone.utc).isoformat(),
           reason='User prioritized supplement1-4; avoid automatic unrelated P2/extension submissions. Existing Slurm jobs unchanged.')
Path(__file__).with_name('legacy-auto-status.json').write_text(json.dumps(event,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(event))
