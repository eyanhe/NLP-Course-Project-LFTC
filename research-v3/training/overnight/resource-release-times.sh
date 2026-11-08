python3 - <<'PY'
import subprocess,json
from collections import defaultdict
out=subprocess.check_output(['squeue','-h','-p','cluster02','-t','R','-o','%b|%e'],text=True)
rows=defaultdict(list)
for line in out.splitlines():
    parts=line.rsplit('|',1)
    if len(parts)!=2:continue
    resource,end=parts
    for model in ('a40','a6000','6000ada','l40'):
        if 'gpu:'+model+':' in resource:rows[model].append(end)
print(json.dumps({model:{'running_gpu_job_count':len(times),'earliest_expected_ends':sorted(times)[:4]} for model,times in rows.items()}))
print(subprocess.check_output(['date','-Is'],text=True).strip())
PY
