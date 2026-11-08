set -euo pipefail
P=/projects/yiyan002humanoid/textclass-overnight-20261107
cd "$P/repo"
"$P/envs/research/bin/python" - <<'PY'
import json,runpy,time
from pathlib import Path
root=Path('research-v3').resolve()
module=runpy.run_path(str(root/'training/convergence-v1/svm-priority.py'))
p=root/'reports/overnight/convergence-v1/svm-priority-status.json'
record=json.loads(p.read_text())
for _ in range(5):
    module['release'](root,record['reserved_run_ids'],record['nonce'])
    time.sleep(.1)
remaining=[]
for rid in record['reserved_run_ids']:
    path=root/'training/convergence-v1/claims'/(rid+'.json')
    if path.exists() and json.loads(path.read_text()).get('nonce')==record['nonce']:remaining.append(rid)
print({'failed_control_reservations_remaining':remaining})
assert not remaining
PY
srun --help | grep -e async -e 'wait-for' || true
srun --version
