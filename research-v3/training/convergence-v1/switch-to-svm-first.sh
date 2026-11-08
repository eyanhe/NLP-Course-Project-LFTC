set -euo pipefail
umask 077
P=/projects/yiyan002humanoid/textclass-overnight-20261107
cd "$P/repo"
F=research-v3/training/convergence-v1
printf '%s  %s\n' '1c8df9d70f468cae69c17d454cb94634780cf7e425a44b86aaa539d662fbd12c' "$P/jobs/svm-first-workers.tar" | sha256sum -c -
exec 9>"$F/svm-first-switch.lock"
flock -n 9
if test -s "$F/job-svm-first-a.txt"; then cat "$F/job-svm-first-a.txt" "$F/job-svm-first-b.txt"; exit 0; fi
test ! -e "$F/scoped-evaluation.freeze.json"
test ! -e research-v3/reports/overnight/convergence-v1/evaluation-started.json
"$P/envs/research/bin/python" - <<'PY'
import json
from pathlib import Path
r=Path('research-v3');f=r/'training/convergence-v1'
for rid in ['e2cf73ef175e2ecd3cbd9d87c824838f47afdf457371a634cdb1293a9bedd1c6','260f965025b77c30d56636f5fe2fdca9930d06144fe0eaa68f50188407197ae7']:
    s=json.loads((r/'runs'/rid/'status.json').read_text());assert s['status']=='complete' and s['reload_verified']
m=json.loads((f/'manifest.json').read_text());entries=list(m['candidates'])
for p in (f/'selections').glob('*.json'):entries.extend(json.loads(p.read_text())['confirmation_entries'])
protected=[]
for e in entries:
    p=r/'runs'/e['run_id']/'status.json'
    if not p.exists():continue
    s=json.loads(p.read_text())
    if s['status'] not in ['running','verifying']:continue
    assert e['model_id']!='tfidf_svm','Do not interrupt an already running SVM'
    info=json.loads((p.parent/'resume.manifest.json').read_text())
    assert info['request_hash']==s['request_hash'] and (p.parent/info['snapshot']).is_file()
    protected.append(e['run_id'])
print({'GoEmotions_requested_repeats_complete':True,'other_neural_runs_with_resume_points':protected})
PY
tar -xf "$P/jobs/svm-first-workers.tar" -C "$P/repo"
sbatch --test-only "$F/svm-first-a.sbatch"
sbatch --test-only "$F/svm-first-b.sbatch"
test "$(squeue -h -j 278188 -o '%T')" = PENDING
scancel --state=PENDING 278188
test -z "$(squeue -h -j 278188 -o '%T')"
cp "$F/job-evaluate.txt" "$F/job-evaluate-before-svm-first.txt"
scancel 278148 278149
for worker in a b; do
  cp "$F/job-cpu-${worker}.txt" "$F/job-cpu-${worker}-before-svm-first.txt"
  sbatch --parsable "$F/svm-first-${worker}.sbatch" > "$F/job-svm-first-${worker}.txt"
  cp "$F/job-svm-first-${worker}.txt" "$F/job-cpu-${worker}.txt"
done
a=$(cat "$F/job-svm-first-a.txt")
b=$(cat "$F/job-svm-first-b.txt")
sbatch --parsable --dependency="afterany:$a:$b" "$F/finish-and-evaluate.sbatch" > "$F/job-evaluate.txt"
printf 'svm_first_jobs=%s,%s\n' "$a" "$b"
printf 'subsequent_test_job='
cat "$F/job-evaluate.txt"
date -Is
squeue -u yiyan002 -o '%.12i %.30j %.10T %.10M %.28R'
