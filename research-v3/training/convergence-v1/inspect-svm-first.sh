set -euo pipefail
P=/projects/yiyan002humanoid/textclass-overnight-20261107
F="$P/repo/research-v3/training/convergence-v1"
date -Is
squeue -u yiyan002 -o '%.12i %.30j %.10T %.10M %.28R'
for worker in a b; do
  job=$(cat "$F/job-svm-first-${worker}.txt")
  printf 'svm_first_worker=%s job=%s\n' "$worker" "$job"
  log="$P/logs/tc-conv-svm-first-${worker}-${job}.out"
  if test -f "$log"; then tail -n 10 "$log"; fi
done
report="$P/repo/research-v3/reports/overnight/convergence-v1/SVM-DistilBERT-validation-comparison.json"
if test -f "$report"; then cat "$report"; fi
