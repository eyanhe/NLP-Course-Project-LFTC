set -euo pipefail
P=/projects/yiyan002humanoid/textclass-overnight-20261107
date -Is
for file in svm-priority-status.json svm-priority-watchdog.json SVM-DistilBERT-validation-comparison.json; do
  path="$P/repo/research-v3/reports/overnight/convergence-v1/$file"
  if test -f "$path"; then printf '%s\n' "$file"; cat "$path"; fi
done
tail -n 6 "$P/logs/svm-priority-controller.log"
tail -n 6 "$P/logs/svm-priority-watchdog.log"
squeue -u yiyan002 -o '%.12i %.30j %.10T %.10M %.28R'
