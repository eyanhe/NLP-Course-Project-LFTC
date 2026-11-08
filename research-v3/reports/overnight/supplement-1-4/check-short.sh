set -euo pipefail
P=/projects/yiyan002humanoid/textclass-overnight-20261107/repo
cd "$P"
for budget in 00:30:00 01:00:00 02:00:00; do
  printf '%s\n' "$budget"
  sbatch --test-only --time="$budget" research-v3/reports/overnight/supplement-1-4/cpu-bank.sbatch
done
squeue -u yiyan002 -o '%.18i %.32j %.10T %.12M %.10l %.20R'
