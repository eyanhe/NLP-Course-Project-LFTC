set -euo pipefail
P=/projects/yiyan002humanoid/textclass-overnight-20261107
cd "$P/repo"
F=research-v3/reports/overnight/supplement-1-4
exec 9>"$F/submit.lock"
flock -n 9
for worker in bank emotion; do
  if [ ! -s "$F/job-${worker}.txt" ]; then
    sbatch --parsable "$F/cpu-${worker}.sbatch" > "$F/job-${worker}.txt"
  fi
  printf '%s ' "$worker"
  cat "$F/job-${worker}.txt"
done
squeue -u yiyan002 -o '%.18i %.32j %.10T %.12M %.10l %.20R'
