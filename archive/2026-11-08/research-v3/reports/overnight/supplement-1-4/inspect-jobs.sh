P=/projects/yiyan002humanoid/textclass-overnight-20261107
F="$P/repo/research-v3/reports/overnight/supplement-1-4"
date -u
squeue -u yiyan002 -o '%.18i %.32j %.10T %.12M %.10l %.20R'
for worker in bank emotion; do
  job=$(cat "$F/job-${worker}.txt")
  scontrol show job "$job"
  tail -c 1800 "$P/logs/tc-sup14-${worker}-${job}.out" 2>/dev/null || true
done
