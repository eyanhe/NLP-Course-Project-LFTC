set -euo pipefail
P=/projects/yiyan002humanoid/textclass-overnight-20261107
printf '%s  %s\n' 'fa29294f45118dbbdd43686f12250b66ad25bd68afccfdafa89cb2606e7a9bb6' "$P/jobs/supplement-runtime-v2.tar" | sha256sum -c -
tar -xf "$P/jobs/supplement-runtime-v2.tar" -C "$P/repo"
cd "$P/repo"
F=research-v3/reports/overnight/supplement-1-4
exec 9>"$F/submit.lock"
flock -n 9
for worker in bank emotion; do
  if [ ! -s "$F/job-${worker}-v2.txt" ]; then
    old=$(cat "$F/job-${worker}.txt")
    sbatch --parsable --dependency="afterany:${old}" "$F/cpu-${worker}.sbatch" > "$F/job-${worker}-v2.txt"
  fi
  printf '%s_v2 ' "$worker"
  cat "$F/job-${worker}-v2.txt"
done
squeue -u yiyan002 -o '%.18i %.32j %.10T %.12M %.10l %.20R'
