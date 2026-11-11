set -euo pipefail
P=/projects/yiyan002humanoid/textclass-overnight-20261107
date -Is
squeue -u yiyan002 -o '%.12i %.30j %.10T %.10M %.10l %.28R'
sacct -nP -j 278148,278149 --format=JobIDRaw,JobName,State,Elapsed,ExitCode,NodeList
for pair in a:278148 b:278149; do
  worker=${pair%:*}
  job=${pair#*:}
  log="$P/logs/tc-conv-cpu-$worker-$job.out"
  if test -f "$log"; then printf 'log_job=%s\n' "$job"; tail -n 12 "$log"; fi
done
