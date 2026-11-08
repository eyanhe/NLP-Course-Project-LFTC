set -euo pipefail
P=/projects/yiyan002humanoid/textclass-overnight-20261107
R="$P/repo/research-v3"
TZ=Asia/Singapore date -Is
squeue -u yiyan002 -o '%.12i %.25j %.10T %.7C %.12m %.10M %.24R'
TZ=Asia/Singapore sacct -P -X -j 278262,278263 --format=JobIDRaw,State,Start,End,ElapsedRaw,NodeList
for stage in S3 S1; do
  job=$(cat "$R/training/supplement-q-v1/job-$stage.txt")
  printf '%s %s\n' "$stage" "$job"
  if [ -f "$P/logs/tc-$stage-Q-$job.out" ]; then tail -n 5 "$P/logs/tc-$stage-Q-$job.out"; fi
  for name in progress.json evaluation-failed.json; do
    if [ -f "$R/reports/overnight/supplement-q-v1/$stage/$name" ]; then cat "$R/reports/overnight/supplement-q-v1/$stage/$name"; fi
  done
  if [ -f "$R/reports/overnight/supplement-q-v1/$stage/test-results.json" ]; then ls -l "$R/reports/overnight/supplement-q-v1/$stage/test-results.json"; fi
done
