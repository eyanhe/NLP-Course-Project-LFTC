set -euo pipefail
P=/projects/yiyan002humanoid/textclass-overnight-20261107
R="$P/repo/research-v3"
TZ=Asia/Singapore date -Is
squeue -u yiyan002 -o '%.12i %.29j %.10T %.7C %.12m %.10M %.24R'
TZ=Asia/Singapore sacct -P -X -j 278233,278234,278235,278242 --format=JobIDRaw,State,Start,End,ElapsedRaw,NodeList
tail -n 10 "$P/logs/tc-conv-finish-test-278235.out"
if [ -f "$P/logs/tc-conv-Q-test-only-278242.out" ]; then tail -n 8 "$P/logs/tc-conv-Q-test-only-278242.out"; fi
for file in training/convergence-v1/scoped-evaluation.freeze.json reports/overnight/convergence-v1/evaluation-started.json reports/overnight/convergence-v1/evaluation-blocked.json reports/overnight/convergence-v1/test-results.json; do
  if [ -f "$R/$file" ]; then ls -l "$R/$file"; fi
done
if [ -d "$R/reports/overnight/convergence-v1/predictions" ]; then
  find "$R/reports/overnight/convergence-v1/predictions" -maxdepth 1 -type f -name '*.manifest.json' | wc -l
fi
