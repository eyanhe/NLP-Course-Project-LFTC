set -euo pipefail
P=/projects/yiyan002humanoid/textclass-overnight-20261107
F="$P/repo/research-v3/reports/overnight/convergence-v1"
TZ=UTC sacct -P -j 278148,278149,278233,278234,278235 --format=JobIDRaw,State,Start,End,ElapsedRaw,AllocCPUS,NodeList > "$F/training-job-timings.psv.part"
mv "$F/training-job-timings.psv.part" "$F/training-job-timings.psv"
cat "$F/training-job-timings.psv"
