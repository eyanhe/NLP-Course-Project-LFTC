set -euo pipefail
P=/projects/yiyan002humanoid/textclass-overnight-20261107
date -Is
squeue -j 278148,278149 -h -o '%i|%j|%T|%M|%N'
for rid in 053930b319299c3e4c5707f082d7cb5c6a655b7025d936cc06d18e42a3160397 3a78df640b4f17311ada7ec2a9907b0fa8c473b3ca3c2325f0d83715937576f9; do
  printf 'run=%s\n' "$rid"
  if test -f "$P/repo/research-v3/runs/$rid/progress.json"; then cat "$P/repo/research-v3/runs/$rid/progress.json"; fi
done
