set -euo pipefail
date -Is
squeue -u yiyan002 -o '%.12i %.29j %.10T %.7C %.12m %.10M %.24R'
for job in 278233 278234 278235; do
  scontrol show job "$job" -o | tr ' ' '\n' | sed -n '/^JobId=/p; /^JobState=/p; /^NumNodes=/p; /^NumCPUs=/p; /^CPUs\/Task=/p; /^NodeList=/p; /^ReqTRES=/p; /^AllocTRES=/p; /^Dependency=/p'
done
sinfo -p cluster02 -N -h -o '%N|%C' | grep '^cpu-'
