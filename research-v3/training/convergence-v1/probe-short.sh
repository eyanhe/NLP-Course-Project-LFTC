set -euo pipefail
P=/projects/yiyan002humanoid/textclass-overnight-20261107
date -Is
squeue -u yiyan002 -h -o '%i|%j|%T|%R'
scontrol show node gpu-pro6000-11 -o | awk '{for(i=1;i<=NF;i++) if($i ~ /^(NodeName|State|CfgTRES|AllocTRES)=/) printf "%s ",$i; print ""}'
"$P/envs/research/bin/python" -c 'from importlib.metadata import version; print("torch="+version("torch")); print("transformers="+version("transformers"))'
