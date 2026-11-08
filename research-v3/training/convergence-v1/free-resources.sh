set -euo pipefail
date -Is
scontrol show nodes -o | awk '/Gres=gpu:/ {name="";cfg=0;alloc=0; for(i=1;i<=NF;i++){if($i ~ /^NodeName=/)name=$i; if($i ~ /^CfgTRES=/){n=split($i,a,",");for(j=1;j<=n;j++)if(a[j] ~ /^gres\/gpu=/){split(a[j],v,"=");cfg=v[2]}} if($i ~ /^AllocTRES=/){n=split($i,a,",");for(j=1;j<=n;j++)if(a[j] ~ /^gres\/gpu=/){split(a[j],v,"=");alloc=v[2]}}} if(cfg>alloc)print name,"available_gpus="cfg-alloc; total+=cfg;used+=alloc} END{print "total_gpus="total,"allocated_gpus="used,"available_gpus="total-used}'
sinfo -p cluster02 -N -h -o '%N|%C' | awk '/^cpu-/'
squeue -u yiyan002 -h -o '%i|%j|%T|%R'
