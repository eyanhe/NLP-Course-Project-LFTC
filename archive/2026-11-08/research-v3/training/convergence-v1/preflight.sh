set -euo pipefail
P=/projects/yiyan002humanoid/textclass-overnight-20261107
date -Is
squeue -u yiyan002 -o '%.12i %.30j %.10T %.10M %.10l %.28R'
sinfo -p cluster02 -N -o '%N|%T|%C|%G'
scontrol show nodes -o | awk '{for(i=1;i<=NF;i++) if($i ~ /^(NodeName|State|Gres|CfgTRES|AllocTRES)=/) printf "%s ",$i; print ""}'
sacctmgr -nP show assoc user=yiyan002 format=Account,QOS,DefaultQOS
sacctmgr -nP show qos msc format=Name,MaxJobsPU,MaxSubmitPU,MaxTRESPU,MaxWall
df -h "$P"
findmnt -T "$P" -no FSTYPE,TARGET
for name in evaluation-started.json selection.lock.json evaluation.freeze.json; do
  if test -e "$P/repo/research-v3/$name"; then printf 'existing_freeze_or_test_journal=%s\n' "$name"; fi
done
