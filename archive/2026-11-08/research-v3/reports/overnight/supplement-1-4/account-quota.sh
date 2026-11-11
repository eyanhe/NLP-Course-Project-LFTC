date '+%Y-%m-%d %H:%M:%S %Z %z'
sshare -u yiyan002 -l -P
sshare -u yiyan002 -n -P -o Account,User,GrpTRESMins,GrpTRESRaw,TRESRunMins
sacctmgr -n -P show assoc where user=yiyan002 format=Cluster,Account,User,QOS,DefaultQOS,GrpTRESMins
sacctmgr -n -P show qos msc format=Name,MaxJobsPU,MaxSubmitPU,MaxTRESPU,MaxWall
scontrol show partition cluster02 -o
command -v storagemgr || true
command -v quota || true
squeue -u yiyan002 -o '%.18i %.30j %.10T %.12M %.24R'
