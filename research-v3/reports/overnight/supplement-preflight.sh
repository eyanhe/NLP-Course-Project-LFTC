date -u
squeue -u yiyan002 -o '%.18i %.32j %.10T %.12M %.10l %.20R'
sinfo -N -o '%N %t %C %G'
sacctmgr -n show assoc where user=yiyan002 format=Account,QOS
ls /projects/yiyan002humanoid/textclass-overnight-20261107/jobs
cat /projects/yiyan002humanoid/textclass-overnight-20261107/jobs/p1-yelp-emotion.sbatch 2>/dev/null || true
