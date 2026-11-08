set -eu
prep=$(sbatch --parsable /projects/yiyan002humanoid/textclass-overnight-20261107/jobs/p1-prep.sbatch)
printf "CPU_PREP_JOB=%s\n" "$prep"
sbatch --parsable --dependency="afterok:$prep" /projects/yiyan002humanoid/textclass-overnight-20261107/jobs/p1-a.sbatch
sbatch --parsable --dependency="afterok:$prep" /projects/yiyan002humanoid/textclass-overnight-20261107/jobs/p1-b.sbatch
