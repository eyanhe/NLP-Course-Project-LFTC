set -eu
prep=$(sbatch --parsable /projects/yiyan002humanoid/textclass-overnight-20261107/jobs/other-prep.sbatch)
printf 'CPU_PREP_JOB=%s\n' "$prep"
sbatch --parsable --dependency="afterok:$prep" /projects/yiyan002humanoid/textclass-overnight-20261107/jobs/other-queue.sbatch
