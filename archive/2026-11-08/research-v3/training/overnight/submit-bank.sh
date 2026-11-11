set -eu
stage=$(sbatch --parsable /projects/yiyan002humanoid/textclass-overnight-20261107/jobs/extract-bank.sbatch)
printf 'CPU_STAGE_JOB=%s\n' "$stage"
sbatch --parsable --dependency="afterok:$stage" /projects/yiyan002humanoid/textclass-overnight-20261107/jobs/bank-queue.sbatch
