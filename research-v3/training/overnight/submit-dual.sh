set -eu
state=$(squeue -h -j 277705 -o '%T')
test "$state" = PENDING
scancel 277705
sbatch --parsable --dependency=afterok:277704 /projects/yiyan002humanoid/textclass-overnight-20261107/jobs/other-a.sbatch
sbatch --parsable --dependency=afterok:277704 /projects/yiyan002humanoid/textclass-overnight-20261107/jobs/other-b.sbatch
