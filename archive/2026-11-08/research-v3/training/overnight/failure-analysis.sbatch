#!/bin/bash -l
#SBATCH --job-name=tc-D-cases-cpu
#SBATCH --partition=cluster02
#SBATCH --account=msc
#SBATCH --qos=msc
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=/projects/yiyan002humanoid/textclass-overnight-20261107/logs/%x-%j.out
set -euo pipefail
umask 077
P=/projects/yiyan002humanoid/textclass-overnight-20261107
export TMPDIR="$P/tmp/$SLURM_JOB_ID" XDG_CACHE_HOME="$P/cache/xdg"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
mkdir -p "$TMPDIR"
cd "$P/transfer"
sha256sum -c failure-inputs.tar.sha256
tar --no-same-owner --no-same-permissions -xf failure-inputs.tar -C "$P/repo"
cd "$P/repo"
export PYTHONPATH="$P/repo"
"$P/envs/research/bin/python" -u "$P/jobs/score-svm-cases.py"
