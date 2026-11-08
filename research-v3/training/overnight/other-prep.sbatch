#!/bin/bash -l
#SBATCH --job-name=tc-other-prep
#SBATCH --partition=cluster02
#SBATCH --account=msc
#SBATCH --qos=msc
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=/projects/yiyan002humanoid/textclass-overnight-20261107/logs/%x-%j.out
set -euo pipefail
umask 077
P=/projects/yiyan002humanoid/textclass-overnight-20261107
export TMPDIR="$P/tmp/$SLURM_JOB_ID"
export HF_HOME="$P/cache/hf" XDG_CACHE_HOME="$P/cache/xdg"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
mkdir -p "$TMPDIR"
cd "$P/transfer"
sha256sum -c other-data.tar.sha256
tar --no-same-owner --no-same-permissions -xf other-data.tar -C "$P/repo"
cd "$P/repo"
"$P/envs/research/bin/python" -u -m research_v3.overnight_training --group bank --kind cpu --cpu-classifiers-only --max-priority 0
"$P/envs/research/bin/python" -u -m research_v3.overnight_training --group other --kind cpu --max-priority 0
