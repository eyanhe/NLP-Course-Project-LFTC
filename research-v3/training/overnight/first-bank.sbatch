#!/bin/bash -l
#SBATCH --job-name=tc-first-bank
#SBATCH --partition=cluster02
#SBATCH --account=msc
#SBATCH --qos=msc
#SBATCH --gres=gpu:l40:1
#SBATCH --time=00:30:00
#SBATCH --output=/projects/yiyan002humanoid/textclass-overnight-20261107/logs/%x-%j.out
set -euo pipefail
umask 077
P=/projects/yiyan002humanoid/textclass-overnight-20261107
export TMPDIR="$P/tmp/$SLURM_JOB_ID"
export PIP_CACHE_DIR="$P/cache/pip" CONDA_PKGS_DIRS="$P/cache/conda"
export HF_HOME="$P/cache/hf" TRITON_CACHE_DIR="$P/cache/triton"
export XDG_CACHE_HOME="$P/cache/xdg"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
mkdir -p "$TMPDIR"
test -s "$P/logs/environment-ready.txt"
cd "$P/transfer"
sha256sum -c first-bank.tar.sha256
tar --no-same-owner --no-same-permissions -xf first-bank.tar -C "$P/repo"
cd "$P/repo"
export PYTHONPATH="$P/repo"
"$P/envs/research/bin/python" -u "$P/jobs/first-bank-run.py"
