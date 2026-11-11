#!/bin/bash -l
#SBATCH --job-name=tc-p2-yelp
#SBATCH --partition=cluster02
#SBATCH --account=msc
#SBATCH --qos=msc
#SBATCH --gres=gpu:l40:1
#SBATCH --time=04:00:00
#SBATCH --output=/projects/yiyan002humanoid/textclass-overnight-20261107/logs/%x-%j.out
set -euo pipefail
umask 077
P=/projects/yiyan002humanoid/textclass-overnight-20261107
export TMPDIR="$P/tmp/$SLURM_JOB_ID"
export HF_HOME="$P/cache/hf" TRITON_CACHE_DIR="$P/cache/triton"
export XDG_CACHE_HOME="$P/cache/xdg"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
mkdir -p "$TMPDIR"
cd "$P/repo"
"$P/envs/research/bin/python" -u -m research_v3.overnight_training --group other --kind gpu --max-priority 2 --datasets yelp
