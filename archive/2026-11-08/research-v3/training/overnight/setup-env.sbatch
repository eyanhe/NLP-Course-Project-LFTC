#!/bin/bash -l
#SBATCH --job-name=tc-night-env
#SBATCH --partition=cluster02
#SBATCH --account=msc
#SBATCH --qos=msc
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=/projects/yiyan002humanoid/textclass-overnight-20261107/logs/%x-%j.out
set -euo pipefail
umask 077
P=/projects/yiyan002humanoid/textclass-overnight-20261107
export TMPDIR="$P/tmp/env-$SLURM_JOB_ID"
export PIP_CACHE_DIR="$P/cache/pip" CONDA_PKGS_DIRS="$P/cache/conda"
export CONDA_ENVS_PATH="$P/envs" HF_HOME="$P/cache/hf"
export TRITON_CACHE_DIR="$P/cache/triton" XDG_CACHE_HOME="$P/cache/xdg"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
mkdir -p "$TMPDIR" "$P/cache"
module load Miniforge3/24.11.3-1
conda create --yes --prefix "$P/envs/research" python=3.11.9 pip
PY="$P/envs/research/bin/python"
"$PY" -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
"$PY" -m pip install -r "$P/transfer/requirements-research.in"
"$PY" -m pip check
"$PY" -m pip freeze > "$P/logs/environment-actual.txt"
"$PY" -c 'import torch,transformers,numpy,pyarrow,sklearn,gensim; print({"torch":torch.__version__,"transformers":transformers.__version__,"numpy":numpy.__version__,"pyarrow":pyarrow.__version__,"sklearn":sklearn.__version__,"gensim":gensim.__version__})'
date -u +%FT%TZ > "$P/logs/environment-ready.txt"
