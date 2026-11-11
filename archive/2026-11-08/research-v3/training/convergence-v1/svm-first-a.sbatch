#!/bin/bash -l
#SBATCH --job-name=tc-conv-svm-first-a
#SBATCH --partition=cluster02
#SBATCH --account=msc
#SBATCH --qos=msc
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=/projects/yiyan002humanoid/textclass-overnight-20261107/logs/%x-%j.out
set -euo pipefail
umask 077
P=/projects/yiyan002humanoid/textclass-overnight-20261107
export TMPDIR="$P/tmp/$SLURM_JOB_ID"
export HF_HOME="$P/cache/hf" TRITON_CACHE_DIR="$P/cache/triton" XDG_CACHE_HOME="$P/cache/xdg"
export PIP_CACHE_DIR="$P/cache/pip" CONDA_PKGS_DIRS="$P/cache/conda"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=""
mkdir -p "$TMPDIR"
cd "$P/repo"
remaining=$(( $(date -u -d '2026-11-08 02:00:00 UTC' +%s) - $(date -u +%s) ))
if (( remaining <= 0 )); then exit 0; fi
timeout --signal=TERM --kill-after=30s "${remaining}s" "$P/envs/research/bin/python" -u -m research_v3.svm_first_worker
