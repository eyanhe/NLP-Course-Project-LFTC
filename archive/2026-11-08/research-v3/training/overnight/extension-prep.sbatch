#!/bin/bash -l
#SBATCH --job-name=tc-extension-prep
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
cd "$P/repo"
cd "$P/repo"
"$P/envs/research/bin/python" -u -m research_v3.overnight_training --group all --kind cpu --max-priority 2 --plan-name extension-run-plan.json
printf '{"ready":true,"job_id":"%s"}\n' "$SLURM_JOB_ID" > "$P/logs/extension-prep-ready.json"
