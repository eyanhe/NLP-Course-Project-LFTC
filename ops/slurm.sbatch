#!/usr/bin/env bash
#SBATCH --job-name=lftc
#SBATCH --output=lftc-%j.out
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
# Add your cluster account/partition and --gres=gpu:1 for a GPU recipe.
set -euo pipefail
export LFTC_ROOT="${LFTC_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the project root or set LFTC_ROOT}}"
export LFTC_CONFIG_DIR="${LFTC_CONFIG_DIR:-$LFTC_ROOT/configs}"
export LFTC_ARTIFACTS_DIR="${LFTC_ARTIFACTS_DIR:-$LFTC_ROOT/artifacts}"
export PYTHONPATH="$LFTC_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
# LFTC_PYTHON may point to a preconfigured cluster environment's Python.
# Dataset files and offline checkpoints must be present in the configured assets directory.
config="${1:?Usage: sbatch ops/slurm.sbatch CONFIG.yaml}"
shift
cd -- "$LFTC_ROOT"
exec "${LFTC_PYTHON:-python}" -m lftc run --config "$config" "$@"
