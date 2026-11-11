set -euo pipefail
P=/projects/yiyan002humanoid/textclass-overnight-20261107
F="$P/repo/research-v3/reports/overnight/supplement-1-4"
# Heavy archive extraction runs only inside the CPU Slurm allocation.
test -n "${SLURM_JOB_ID:-}"
cd "$P/transfer"
sha256sum -c supplement-historical.tar.sha256
tar --skip-old-files -xf "$P/transfer/supplement-historical.tar" -C "$P/repo"
