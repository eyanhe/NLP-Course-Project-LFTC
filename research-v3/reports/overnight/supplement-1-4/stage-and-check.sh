set -euo pipefail
P=/projects/yiyan002humanoid/textclass-overnight-20261107
printf '%s  %s\n' '9b0dc2d4154543ba630e36fe394e11ca47caeeb9ad1cb66e53da99adacd5f492' "$P/jobs/supplement-code.tar" | sha256sum -c -
tar -xf "$P/jobs/supplement-code.tar" -C "$P/repo"
cd "$P/repo"
sacctmgr -n show qos msc format=Name,MaxJobsPU,MaxSubmitPU,MaxTRESPU,MaxWall
sbatch --test-only research-v3/reports/overnight/supplement-1-4/cpu-bank.sbatch
sbatch --test-only research-v3/reports/overnight/supplement-1-4/cpu-emotion.sbatch
