#!/bin/bash -l
#SBATCH --job-name=tc-bank-stage
#SBATCH --partition=cluster02
#SBATCH --account=msc
#SBATCH --qos=msc
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=/projects/yiyan002humanoid/textclass-overnight-20261107/logs/%x-%j.out
set -euo pipefail
umask 077
P=/projects/yiyan002humanoid/textclass-overnight-20261107
cd "$P/transfer"
sha256sum -c bank-queue.tar.sha256
tar --no-same-owner --no-same-permissions -xf bank-queue.tar -C "$P/repo"
