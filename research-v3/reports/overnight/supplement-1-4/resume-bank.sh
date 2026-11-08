set -euo pipefail
P=/projects/yiyan002humanoid/textclass-overnight-20261107
printf '%s  %s\n' 'febcc3fe589ecd9e51535905c59ca274febb863e73a1b7418bd53e98d5798b62' "$P/jobs/exact-condition-inputs.tar" | sha256sum -c -
tar --skip-old-files -xf "$P/jobs/exact-condition-inputs.tar" -C "$P/repo"
cd "$P/repo"
F=research-v3/reports/overnight/supplement-1-4
exec 9>"$F/submit.lock"
flock -n 9
if [ ! -s "$F/job-bank-v3.txt" ]; then
  sbatch --parsable --dependency=afterany:278031 "$F/cpu-bank.sbatch" > "$F/job-bank-v3.txt"
fi
cat "$F/job-bank-v3.txt"
squeue -u yiyan002 -o '%.18i %.32j %.10T %.12M %.10l %.20R'
