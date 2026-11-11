set -euo pipefail
umask 077
P=/projects/yiyan002humanoid/textclass-overnight-20261107
printf '%s  %s\n' '4b75498a0596c8630591769a63ff3eadde8db0056318eeac284c0042de9fe1cf' "$P/jobs/convergence-v1-evaluation.tar" | sha256sum -c -
tar -xf "$P/jobs/convergence-v1-evaluation.tar" -C "$P/repo"
cd "$P/repo"
F=research-v3/training/convergence-v1
exec 9>"$F/submit-evaluation.lock"
flock -n 9
sbatch --test-only --dependency=afterany:278148:278149 "$F/evaluate.sbatch"
if [ ! -s "$F/job-evaluate.txt" ]; then
  sbatch --parsable --dependency=afterany:278148:278149 "$F/evaluate.sbatch" > "$F/job-evaluate.txt"
fi
printf 'convergence_evaluation='
cat "$F/job-evaluate.txt"
squeue -u yiyan002 -o '%.12i %.30j %.10T %.10M %.10l %.28R'
