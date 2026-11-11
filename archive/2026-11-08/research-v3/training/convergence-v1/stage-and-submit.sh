set -euo pipefail
umask 077
P=/projects/yiyan002humanoid/textclass-overnight-20261107
printf '%s  %s\n' '4ae82b7af0d3ea305973fc3563aad6d60871e5f8bcd3ab25a70be2f6966b6ae2' "$P/jobs/convergence-v1-code.tar" | sha256sum -c -
tar -xf "$P/jobs/convergence-v1-code.tar" -C "$P/repo"
cd "$P/repo"
F=research-v3/training/convergence-v1
exec 9>"$F/submit.lock"
flock -n 9
sbatch --test-only "$F/cpu-a.sbatch"
sbatch --test-only "$F/cpu-b.sbatch"
for worker in a b; do
  if [ ! -s "$F/job-cpu-${worker}.txt" ]; then
    sbatch --parsable "$F/cpu-${worker}.sbatch" > "$F/job-cpu-${worker}.txt"
  fi
  printf 'convergence_cpu_%s=' "$worker"
  cat "$F/job-cpu-${worker}.txt"
done
date -Is
squeue -u yiyan002 -o '%.12i %.30j %.10T %.10M %.10l %.28R'
