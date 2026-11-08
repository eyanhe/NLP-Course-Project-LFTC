set -euo pipefail
P=/projects/yiyan002humanoid/textclass-overnight-20261107
printf '%s  %s\n' 'bb1d0a381132e3f34d62b8340937d8fffa95b47839dbaaefcb78baac5c66560c' "$P/jobs/ai-reviewed-evaluation.tar" | sha256sum -c -
tar -xf "$P/jobs/ai-reviewed-evaluation.tar" -C "$P/repo"
cd "$P/repo"
F=research-v3/reports/overnight/supplement-1-4/S4-ai-reviewed
test -f research-v3/runs/5aa9bf514fa60c7ef6651214d304dc213b0012a265a8af0e2e2bfa46913a9488/bundle.joblib
test -f research-v3/runs/a9b8318f052466eb376cd08d7879ec3752c122ceb4113cda16303318270298c0/checkpoint/model.safetensors
test -f research-v3/data/banking77/discovery.parquet
test -f research-v3/inputs/banking77.parquet
exec 9>"$F/submit.lock"
flock -n 9
if [ ! -s "$F/job-id.txt" ]; then
  sbatch --test-only "$F/cpu-evaluate.sbatch"
  sbatch --parsable "$F/cpu-evaluate.sbatch" > "$F/job-id.txt"
fi
cat "$F/job-id.txt"
squeue -u yiyan002 -o '%.18i %.30j %.10T %.12M %.24R'
