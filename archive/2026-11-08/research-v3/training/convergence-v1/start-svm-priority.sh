set -euo pipefail
umask 077
P=/projects/yiyan002humanoid/textclass-overnight-20261107
cd "$P/repo"
F=research-v3/training/convergence-v1
printf '%s  %s\n' '929c3b4f1abf051e05b49ee3f37bc3ebb4a63539f8f4c829b65d328a64971440' "$F/svm-priority.py" | sha256sum -c -
test "$(squeue -h -j 278148 -o '%T')" = RUNNING
test "$(squeue -h -j 278149 -o '%T')" = RUNNING
srun --help | grep -E -- '--jobid|--overlap|--cpus-per-task|--mem='
exec 9>"$F/svm-priority-launch.lock"
flock -n 9
if test -e "$F/svm-priority-launch.json"; then cat "$F/svm-priority-launch.json"; exit 0; fi
export SVM_PRIORITY_NONCE="$(date -u +%s)-svm-first"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
nohup srun --jobid=278149 --overlap --ntasks=1 --cpus-per-task=1 --mem=1G "$P/envs/research/bin/python" -u "$F/svm-priority.py" --watchdog > "$P/logs/svm-priority-watchdog.log" 2>&1 < /dev/null &
watchdog_pid=$!
nohup srun --jobid=278148 --overlap --ntasks=1 --cpus-per-task=1 --mem=1G "$P/envs/research/bin/python" -u "$F/svm-priority.py" > "$P/logs/svm-priority-controller.log" 2>&1 < /dev/null &
controller_pid=$!
printf '{"nonce":"%s","controller_srun_pid":%s,"watchdog_srun_pid":%s,"owner_cpu_jobs":[278148,278149],"additional_allocations":0}\n' "$SVM_PRIORITY_NONCE" "$controller_pid" "$watchdog_pid" > "$F/svm-priority-launch.json"
cat "$F/svm-priority-launch.json"
date -Is
