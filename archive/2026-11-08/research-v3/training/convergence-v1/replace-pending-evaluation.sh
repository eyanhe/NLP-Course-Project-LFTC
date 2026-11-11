set -euo pipefail
umask 077
P=/projects/yiyan002humanoid/textclass-overnight-20261107
printf '%s  %s\n' 'd4cfaad990056ca65b5c71a6bc1f046260e9b5f6495fb6f98aacdf4e6fe4863c' "$P/jobs/convergence-v1-budget-amendment.tar" | sha256sum -c -
cd "$P/repo"
F=research-v3/training/convergence-v1
exec 9>"$F/budget-replacement.lock"
flock -n 9
old=$(cat "$F/job-evaluate.txt")
if [ "$old" != 278170 ]; then printf 'already_replaced_evaluation=%s\n' "$old"; exit 0; fi
test ! -e "$F/scoped-evaluation.freeze.json"
test ! -e research-v3/reports/overnight/convergence-v1/evaluation-started.json
state=$(squeue -h -j "$old" -o '%T')
test "$state" = PENDING
tar -xf "$P/jobs/convergence-v1-budget-amendment.tar" -C "$P/repo" "$F/finish-and-evaluate.sbatch"
sbatch --test-only --dependency=afterany:278148:278149 "$F/finish-and-evaluate.sbatch"
scancel --state=PENDING "$old"
state=$(squeue -h -j "$old" -o '%T')
test -z "$state"
cp "$F/job-evaluate.txt" "$F/job-evaluate-before-budget-amendment.txt"
tar -xf "$P/jobs/convergence-v1-budget-amendment.tar" -C "$P/repo"
sbatch --parsable --dependency=afterany:278148:278149 "$F/finish-and-evaluate.sbatch" > "$F/job-evaluate.txt.part"
mv "$F/job-evaluate.txt.part" "$F/job-evaluate.txt"
printf 'replacement_evaluation='
cat "$F/job-evaluate.txt"
date -Is
squeue -u yiyan002 -o '%.12i %.30j %.10T %.10M %.10l %.28R'
scontrol show job "$(cat "$F/job-evaluate.txt")" -o | tr ' ' '\n' | sed -n '/^JobId=/p; /^JobState=/p; /^Dependency=/p'
