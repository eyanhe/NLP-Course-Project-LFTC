set -euo pipefail
umask 077
P=/projects/yiyan002humanoid/textclass-overnight-20261107
R="$P/repo"
S="$P/jobs/supplement-q-v1-staging"
printf '%s  %s\n' 1b31a33ecc42f78f2f1d07240ad8bca9dacec0d5ed6ddad85d0441269eea2459 "$P/jobs/supplement-q-v1.tar" | sha256sum -c -
test ! -e "$R/research-v3/reports/overnight/supplement-q-v1/S1/evaluation.freeze.json"
test ! -e "$R/research-v3/reports/overnight/supplement-q-v1/S3/evaluation.freeze.json"
mkdir -p "$S"
cd "$R"
sha256sum research_v3/scoped_evaluation.py research_v3/timed_inference.py research-v3/training/convergence-v1/scoped-evaluation.freeze.json > "$S/previous-Q-before.sha256"
tar -xf "$P/jobs/supplement-q-v1.tar" -C "$S"
for path in research_v3/supplement_q.py research-v3/training/supplement-q-v1/authorization.json research-v3/training/supplement-q-v1/evaluate.sbatch; do
  mkdir -p "$(dirname "$R/$path")"
  cp "$S/$path" "$R/$path.part"
  mv "$R/$path.part" "$R/$path"
done
sha256sum -c <<'SHA'
bff068a3807d6430a026143e8fa95fa1b9327a8e3defe559ad4c88526456aaf0  research_v3/supplement_q.py
8989ef52378d58eda8a0ec1fc7d0565ed52e8740ea205596530d971efec9c4d1  research-v3/training/supplement-q-v1/authorization.json
1be47bcada3897c88a3b5b5af05bf83d9bc2d98eab9f134901cc3bd372aef550  research-v3/training/supplement-q-v1/evaluate.sbatch
SHA
bash -n research-v3/training/supplement-q-v1/evaluate.sbatch
sha256sum research_v3/scoped_evaluation.py research_v3/timed_inference.py research-v3/training/convergence-v1/scoped-evaluation.freeze.json > "$S/previous-Q-after.sha256"
cmp "$S/previous-Q-before.sha256" "$S/previous-Q-after.sha256"
for stage in S3 S1; do
  pointer="research-v3/training/supplement-q-v1/job-$stage.txt"
  if [ ! -f "$pointer" ]; then
    sbatch --parsable --job-name="tc-$stage-Q" --export="ALL,SUPPLEMENT_STAGE=$stage" research-v3/training/supplement-q-v1/evaluate.sbatch > "$pointer.part"
    mv "$pointer.part" "$pointer"
  fi
  printf '%s ' "$stage"
  cat "$pointer"
done
TZ=Asia/Singapore date -Is
squeue -u yiyan002 -o '%.12i %.25j %.10T %.7C %.12m %.10M %.24R'
