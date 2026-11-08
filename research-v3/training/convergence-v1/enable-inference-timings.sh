set -euo pipefail
umask 077
P=/projects/yiyan002humanoid/textclass-overnight-20261107
cd "$P/repo"
test ! -e research-v3/training/convergence-v1/scoped-evaluation.freeze.json
test ! -e research-v3/reports/overnight/convergence-v1/evaluation-started.json
printf '%s  %s\n' 'b8b762f93d9c6c821593819a380d4939abc23ef9798159f2120a9df611fb8942' "$P/jobs/timed-inference-code.tar" | sha256sum -c -
stage="$P/jobs/timed-inference-staging"
mkdir -p "$stage"
tar -xf "$P/jobs/timed-inference-code.tar" -C "$stage"
cp "$stage/research_v3/timed_inference.py" research_v3/timed_inference.py.part
mv research_v3/timed_inference.py.part research_v3/timed_inference.py
test ! -e research-v3/training/convergence-v1/scoped-evaluation.freeze.json
cp "$stage/research_v3/scoped_evaluation.py" research_v3/scoped_evaluation.py.part
mv research_v3/scoped_evaluation.py.part research_v3/scoped_evaluation.py
date -Is
sha256sum research_v3/timed_inference.py research_v3/scoped_evaluation.py
squeue -u yiyan002 -o '%.12i %.29j %.10T %.10M %.24R'
