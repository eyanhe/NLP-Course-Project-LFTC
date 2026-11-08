set -eu
P=/projects/yiyan002humanoid/textclass-overnight-20261107
umask 077
mkdir -p "$P/control"
if test -s "$P/control/extension-jobs.json"; then
    cat "$P/control/extension-jobs.json"
    exit 0
fi
if ! test -s "$P/logs/extension-prep-ready.json"; then
    printf '{"waiting_for":"extension_preparation"}\n'
    exit 0
fi
if ! mkdir "$P/control/extension-submission" 2>/dev/null; then
    printf 'Prior extension submission is incomplete; inspect before retrying\n' >&2
    exit 2
fi
a=$(sbatch --parsable "$P/jobs/extension-a.sbatch")
printf '%s\n' "$a" > "$P/control/extension-submission/gpu-a"
b=$(sbatch --parsable "$P/jobs/extension-b.sbatch")
printf '%s\n' "$b" > "$P/control/extension-submission/gpu-b"
printf '{"gpu_a":"%s","gpu_b":"%s"}\n' "$a" "$b" > "$P/control/extension-jobs.json.part"
mv "$P/control/extension-jobs.json.part" "$P/control/extension-jobs.json"
cat "$P/control/extension-jobs.json"
