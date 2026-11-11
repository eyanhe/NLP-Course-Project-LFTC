set -eu
P=/projects/yiyan002humanoid/textclass-overnight-20261107
umask 077
mkdir -p "$P/control"
if test -s "$P/control/p2-jobs.json"; then
    cat "$P/control/p2-jobs.json"
    exit 0
fi
if ! mkdir "$P/control/p2-submission" 2>/dev/null; then
    printf 'Prior P2 submission is incomplete; inspect before retrying\n' >&2
    exit 2
fi
prep=$(sbatch --parsable "$P/jobs/p2-prep.sbatch")
printf '%s\n' "$prep" > "$P/control/p2-submission/prep"
a=$(sbatch --parsable --dependency="afterok:$prep" "$P/jobs/p2-a.sbatch")
printf '%s\n' "$a" > "$P/control/p2-submission/gpu-a"
b=$(sbatch --parsable --dependency="afterok:$prep" "$P/jobs/p2-b.sbatch")
printf '%s\n' "$b" > "$P/control/p2-submission/gpu-b"
printf '{"prep":"%s","gpu_a":"%s","gpu_b":"%s"}\n' "$prep" "$a" "$b" > "$P/control/p2-jobs.json.part"
mv "$P/control/p2-jobs.json.part" "$P/control/p2-jobs.json"
cat "$P/control/p2-jobs.json"
