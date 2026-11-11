set -eu
P=/projects/yiyan002humanoid/textclass-overnight-20261107
test "$(squeue -h -j 277767 -o '%T')" = PENDING
test "$(squeue -h -j 277768 -o '%T')" = PENDING
scancel 277767 277768
a=$(sbatch --parsable --gres=gpu:a40:1 --time=01:00:00 "$P/jobs/p1-a.sbatch")
printf 'REPLACEMENT_A=%s\n' "$a"
b=$(sbatch --parsable --gres=gpu:6000ada:1 --time=01:00:00 "$P/jobs/p1-b.sbatch")
printf 'REPLACEMENT_B=%s\n' "$b"
