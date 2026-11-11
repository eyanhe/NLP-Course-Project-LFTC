set -eu
P=/projects/yiyan002humanoid/textclass-overnight-20261107
test "$(readlink -f /projects/yiyan002humanoid)" = /projects/_ssd/yiyan002humanoid
test "$(stat -c %U /projects/yiyan002humanoid/)" = yiyan002
umask 077
mkdir -p "$P/transfer" "$P/jobs" "$P/logs" "$P/repo" "$P/envs" "$P/cache" "$P/tmp"
stat -c '%U %a %n' "$P" "$P/transfer" "$P/repo"
