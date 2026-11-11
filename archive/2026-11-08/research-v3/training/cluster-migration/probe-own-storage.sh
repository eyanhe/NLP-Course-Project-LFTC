readlink -f /projects/yiyan002humanoid
stat -c '%U %G %a %n' /projects/yiyan002humanoid/
test -w /projects/yiyan002humanoid && printf 'OWN_PROJECT_WRITABLE\n'
