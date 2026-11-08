P=/projects/yiyan002humanoid/textclass-overnight-20261107
F="$P/repo/research-v3/reports/overnight/supplement-1-4"
date -u
squeue -u yiyan002 -o '%.18i %.32j %.10T %.12M %.10l %.20R'
sacct -j 278000,278001,278031,278032 --noheader --parsable2 --format=JobID,JobName,State,Elapsed,ExitCode
if [ -s "$F/job-bank-v3.txt" ]; then
  job=$(cat "$F/job-bank-v3.txt")
  printf 'bank_v3 %s\n' "$job"
  sacct -j "$job" --noheader --parsable2 --format=JobID,JobName,State,Elapsed,ExitCode
  tail -c 1800 "$P/logs/tc-sup14-bank-${job}.out" 2>/dev/null || true
fi
tail -c 1400 "$P/logs/tc-sup14-emotion-278032.out" 2>/dev/null || true
ls -lh "$P/transfer/supplement-historical.tar" "$P/transfer/supplement-historical.tar.part" 2>/dev/null || true
