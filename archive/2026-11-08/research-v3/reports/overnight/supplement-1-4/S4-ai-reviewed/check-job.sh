P=/projects/yiyan002humanoid/textclass-overnight-20261107
F="$P/repo/research-v3/reports/overnight/supplement-1-4/S4-ai-reviewed"
date -u
job=$(cat "$F/job-id.txt")
sacct -j "$job" --noheader --parsable2 --format=JobID,JobName,State,Elapsed,ExitCode
tail -c 4000 "$P/logs/tc-sup4-ai-${job}.out" 2>/dev/null || true
cat "$F/result-manifest.json" 2>/dev/null || true
