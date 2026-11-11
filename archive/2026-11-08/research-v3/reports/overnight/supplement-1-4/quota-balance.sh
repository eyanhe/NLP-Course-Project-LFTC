date '+%Y-%m-%d %H:%M:%S %Z %z'
sshare -U -nP -o GrpTRESMins,GrpTRESRaw | awk -F'|' '{match($1,/billing=([0-9]+)/,c); match($2,/billing=([0-9]+)/,u); if (c[1]>0) printf "used=%d cap=%d remaining=%d SU remaining_percent=%.4f\n",u[1],c[1],c[1]-u[1],100*(c[1]-u[1])/c[1]; else print "No personal cap."}'
bash -lc 'type storagemgr; command -v quota' 2>&1 || true
