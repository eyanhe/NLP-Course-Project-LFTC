date '+%Y-%m-%d %H:%M:%S %Z %z'
scontrol show nodes -o | awk '{n="";c="";a=""; for(i=1;i<=NF;i++){if($i~/^NodeName=/)n=$i;if($i~/^CfgTRES=/)c=$i;if($i~/^AllocTRES=/)a=$i;} if(n~/NodeName=gpu-/) print n,c,a;}'
squeue --start -j 277871,277872 -o '%.18i %.32j %.10T %.22S %.24R'
