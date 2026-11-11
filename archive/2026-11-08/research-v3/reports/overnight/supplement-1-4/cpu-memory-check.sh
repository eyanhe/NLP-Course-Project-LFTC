scontrol show node=cpu-1
scontrol show node=cpu-2
sinfo -o '%P %a %l %D %N'
scontrol show partition cluster02
squeue -t R -o '%.10i %.9u %.10Q %.10C %.12m %.12L %.14b %.20N'
