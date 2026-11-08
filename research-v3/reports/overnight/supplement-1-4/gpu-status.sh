date -u
sinfo -N -o '%N %t %C %G'
squeue -u yiyan002 -o '%.18i %.32j %.10T %.12M %.10l %.30R'
for node in gpu-5090-1 gpu-5090-2 gpu-5090-3 gpu-5090-4 gpu-5090-5 gpu-6000ada-1 gpu-6000ada-2 gpu-6000ada-3 gpu-a40-1 gpu-a40-2 gpu-a6000-1 gpu-l40-1 gpu-l40-2 gpu-pro6000-1 gpu-pro6000-2 gpu-pro6000-3 gpu-pro6000-4 gpu-pro6000-5 gpu-pro6000-6 gpu-pro6000-7 gpu-pro6000-8 gpu-pro6000-9 gpu-pro6000-10 gpu-pro6000-11 gpu-pro6000-12 gpu-pro6000-13 gpu-pro6000-14; do
  scontrol show node="$node" -o
done
squeue --start -j 277871,277872 -o '%.18i %.32j %.10T %.22S %.24R'
