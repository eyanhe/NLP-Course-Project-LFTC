python3 - <<'PY'
import subprocess,json
helptext=subprocess.check_output(['sbatch','--help'],text=True)
assert '--test-only' in helptext
print('Testing only: no jobs are submitted or resources allocated')
for model in ('a40','a6000','6000ada','l40'):
    command=['sbatch','--test-only','--gres=gpu:'+model+':1','--time=01:00:00','/projects/yiyan002humanoid/textclass-overnight-20261107/jobs/p1-a.sbatch']
    result=subprocess.run(command,capture_output=True,text=True,timeout=20)
    print(json.dumps({'model':model,'requested_hours':1,'exit_code':result.returncode,'prediction':result.stdout+result.stderr}))
print('Server clock:',subprocess.check_output(['date','-Is'],text=True).strip())
PY
