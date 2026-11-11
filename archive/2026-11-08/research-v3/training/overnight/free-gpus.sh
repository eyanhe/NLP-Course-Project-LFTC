python3 - <<'PY'
import subprocess,re,json
out=subprocess.check_output(['scontrol','show','nodes','-o'],text=True)
rows=[]
for line in out.splitlines():
    fields=dict(re.findall(r'(\w+)=([^\s]+)',line))
    model=next((m for m in ('a40','a6000','l40','6000ada') if 'gres/gpu:'+m+'=' in fields.get('CfgTRES','')),None)
    if model is None:continue
    def count(text,key):
        match=re.search(r'(?:^|,)'+re.escape(key)+r'=(\d+)',text)
        return int(match.group(1)) if match else 0
    key='gres/gpu:'+model
    free=count(fields.get('CfgTRES',''),key)-count(fields.get('AllocTRES',''),key)
    cpus=int(fields.get('CPUTot',0))-int(fields.get('CPUAlloc',0))
    if free>0 and cpus>=4:rows.append({'node':fields.get('NodeName'),'model':model,'free_gpus':free,'free_cpus':cpus,'state':fields.get('State')})
print(json.dumps({'compatible_free_nodes':rows}))
PY
