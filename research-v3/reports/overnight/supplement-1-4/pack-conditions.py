"""Copy exact registered support-ID files, never create new subsets."""
import json
from pathlib import Path
import sys
import tarfile
HERE=Path(__file__).resolve().parent;PROJECT=HERE.parents[3]
sys.path.insert(0,str(PROJECT))
from research_v3.storage import read_json,hash_file
root=PROJECT/'research-v3'
manifest=read_json(HERE/'manifest.json')
design={r['condition_id']:r for r in read_json(root/'matrix/design_manifest.json')['conditions']}
paths={}
for r in manifest['fits']:
    condition=design[r['entry']['condition_id']]
    if 'train_ids_file' in condition:
        path=root/condition['train_ids_file']
        assert hash_file(path)==condition['train_ids_file_sha256']
        paths[path]=path.relative_to(PROJECT).as_posix()
with tarfile.open(HERE/'exact-condition-inputs.tar','w') as tar:
    for path,name in paths.items():tar.add(path,arcname=name,recursive=False)
print(json.dumps(dict(files=len(paths),sha256=hash_file(HERE/'exact-condition-inputs.tar'))))
