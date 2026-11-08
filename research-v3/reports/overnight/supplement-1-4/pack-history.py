"""Package only the ten authenticated, already-trained neural checkpoints."""
import hashlib
import json
from pathlib import Path
import tarfile
import sys
HERE=Path(__file__).resolve().parent;PROJECT=HERE.parents[3]
sys.path.insert(0,str(PROJECT))
from research_v3.storage import hash_file
manifest=json.loads((HERE/'manifest.json').read_text(encoding='utf-8'))
verified=json.loads((PROJECT/'research-v3/training/cluster-migration/verified-checkpoints.json').read_text(encoding='utf-8'))
expected={r['run_id']:r['status_sha256'] for r in verified['formal_checkpoints']}
archive=HERE/'supplement-historical.tar';records=[]
with tarfile.open(archive,'w') as tar:
    for record in manifest['trajectory_sources']:
        if not record['completed_checkpoint_reuse'] or record['model_id'] not in {'bilstm','distilbert'}:continue
        rid=record['entry']['run_id'];directory=PROJECT/'research-v3/runs'/rid
        if hash_file(directory/'status.json')!=expected[rid]:raise ValueError('Historical status changed')
        status=json.loads((directory/'status.json').read_text(encoding='utf-8'))
        for rel,sha in {**status['files'],'status.json':expected[rid]}.items():
            path=(directory/rel).resolve()
            if not path.is_relative_to(directory) or hash_file(path)!=sha:raise ValueError('Historical checkpoint bytes changed')
            tar.add(path,arcname=path.relative_to(PROJECT).as_posix(),recursive=False)
            records.append(dict(path=path.relative_to(PROJECT).as_posix(),sha256=sha,bytes=path.stat().st_size))
digest=hash_file(archive)
(HERE/'supplement-historical.tar.sha256').write_bytes((digest+'  supplement-historical.tar\n').encode())
(HERE/'historical-transfer-manifest.json').write_text(json.dumps(dict(files=records,archive_sha256=digest,archive_bytes=archive.stat().st_size),indent=2),encoding='utf-8')
print(json.dumps(dict(files=len(records),bytes=archive.stat().st_size,sha256=digest)),flush=True)
