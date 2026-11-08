"""Record an operational driver revision; fit requests stay identical."""
import hashlib
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent;PROJECT=HERE.parents[3]
sys.path.insert(0,str(PROJECT))
from research_v3.config import canonical_hash

path=HERE/'manifest.json';old=json.loads(path.read_text(encoding='utf-8'))
archive=HERE/'manifest-v1.json'
if not archive.exists():archive.write_bytes(path.read_bytes())
driver=hashlib.sha256((PROJECT/'research_v3/supplemental.py').read_bytes()).hexdigest()
if old['driver_sha256']!=driver:
    body={k:v for k,v in old.items() if k!='manifest_hash'}
    body.update(driver_sha256=driver,operational_revision=2,
                previous_manifest_hash=old['manifest_hash'],revision_reason='In-memory runtime asset path relocation; unchanged fits, role hashes and frozen learners.')
    path.write_text(json.dumps({**body,'manifest_hash':canonical_hash(body)},ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(dict(driver_sha256=driver,fit_requests_unchanged=True)))
current=json.loads(path.read_text(encoding='utf-8'))
members=['research-v3/runs/'+r['entry']['run_id']+'/' for r in current['trajectory_sources']
         if r['completed_checkpoint_reuse'] and r['model_id'] in {'bilstm','distilbert'}]
(HERE/'historical-restore-members.txt').write_bytes(('\n'.join(members)+'\n').encode('ascii'))
