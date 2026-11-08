"""Fix an exploratory D-only frame before revealing outputs; preserve blind pools."""
import hashlib
import json
import tarfile
from pathlib import Path
import pyarrow.parquet as pq

root=Path('research-v3');folder=root/'reports/overnight'
plans=[json.loads((root/'training/overnight'/name).read_text(encoding='utf-8')) for name in ('run-plan.json','extension-run-plan.json')]
selected={row['entry']['run_id']:row for p in plans for row in p['runs']}
observed={row['run_id'] for row in json.loads((folder/'latest-cluster-results.json').read_text(encoding='utf-8'))['completed']}
prepared=json.loads((root/'data/prepared_manifest.json').read_text(encoding='utf-8'))['datasets']
datasets=('yelp','go_emotions','banking77','yahoo_answers_topics')
frames={};checkpoints=[];files=set()
for ds in datasets:
    excluded=set()
    for line in (root/'training/discovery-guide-review'/ds/'reviewer-A.blank.jsonl').read_text(encoding='utf-8').splitlines():
        excluded.add(json.loads(line)['group_id'])
    for pilot in (root/'training/discovery-review').iterdir():
        manifest=pilot/'manifest.json'
        if not manifest.exists():continue
        info=json.loads(manifest.read_text(encoding='utf-8'))
        if info.get('dataset_id')!=ds:continue
        for line in (pilot/'candidates.jsonl').read_text(encoding='utf-8').splitlines():
            excluded.add(json.loads(line)['group_id'])
    meta=prepared[ds];source=root/meta['partitions']['discovery']['file']
    rows=pq.read_table(source,columns=['sample_id','group_id','text','labels']).to_pylist()
    eligible=[row for row in rows if row['group_id'] not in excluded]
    eligible.sort(key=lambda row:hashlib.sha256(('overnight-discovery-7302|'+row['sample_id']).encode()).hexdigest())
    by_group={}
    for row in eligible:by_group.setdefault(row['group_id'],row)
    frame=list(by_group.values())[:500]
    assert all(row['group_id'] not in excluded for row in frame)
    frames[ds]={'source_partition':'discovery','source_file':meta['partitions']['discovery']['file'],
        'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'sampling_seed':7302,
        'excluded_reserved_review_groups':len(excluded),'eligible_unreserved_groups':len(by_group),
        'sample_ids':[row['sample_id'] for row in frame],'group_ids':[row['group_id'] for row in frame],
        'note':'Conditional unreserved D frame; not representative of all D or Q; no human review'}
    files.add(source)
    for run_id,row in selected.items():
        entry=row['entry']
        if entry['dataset_id']!=ds or entry['model_id']!='tfidf_svm' or entry['data_seed']!=2026:continue
        if not any(alias['experiment_id'] in ('E1','E2') for alias in entry['semantic_aliases']):continue
        if not row['completed_checkpoint_reuse'] and run_id not in observed:continue
        checkpoints.append({'run_id':run_id,'dataset_id':ds,'entry':entry,'training_rows':row['training_rows']})
        if row['completed_checkpoint_reuse']:
            status_path=root/'runs'/run_id/'status.json'
            assert hashlib.sha256(status_path.read_bytes()).hexdigest()==row['status_sha256']
            status=json.loads(status_path.read_text(encoding='utf-8'));files.add(status_path)
            for relative in status['files']:files.add(status_path.parent/relative)
packet={'purpose':'unreviewed_exploratory_failure_analysis','frames':frames,'checkpoints':checkpoints,
    'human_review_status':'pending','official_test_predictions_allowed':False,'max_cases_per_dataset':20}
(folder/'failure-frame-plan.json').write_text(json.dumps(packet,indent=2)+'\n',encoding='utf-8')
files.add(folder/'failure-frame-plan.json')
archive=root/'training/overnight/failure-inputs.tar'
with tarfile.open(archive,'w') as tar:
    for path in sorted(files):tar.add(path,arcname=path.as_posix(),recursive=False)
digest=hashlib.sha256(archive.read_bytes()).hexdigest()
archive.with_suffix('.tar.sha256').write_text(digest+'  failure-inputs.tar\n',encoding='ascii')
print(json.dumps({'checkpoint_count':len(checkpoints),'frames':{ds:len(frame['sample_ids']) for ds,frame in frames.items()},'archive_bytes':archive.stat().st_size,'sha256':digest}))
