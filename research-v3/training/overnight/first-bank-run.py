"""One already-declared P0 fit, invoked only in the authorized Slurm job."""
import json
import os
from pathlib import Path
from research_v3.batch import fit_declared_run
from research_v3.storage import read_json

project = Path('/projects/yiyan002humanoid/textclass-overnight-20261107/repo')
assert os.environ.get('SLURM_JOB_ID'), 'Cluster batch job required'
assert Path.cwd().resolve() == project.resolve()
root = project/'research-v3'
permit = read_json(root/'training/overnight/first-bank-request.json')
entry = permit['entry']
assert permit['authorized_host'] == '10.97.216.128'
assert permit['official_test_predictions_allowed'] is False
assert entry['dataset_id'] == 'banking77' and entry['model_id'] == 'bilstm'
assert entry['run']['strict_no_val'] is True
assert entry['training_seed'] == 42 and entry['data_seed'] == 2026
print(json.dumps({'status':'starting_actual_training','slurm_job_id':os.environ['SLURM_JOB_ID'],'run_id':entry['run_id'],'model':'bilstm','dataset':'banking77','shots_per_class':10,'epochs':10}), flush=True)
result = fit_declared_run(entry, root, read_json(root/'data/prepared_manifest.json'), read_json(root/'assets/assets.lock.json'))
print(json.dumps({'status':result['status'],'run_id':entry['run_id'],'epochs_completed':result.get('epochs_completed'),'reload_verified':result.get('reload_verified'),'total_seconds':result.get('total_seconds')}), flush=True)
