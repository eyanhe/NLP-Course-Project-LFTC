"""Small Slurm-only subset dispatcher; unchanged learner and exact requests."""
from __future__ import annotations
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal


def may_start(deadline, now=None):
    return (now or datetime.now(timezone.utc)) < datetime.fromisoformat(deadline)


def gpu_job_required(entry):
    return entry['engine'] == 'diagnostics' or entry['model_id'] in {'bilstm', 'distilbert', 'modernbert'}


def validate_entries(rows, source):
    original = {entry['run_id']:entry for entry in source['runs']}
    ids = [row['entry']['run_id'] for row in rows]
    if len(ids) != len(set(ids)) or any(row['entry'] != original.get(row['entry']['run_id']) for row in rows):
        raise ValueError('Overnight request differs from parent or duplicates a run')


def runtime_assets(locked, project):
    assets = deepcopy(locked)
    project = Path(project).resolve()
    for name, source in assets['models'].items():
        records = [record for record in locked['files'] if record.get('model_id') == name]
        if records:
            directory = (project / records[0]['path']).parent.resolve()
            if not directory.is_relative_to(project):
                raise ValueError('Runtime asset path escaped project')
            source['directory'] = str(directory)
    return assets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--group', choices=['bank', 'other', 'all'], required=True)
    parser.add_argument('--kind', choices=['cpu', 'gpu'], required=True)
    parser.add_argument('--max-priority', type=int, choices=[0, 1, 2], default=0)
    parser.add_argument('--cpu-classifiers-only', action='store_true')
    parser.add_argument('--datasets', nargs='+', choices=['banking77', 'yelp', 'go_emotions', 'yahoo_answers_topics'])
    parser.add_argument('--plan-name', choices=['run-plan.json', 'extension-run-plan.json'], default='run-plan.json')
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise PermissionError('Only authorized cluster Slurm execution')
    project = Path(__file__).resolve().parents[1]
    expected = Path('/projects/yiyan002humanoid/textclass-overnight-20261107/repo').resolve()
    if project != expected:
        raise PermissionError('This overnight authorization is cluster-project specific')
    root = project/'research-v3'
    from .storage import read_json, hash_file, atomic_write_json
    from .batch import fit_declared_run, _run, shared_tfidf_features
    from .orchestrator import _condition_ids, load_case
    from .train import _features
    plan = read_json(root/'training/overnight'/args.plan_name)
    source_path = root/plan['source_manifest_path']
    if hash_file(source_path) != plan['source_manifest_sha256']:
        raise ValueError('Parent training requests changed')
    identity = hashlib.sha256(json.dumps({k:v for k,v in plan.items() if k != 'subset_hash'}, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if identity != plan['subset_hash'] or plan['official_test_predictions_allowed'] is not False:
        raise ValueError('Overnight subset or Q restriction changed')
    validate_entries(plan['runs'], read_json(source_path))
    hard_stop = datetime.fromisoformat(plan['hard_stop_training_at_singapore'])
    seconds = int((hard_stop-datetime.now(timezone.utc)).total_seconds())
    if seconds <= 0:
        return
    signal.signal(signal.SIGALRM, lambda *unused: (_ for _ in ()).throw(TimeoutError('Overnight training hard deadline')))
    signal.alarm(seconds)
    prepared = read_json(root/'data/prepared_manifest.json')
    assets = runtime_assets(read_json(root/'assets/assets.lock.json'), project)
    design = {row['condition_id']:row for row in read_json(root/'matrix/design_manifest.json')['conditions']}
    progress_path = root/'training/overnight'/f'progress-{args.group}-{args.kind}-{os.environ["SLURM_JOB_ID"]}.json'
    done = []
    rows = sorted(plan['runs'], key=lambda row:(row['priority'], row['dataset_id'], row['training_rows'], row['model_id'], row['data_seed']))
    for row in rows:
        if row['priority'] > args.max_priority:
            continue
        if row['completed_checkpoint_reuse']:
            continue
        if args.datasets and row['dataset_id'] not in args.datasets:
            continue
        if args.group != 'all' and (row['dataset_id'] == 'banking77') != (args.group == 'bank'):
            continue
        if not may_start(plan['stop_starting_new_fits_at_singapore']):
            break
        entry = row['entry']
        neural = entry['model_id'] in {'bilstm', 'distilbert', 'modernbert', 'm2_distilbert', 'frozen_generic_distilbert_head'}
        requires_gpu = gpu_job_required(entry)
        if args.kind == 'cpu' and args.cpu_classifiers_only and requires_gpu:
            continue
        if args.kind == 'gpu' and not requires_gpu:
            continue
        event = {'status':'fitting' if args.kind == 'gpu' or not neural else 'preparing_train_only_features', 'run_id':entry['run_id'], 'dataset':entry['dataset_id'], 'model':entry['model_id'], 'priority':row['priority'], 'rows':row['training_rows'], 'slurm_job_id':os.environ['SLURM_JOB_ID'], 'completed_in_worker':len(done), 'official_test_predictions_generated':False}
        atomic_write_json(progress_path, event, root)
        print(json.dumps(event), flush=True)
        if args.kind == 'cpu' and requires_gpu:
            if entry['model_id'] == 'fixed_tfidf_svm':
                shared_tfidf_features(root, prepared['datasets'][entry['dataset_id']])
                result = {'status':'features_ready'}
            elif entry['model_id'] == 'bilstm':
                run = _run(entry)
                condition = design[run.condition_id]
                ids = _condition_ids(condition, root)
                data, inputs = load_case(root, prepared['datasets'][entry['dataset_id']], ids, selection_ids=(), discovery_ids=())
                _features(run, tuple(inputs[sid] for sid in sorted(ids)), root)
                result = {'status':'features_ready'}
            else:
                continue
        else:
            result = fit_declared_run(entry, root, prepared, assets)
            if result['status'] != 'complete' or not result.get('reload_verified'):
                raise ValueError('Actual fit did not pass checkpoint reload')
        done.append(entry['run_id'])
        print(json.dumps({'status':result['status'], 'run_id':entry['run_id'], 'epochs_completed':result.get('epochs_completed'), 'reload_verified':result.get('reload_verified'), 'total_seconds':result.get('total_seconds')}), flush=True)
        import gc
        gc.collect()
        if args.kind == 'gpu':
            import torch
            torch.cuda.empty_cache()
    atomic_write_json(progress_path, {'status':'worker_finished', 'run_ids':done, 'slurm_job_id':os.environ['SLURM_JOB_ID'], 'official_test_predictions_generated':False}, root)


if __name__ == '__main__':
    main()
