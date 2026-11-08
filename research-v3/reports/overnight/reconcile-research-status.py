"""Reconcile declared experiments with existing evidence; no execution changes."""
import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path

FOLDER = Path(__file__).resolve().parent
ROOT = FOLDER.parents[1]

def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

base_path = ROOT/'training/overnight/run-plan.json'
extra_path = ROOT/'training/overnight/extension-run-plan.json'
snapshot_path = ROOT/'training/overnight/cluster-progress.json'
verified_path = FOLDER/'verified-cluster-progress.json'
if verified_path.exists() and read(verified_path)['recorded_at_utc'] > read(snapshot_path)['recorded_at_utc']:
    snapshot_path = verified_path
history_path = ROOT/'training/cluster-migration/verified-checkpoints.json'
original_path = ROOT/'training/fixed-main-run-manifest.json'
base, extra, snapshot, history, original = map(read, (base_path, extra_path, snapshot_path, history_path, original_path))
selected = {r['entry']['run_id']: r for r in base['runs'] + extra['runs']}
all_runs = {r['run_id']: r for r in original['runs']}
assert len(selected) == 201 and len(all_runs) == len(original['runs']) == 2859
assert set(selected) <= set(all_runs)
observed = {r['run_id']: r for r in snapshot['completed']}
historical = {r['run_id']: r for r in history['formal_checkpoints']}
assert len(historical) == history['formal_complete_count'] == 131
verified_statuses = {}
for rid, record in historical.items():
    path = ROOT/'runs'/rid/'status.json'
    assert sha(path) == record['status_sha256']
    status = read(path)
    assert status['status'] == 'complete' and status['reload_verified']
    verified_statuses[rid] = status
completed_all = set(historical) | set(observed)
assert completed_all <= set(all_runs)
completed_selected = set(selected) & completed_all
assert len(completed_selected) == 120
design = {r['condition_id']: r for r in read(ROOT/'matrix/design_manifest.json')['conditions']}
register = []
for rid, entry in all_runs.items():
    status = verified_statuses.get(rid)
    condition = design[entry['condition_id']]
    metric = 'macro_ap' if entry['dataset_id'] == 'go_emotions' else 'macro_f1'
    value = status.get('validation', {}).get(metric) if status else observed.get(rid, {}).get(metric)
    register.append({
        'run_id': rid, 'scope': 'selected_overnight' if rid in selected else 'outside_overnight_scope',
        'evidence_state': 'historical_status_hash_verified' if rid in historical else
            'cluster_completed_snapshot' if rid in observed else 'completion_not_confirmed_in_available_evidence',
        'experiments': ','.join(sorted({a['experiment_id'] for a in entry['semantic_aliases']})),
        'dataset': entry['dataset_id'], 'model': entry['model_id'],
        'n': condition['metadata']['rows'], 'data_seed': entry['data_seed'],
        'training_seed': entry['training_seed'], 'condition_id': entry['condition_id'],
        'strict_no_validation': bool(entry['run'].get('strict_no_val')),
        'metric': metric, 'validation_score': value,
        'scores_are_not_formal_Q': True,
    })
with (FOLDER/'experiment-register.csv').open('w', encoding='utf-8-sig', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(register[0]))
    writer.writeheader(); writer.writerows(register)
selected_records = [r for r in register if r['scope'] == 'selected_overnight']

def primary(r):
    return r['data_seed'] == 2026 and r['training_seed'] in (None, 42)

def find(ds, model, n, *, experiment='E1', ratio=None):
    out = []
    for rid, row in selected.items():
        e = row['entry']; c = design[e['condition_id']]
        if e['dataset_id'] == ds and e['model_id'] == model and row['training_rows'] == n and primary(e) \
                and c['experiment_id'] == experiment and (ratio is None or c['metadata'].get('requested_ratio') == ratio):
            out.append(rid)
    assert len(out) == 1, (ds, model, n, experiment, ratio, len(out))
    return out[0]

by_id = {r['run_id']: r for r in register}
boundary_points = []
scale_pairs = []
datasets = ('yelp', 'go_emotions', 'banking77', 'yahoo_answers_topics')
for ds in datasets:
    for model in ('tfidf_svm', 'bilstm', 'distilbert'):
        pair = [find(ds, model, n) for n in (1000, 6000)]
        scale_pairs.append({'dataset': ds, 'model': model, 'run_ids': pair,
                            'complete': all(rid in completed_selected for rid in pair)})
    for n in (1000, 6000):
        a, b = [by_id[find(ds, model, n)] for model in ('tfidf_svm', 'distilbert')]
        boundary_points.append({'dataset': ds, 'n': n, 'svm': a['validation_score'],
            'distilbert': b['validation_score'], 'delta_distilbert_minus_svm': b['validation_score']-a['validation_score'],
            'run_ids': [a['run_id'], b['run_id']]})
imbalance = []
for ds, n in (('yelp',6000), ('banking77',781), ('yahoo_answers_topics',6000)):
    for model in ('tfidf_svm','bilstm','distilbert'):
        pair = [find(ds,model,n,experiment='E3',ratio=r) for r in (1,10)]
        a,b = [by_id[rid] for rid in pair]
        imbalance.append({'dataset':ds,'model':model,'n':n,'run_ids':pair,
            'complete':all(rid in completed_selected for rid in pair),
            'balanced':a['validation_score'],'skewed':b['validation_score'],
            'drop_percentage_points':100*(a['validation_score']-b['validation_score'])})
logical = defaultdict(set)
for rid,row in selected.items():
    for alias in row['entry']['semantic_aliases']:
        logical[alias['experiment_id']].add(rid)
semantic_counts = {name:{'selected_physical_requests':len(ids),
                        'completed':len(ids & completed_selected),
                        'warning':'shared references across modules; do not sum module counts'}
                   for name,ids in logical.items()}
go_svm = verified_statuses[find('go_emotions','tfidf_svm',6000)]
go_deep = verified_statuses[find('go_emotions','distilbert',6000)]
labels = read(ROOT/'data/prepared_manifest.json')['datasets']['go_emotions']['label_names']
tail_contribution=0
for i,name in enumerate(labels):
    if name in {'pride','embarrassment','grief'}:
        tail_contribution += (go_svm['validation']['per_label'][i]['ap']-go_deep['validation']['per_label'][i]['ap'])/28
watchdog = read(ROOT/'training/overnight/watchdog-heartbeat.json')
retrieval = read(FOLDER/'cluster-evidence-retrieval.json')
case_folder = FOLDER/'discovery-cases'
for artifact in retrieval['downloaded']:
    assert sha(case_folder/artifact['name']) == artifact['sha256']
case_manifest = read(case_folder/'manifest.json')
cases = [json.loads(line) for line in (case_folder/'cases.jsonl').read_text(encoding='utf-8').splitlines()]
predictions = [json.loads(line) for line in (case_folder/'D-predictions.jsonl').read_text(encoding='utf-8').splitlines()]
assert len(cases) == case_manifest['total_cases'] == 80
assert len({(c['dataset_id'],c['sample_id']) for c in cases}) == 80
assert all(c['source_partition']=='discovery' and c['semantic_failure_reason']=='unknown' for c in cases)
assert all(p['partition']=='discovery' and p['score_kind']=='svm_margin' for p in predictions)
by_prediction = {(p['dataset_id'],p['sample_id'],p['run_id']):p for p in predictions}
assert len(by_prediction) == len(predictions)
for c in cases:
    assert abs(c['linear_evidence']['reconstruction_error']) < 1e-10
    for trajectory in c['trajectory']:
        p = by_prediction[c['dataset_id'],c['sample_id'],trajectory['run_id']]
        assert p['prediction'] == trajectory['predicted']
        assert trajectory['correct'] == (set(p['prediction']) == set(p['dataset_labels']))
        assert p['group_id'] == c['group_id'] and p['dataset_labels'] == c['dataset_label_ids']
frames=defaultdict(dict)
for p in predictions:
    frames[p['dataset_id'],p['sample_id']][p['condition']]=p
transitions=[]
for ds in datasets:
    for before,after in (('N1000','N6000'),('N6000','N20000'),('N20000','N100000'),('K1','K10')):
        count={'correct_both':0,'corrected':0,'new_error':0,'wrong_both':0}
        for (dataset,sid),conditions in frames.items():
            if dataset!=ds or not {before,after}<=conditions.keys():continue
            a,b=conditions[before],conditions[after]
            assert a['dataset_labels']==b['dataset_labels'] and a['group_id']==b['group_id']
            ca=set(a['prediction'])==set(a['dataset_labels'])
            cb=set(b['prediction'])==set(b['dataset_labels'])
            count['correct_both' if ca and cb else 'corrected' if cb else 'new_error' if ca else 'wrong_both']+=1
        total=sum(count.values())
        if total:
            transitions.append({'dataset':ds,'model':'tfidf_svm','before':before,'after':after,
                'paired_D_rows':total,**count,'correctness_definition':'exact_label_set_match' if ds=='go_emotions' else 'single_label_match',
                'scope':'conditional_D_frame_after_reserved_group_exclusion_not_population_estimate',
                'human_review_status':'pending'})
with (FOLDER/'failure-transition-summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(transitions[0]));writer.writeheader();writer.writerows(transitions)
summary = {
    'recorded_at_singapore':datetime.now(timezone(timedelta(hours=8))).isoformat(),
    'last_verified_cluster_snapshot_utc':snapshot['recorded_at_utc'],
    'live_connection_status':'SSH restored; remote files downloaded with SHA256 verification',
    'remote_evidence_verified_at_utc':retrieval['recorded_at_utc'],
    'watchdog_status_at_read':watchdog.get('status'),
    'watchdog_last_check_utc':watchdog.get('checked_at_utc'),
    'original_physical_requests':len(all_runs),
    'historical_verified_checkpoints':len(historical),
    'completed_in_union_of_available_evidence':len(completed_all),
    'overnight_selected':len(selected), 'overnight_completed':len(completed_selected),
    'overnight_reused':sum(r['completed_checkpoint_reuse'] for r in selected.values()),
    'overnight_new_completed':len(observed),
    'overnight_not_completed_in_snapshot':len(selected)-len(completed_selected),
    'outside_overnight_scope':len(all_runs)-len(selected),
    'historical_completed_outside_overnight_scope':len(set(historical)-set(selected)),
    'completed_numeric_validation_records':sum(r['validation_score'] is not None for r in selected_records),
    'completed_strict_no_validation_records':sum(r['run_id'] in completed_selected and r['strict_no_validation'] for r in selected_records),
    'core_scale_pairs':scale_pairs, 'boundary_point_estimates':boundary_points,
    'core_imbalance_pairs':imbalance, 'semantic_module_counts':semantic_counts,
    'go_emotions_three_rare_label_gap_contribution':tail_contribution,
    'human_review_status':base['human_review_status'],
    'local_discovery_cases_manifest_exists':(FOLDER/'discovery-cases/manifest.json').exists(),
    'discovery_case_manifest':case_manifest,
    'discovery_prediction_rows':len(predictions),
    'discovery_unique_samples':len(frames),
    'case_trajectories_and_linear_decompositions_verified':len(cases),
    'discovery_transitions':transitions,
    'remote_discovery_case_outputs':'downloaded and verified; SVM only; human semantic causes remain unknown',
    'selection_lock_exists':(ROOT/'selection.lock.json').exists(),
    'evaluation_freeze_exists':(ROOT/'evaluation.freeze.json').exists(),
    'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in (base_path,extra_path,history_path,original_path)},
    'experiment_register_sha256':sha(FOLDER/'experiment-register.csv'),
    'new_training_submitted':False,'cluster_jobs_changed':False,
}
(FOLDER/'research-status.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:summary[k] for k in ('original_physical_requests','historical_verified_checkpoints',
    'completed_in_union_of_available_evidence','overnight_selected','overnight_completed',
    'overnight_new_completed','historical_completed_outside_overnight_scope',
    'completed_numeric_validation_records','completed_strict_no_validation_records')},ensure_ascii=False))
