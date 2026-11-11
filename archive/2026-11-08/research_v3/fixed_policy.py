"""User amendment: fixed defaults, no remaining hyperparameter search."""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime,timezone
import hashlib
from pathlib import Path
import shutil

from .config import PROJECT_ROOT,canonical_hash,load_protocol,output_root
from .matrix import strict_hparams,training_policy
from .orchestrator import _spec,_condition_ids,_ids_hash,load_case
from .storage import atomic_write_json,read_json,hash_file
from .train import learner_code_hash,fit_run
from .types import RunSpec

EMPTY_IDS_HASH=hashlib.sha256(b'').hexdigest()
POLICY_FILE='training/fixed-hyperparameters.json'


def default_candidate(model,p,grid):
    defaults=strict_hparams(model,p)
    matches=[row for row in grid if all(defaults.get(k)==v for k,v in row.items())]
    if len(matches)!=1:raise ValueError(f'Exactly one predeclared default configuration is required: {model}')
    return deepcopy(matches[0])


def build_fixed_policy(p,design,prepared,selection_hashes,asset_hash,code_hash,user_instruction):
    if not user_instruction:raise PermissionError('A user amendment is required to replace the search')
    if design['protocol_hash']!=p['_protocol_hash']:raise ValueError('Fixed policy protocol identity changed')
    metadata=prepared.get('datasets',prepared);conditions={c['condition_id']:c for c in design['conditions']}
    nodes={};defaults={}
    for node in design['selection_nodes']:
        ds=node['dataset_id'];model=node['model_id'];meta=metadata[ds]
        candidate=default_candidate(model,p,node['candidates']);defaults[model]=candidate
        hp={**training_policy(model,p),**candidate,'role_ids_hash':{
            'train':conditions[node['anchor_condition_id']]['training_set_hash'],
            'select':selection_hashes[ds],'discovery':EMPTY_IDS_HASH},'target_task':meta['task']}
        if meta['task']!='single_label':hp['select_ap_mask']=meta['partitions']['select']['ap_mask']
        run=_spec(p,meta,node['anchor_condition_id'],model,hp,asset_hash,code_hash)
        nodes[node['node_id']]={'node_id':node['node_id'],'dataset_id':ds,'model_id':model,
            'hparams':candidate,'run':asdict(run)}
    policy={'schema_version':1,'mode':'fixed_defaults','user_instruction':user_instruction,
        'protocol_hash':p['_protocol_hash'],'design_hash':design['design_hash'],
        'learner_code_hash':code_hash,'asset_hash':asset_hash,'model_defaults':defaults,'nodes':nodes,
        'grid_search_enabled':False,'validation_selects_hyperparameters':False,
        'checkpoint_rule':'registered_full_epoch_budget_with_existing_V_checkpoint_rule',
        'checkpoint_training_required_to_resolve_configuration':False,
        'only_required_pilot_checkpoints_before_selection_lock':True,'official_test_predictions_generated':False}
    policy['policy_hash']=canonical_hash(policy)
    return policy


def fixed_choices(root,policy):
    root=Path(root);evidence=hash_file(root/POLICY_FILE);choices={}
    for nid,row in policy['nodes'].items():
        run=row['run'];path=root/'runs'/run['run_id']/'status.json';status=read_json(path) if path.exists() else {}
        ready=status.get('status')=='complete' and status.get('reload_verified') is True
        if ready and status.get('run')!=run:raise ValueError('Existing default checkpoint request identity changed')
        choices[nid]={'node_id':nid,'dataset_id':row['dataset_id'],'model_id':row['model_id'],
            'status':'complete','configuration_complete':True,'selection_policy':'fixed_defaults',
            'hparams':deepcopy(row['hparams']),'selected_run_id':run['run_id'],
            'evidence_sha256':evidence,'parameter_policy_hash':policy['policy_hash'],
            'checkpoint_status':'complete' if ready else 'pending',
            'checkpoint_evidence_sha256':hash_file(path) if ready else None,
            'selected_epoch':status.get('selected_epoch') if ready else None,
            'validation':status.get('validation') if ready else None,'candidates':[]}
    return choices


def verify_fixed_choice(root,node,choice,files):
    root=Path(root);path=root/POLICY_FILE;policy=read_json(path)
    if (policy.get('mode')!='fixed_defaults' or policy.get('grid_search_enabled') is not False
            or not policy.get('user_instruction') or policy.get('policy_hash')!=canonical_hash({k:v for k,v in policy.items() if k!='policy_hash'})
            or policy.get('learner_code_hash')!=learner_code_hash()):
        raise ValueError('Fixed configuration policy identity changed')
    p=load_protocol(PROJECT_ROOT/'configs/research-v3.json')
    if policy['protocol_hash']!=p['_protocol_hash']:raise ValueError('Fixed policy source configuration changed')
    expected=default_candidate(node['model_id'],p,node['candidates']);row=policy['nodes'][node['node_id']]
    if (choice.get('status')!='complete' or choice.get('hparams')!=expected or row['hparams']!=expected
            or choice.get('selected_run_id')!=row['run']['run_id'] or choice.get('parameter_policy_hash')!=policy['policy_hash']
            or choice.get('evidence_sha256')!=hash_file(path) or row['run']['condition_id']!=node['anchor_condition_id']):
        raise ValueError('Fixed configuration differs from the user-authorized defaults')
    files['fixed_hyperparameter_policy']={'path':POLICY_FILE,'sha256':hash_file(path)}
    return {'configuration_verified':True,'grid_search_required':False,'checkpoint_status':choice['checkpoint_status']}


def prepare_fixed_policy(p,root,user_instruction):
    import pyarrow.parquet as pq
    root=Path(root);design=read_json(root/'matrix/design_manifest.json');prepared=read_json(root/'data/prepared_manifest.json')
    assets=read_json(root/'assets/assets.lock.json');hashes={}
    for ds,meta in prepared['datasets'].items():
        ids=pq.read_table(root/meta['partitions']['select']['file'],columns=['sample_id']).column(0).to_pylist()
        hashes[ds]=_ids_hash(ids)
    policy=build_fixed_policy(p,design,prepared,hashes,assets['asset_hash'],learner_code_hash(),user_instruction)
    if (root/POLICY_FILE).exists() and read_json(root/POLICY_FILE)!=policy:raise ValueError('Existing fixed parameter policy differs')
    atomic_write_json(root/POLICY_FILE,policy,root)
    previous=read_json(root/'selections.json');history=root/'history/hyperparameter-search/selections.json'
    if not history.exists():
        history.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(root/'selections.json',history)
    choices=fixed_choices(root,policy)
    result={'status':'fixed_parameters_ready_pilots_pending','design_hash':design['design_hash'],
        'learner_code_hash':learner_code_hash(),'selection_policy':'fixed_defaults','choices':choices,
        'discovery_pilots':previous.get('discovery_pilots',{}),'selection_lock_created':False,
        'official_test_predictions_generated':False}
    atomic_write_json(root/'selections.json',result,root)
    return result


def fit_needed_pilot_baselines(p,root,authorization):
    from .execution import authorize_phase,run_lease
    from .freeze import _checkpoint
    root=Path(root);authorize_phase(p,root,authorization,'fixed_baselines')
    policy=read_json(root/POLICY_FILE);design=read_json(root/'matrix/design_manifest.json')
    prepared=read_json(root/'data/prepared_manifest.json');assets=read_json(root/'assets/assets.lock.json')
    conditions={c['condition_id']:c for c in design['conditions']}
    with run_lease(root/'training/workers/fixed-baselines',policy['policy_hash']) as owner:
        for node in design['discovery_pilot_nodes']:
            model='tfidf_svm' if node['mechanism']=='M1' else 'distilbert'
            nid=f'select:{node["dataset_id"]}:{model}';run=RunSpec(**policy['nodes'][nid]['run'])
            status_path=root/'runs'/run.run_id/'status.json'
            if status_path.exists() and read_json(status_path).get('status')=='complete':_checkpoint(root,run.run_id,{})
            else:
                atomic_write_json(root/'training/progress.json',{'phase':'fixed_pilot_baselines','status':'fitting',
                    'dataset':node['dataset_id'],'model_id':model,'run_id':run.run_id,'grid_search_enabled':False,
                    'pid':owner['pid'],'process_created':owner['process_created'],
                    'recorded_at_utc':datetime.now(timezone.utc).isoformat()},root)
                ids=_condition_ids(conditions[run.condition_id],root)
                data,inputs=load_case(root,prepared['datasets'][node['dataset_id']],ids,selection_ids=None,discovery_ids=())
                fit_run(run,data,inputs,assets,root/'runs'/run.run_id)
            selections=read_json(root/'selections.json');selections['choices']=fixed_choices(root,policy)
            atomic_write_json(root/'selections.json',selections,root)
    atomic_write_json(root/'training/progress.json',{'phase':'fixed_parameters','status':'pilot_reviews_pending',
        'grid_search_enabled':False,'configurations_fixed':len(policy['nodes']),
        'recorded_at_utc':datetime.now(timezone.utc).isoformat()},root)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--fit-needed-pilots',action='store_true');args=parser.parse_args()
    p=load_protocol(PROJECT_ROOT/'configs/research-v3.json');root=output_root(p)
    if args.prepare:prepare_fixed_policy(p,root,'这部分超参数固定下来不就行了，浪费很多时间了')
    if args.fit_needed_pilots:fit_needed_pilot_baselines(p,root,root/'training/authorization.json')


if __name__=='__main__':main()
