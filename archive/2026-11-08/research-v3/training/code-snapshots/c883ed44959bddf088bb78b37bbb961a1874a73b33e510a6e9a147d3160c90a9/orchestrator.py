"""Single-worker E0 and declared anchor search over immutable prepared views."""
from __future__ import annotations

from dataclasses import asdict,replace
from datetime import datetime,timezone
import gc
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
import psutil

from .config import canonical_hash
from .execution import authorize_phase,run_lease
from .matrix import strict_hparams,training_policy
from .metrics import metric_bundle
from .storage import atomic_write_json,bounded_path,hash_file,read_json
from .train import fit_run,learner_code_hash,score_saved,target_matrix
from .types import Example,PreparedDataset,RunSpec,VisibleInput


def _ids_hash(ids):
    return hashlib.sha256('\n'.join(sorted(ids)).encode()).hexdigest()


def _spec(p,metadata,condition_id,model_id,hp,asset_hash,code_hash):
    hp={**hp,'device':p['runtime']['device'],'micro_batch':p['runtime']['micro_batch'],
        'cpu_threads':p['runtime']['cpu_threads'],'protocol_hash':p['_protocol_hash'],
        'label_count':len(metadata['label_names'])}
    seed=42 if model_id in {'bilstm','distilbert','modernbert'} else None
    run=RunSpec('',condition_id,model_id,seed,hp,metadata['inputs']['input_hash'],asset_hash,code_hash,False)
    return replace(run,run_id=canonical_hash(asdict(run)))


def engineering_spec(p,packet,model_id,metadata,asset_hash,code_hash,root):
    if model_id not in p['scope']['main_models']:raise ValueError('E0 uses the five main models')
    roles={'train':packet['training_ids'],'select':packet['selection_ids'],
           'discovery':packet['engineering_evaluation_ids']}
    sizes=p['experiments']['E0']
    for purpose,want in [('train',sizes['training_unique_rows']),('select',sizes['validation_rows']),('discovery',sizes['evaluation_rows'])]:
        if len(roles[purpose])!=want or len(set(roles[purpose]))!=want:
            raise ValueError('E0 role IDs violate registered unique-row budgets')
    if len(set().union(*map(set,roles.values())))!=sum(map(len,roles.values())):
        raise ValueError('E0 role overlap is forbidden')
    if packet['label_names']!=metadata['label_names'] or packet['label_count']!=len(metadata['label_names']):
        raise ValueError('E0 full target label mapping changed')
    condition_id=canonical_hash({'dataset':metadata['dataset_id'],'train_ids':sorted(roles['train']),
                                'pool':metadata['manifest_hash'],'purpose':'engineering'})
    record={'condition_id':condition_id,'dataset_id':metadata['dataset_id'],'train_ids':sorted(roles['train']),
            'metadata':{'pool_manifest_hash':metadata['manifest_hash'],'rows':len(roles['train'])}}
    atomic_write_json(Path(root)/'training/conditions'/f'{condition_id}.json',record,root)
    hp={**training_policy(model_id,p),**strict_hparams(model_id,p),
        'role_ids_hash':{name:_ids_hash(ids) for name,ids in roles.items()}}
    return _spec(p,metadata,condition_id,model_id,hp,asset_hash,code_hash)


def load_case(root,metadata,training_ids,selection_ids=None,discovery_ids=()):
    import pyarrow.parquet as pq
    root=Path(root);partitions={};all_ids=set()
    for purpose,requested in [('train',training_ids),('select',selection_ids),('discovery',discovery_ids)]:
        if requested is not None and not requested:
            partitions[purpose]=();continue
        path=bounded_path(root/metadata['partitions'][purpose]['file'],root)
        filters=None if requested is None else [('sample_id','in',list(requested))]
        table=pq.read_table(path,columns=['sample_id','labels','official_split','group_id'],filters=filters)
        rows=tuple(Example(metadata['dataset_id'],row['sample_id'],'',tuple(row['labels']),
                  row['official_split'],row['group_id']) for row in table.to_pylist())
        if requested is not None and {row.sample_id for row in rows}!=set(requested):
            raise ValueError(f'ID requested outside actual {purpose} partition')
        partitions[purpose]=rows;all_ids.update(row.sample_id for row in rows)
    partitions['test']=();partitions['quarantine']=()
    table=pq.read_table(bounded_path(root/metadata['inputs']['file'],root),
        columns=['sample_id','text','start_char','end_char','content_tokens','full_tokens','text_hash'],
        filters=[('sample_id','in',list(all_ids))])
    inputs={row['sample_id']:VisibleInput(**row) for row in table.to_pylist()}
    if set(inputs)!=all_ids:raise ValueError('Missing or extra common visible input rows')
    data=PreparedDataset(metadata['dataset_id'],metadata['task'],tuple(metadata['label_names']),
                         partitions,metadata['manifest_hash'])
    return data,inputs


def choose_candidate(records,task):
    if not records:raise ValueError('No completed candidates')
    metric='macro_f1' if task=='single_label' else 'macro_ap'
    if any(record['validation'].get(metric) is None or not np.isfinite(record['validation'][metric]) for record in records):
        raise ValueError('Anchor metric is missing or unsupported')
    return min(records,key=lambda record:(-record['validation'][metric],record['candidate_index']))


def _progress(root,phase,**kwargs):
    event={'phase':phase,'pid':os.getpid(),'process_created':psutil.Process().create_time(),
           'recorded_at_utc':datetime.now(timezone.utc).isoformat(),**kwargs}
    atomic_write_json(root/'training/progress.json',event,root)
    print(json.dumps(event,ensure_ascii=False),flush=True)


def _condition_ids(record,root):
    if 'train_ids' in record:return tuple(record['train_ids'])
    path=bounded_path(root/record['train_ids_file'],root)
    if hash_file(path)!=record['train_ids_file_sha256']:raise ValueError('Anchor condition IDs changed')
    with gzip.open(path,'rt',encoding='utf-8') as stream:return tuple(line.rstrip('\n') for line in stream)


def _export_scores(root,relative,run,rows,scores,score_kind):
    import pyarrow as pa
    import pyarrow.parquet as pq
    records=[]
    for row,score in zip(rows,scores):
        predicted=(np.flatnonzero(score>=(.5 if score_kind=='probability' else 0)).tolist()
                   if len(row.labels)>1 or run.hparams.get('target_task') in {'multilabel','multi_label'}
                   else [int(np.argmax(score))])
        records.append({'sample_id':row.sample_id,'group_id':row.group_id,'labels':list(row.labels),
            'scores':score.tolist(),'prediction':predicted,'score_kind':score_kind,'run_id':run.run_id,'input_hash':run.input_hash})
    path=bounded_path(root/relative,root);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.parquet.part')
    pq.write_table(pa.Table.from_pylist(records),temporary,compression='zstd');temporary.replace(path)
    return {'path':relative,'sha256':hash_file(path),'rows':len(records)}


def run_engineering(p,root,authorization):
    root=Path(root);permit=authorize_phase(p,root,authorization,'engineering')
    design=read_json(root/'matrix/design_manifest.json');prepared=read_json(root/'data/prepared_manifest.json')
    assets=read_json(root/'assets/assets.lock.json');packets=read_json(root/'engineering-smoke-inputs.json')
    if packets['design_hash']!=design['design_hash']:raise ValueError('E0 packet design identity changed')
    code_hash=learner_code_hash();cases=[];start=time.monotonic()
    with run_lease(root/'training/workers/engineering',canonical_hash({'permit':permit,'code_hash':code_hash})):
        for packet in packets['packets']:
            meta=prepared['datasets'][packet['dataset_id']]
            _progress(root,'engineering',status='loading_inputs',dataset=meta['dataset_id'],completed=len(cases),total=15)
            data,inputs=load_case(root,meta,packet['training_ids'],packet['selection_ids'],packet['engineering_evaluation_ids'])
            for model_id in p['scope']['main_models']:
                authorize_phase(p,root,authorization,'engineering')
                run=engineering_spec(p,packet,model_id,meta,assets['asset_hash'],code_hash,root)
                hp={**run.hparams,'target_task':data.task}
                if data.task!='single_label':
                    vy=target_matrix(data.partitions['select'],len(data.label_names),data.task)
                    hp['select_ap_mask']=[int(c) for c,n in enumerate(vy.sum(axis=0)) if 0<n<len(vy)]
                run=replace(run,hparams=hp,run_id='');run=replace(run,run_id=canonical_hash(asdict(run)))
                _progress(root,'engineering',status='fitting',dataset=data.dataset_id,model_id=model_id,
                          completed=len(cases),total=15,run_id=run.run_id)
                p['_training_started']=True
                result=fit_run(run,data,inputs,assets,root/'runs'/run.run_id)
                rows=data.partitions['discovery'];scores=score_saved(root/'runs'/run.run_id,tuple(inputs[row.sample_id] for row in rows))
                metrics=metric_bundle(target_matrix(rows,len(data.label_names),data.task),scores,data.task,result['score_kind'])
                saved=_export_scores(root,f'training/engineering/{run.run_id}.parquet',run,rows,scores,result['score_kind'])
                cases.append({'dataset_id':data.dataset_id,'model_id':model_id,'run_id':run.run_id,
                    'run_status':f'runs/{run.run_id}/status.json','run_status_sha256':hash_file(root/'runs'/run.run_id/'status.json'),
                    'reload_verified':result['reload_verified'],'training_rows':result['training_unique_rows'],
                    'validation_rows':len(data.partitions['select']),'evaluation_rows':len(rows),
                    'evaluation_source':'discovery','score_rows':saved,'engineering_metrics':metrics})
                atomic_write_json(root/'training/engineering-smoke.json',{'complete':len(cases)==15,'cases':cases,
                    'expected_cases':15,'design_hash':design['design_hash'],'learner_code_hash':code_hash,
                    'actual_model_training_run':True,'official_test_predictions_generated':False,
                    'no_inferential_claims':True,'seconds':time.monotonic()-start},root)
                _progress(root,'engineering',status='case_complete',dataset=data.dataset_id,model_id=model_id,
                          completed=len(cases),total=15,run_id=run.run_id)
            del data,inputs;gc.collect()
    _progress(root,'engineering',status='complete',completed=len(cases),total=15)
    return {'phase':'engineering','complete':True,'cases':len(cases),'actual_model_training_run':True,
            'official_test_predictions_generated':False}


def run_anchors(p,root,authorization):
    root=Path(root);permit=authorize_phase(p,root,authorization,'anchors')
    design=read_json(root/'matrix/design_manifest.json');prepared=read_json(root/'data/prepared_manifest.json')
    assets=read_json(root/'assets/assets.lock.json');code_hash=learner_code_hash()
    conditions={node['condition_id']:node for node in design['conditions']};choices={};all_candidates=[]
    with run_lease(root/'training/workers/anchors',canonical_hash({'permit':permit,'code_hash':code_hash})):
        for dataset in p['scope']['single_text_datasets']:
            meta=prepared['datasets'][dataset]
            nodes=[node for node in design['selection_nodes'] if node['dataset_id']==dataset]
            anchor=conditions[nodes[0]['anchor_condition_id']];ids=_condition_ids(anchor,root)
            _progress(root,'anchors',status='loading_inputs',dataset=dataset,completed=len(choices),total=39)
            data,inputs=load_case(root,meta,ids,selection_ids=None)
            roles_hash={'train':_ids_hash(ids),'select':_ids_hash([row.sample_id for row in data.partitions['select']]),
                        'discovery':_ids_hash([])}
            for node in nodes:
                authorize_phase(p,root,authorization,'anchors')
                records=[]
                for index,candidate in enumerate(node['candidates']):
                    hp={**training_policy(node['model_id'],p),**candidate,'role_ids_hash':roles_hash,'target_task':data.task}
                    if data.task!='single_label':hp['select_ap_mask']=meta['partitions']['select']['ap_mask']
                    run=_spec(p,meta,anchor['condition_id'],node['model_id'],hp,assets['asset_hash'],code_hash)
                    _progress(root,'anchors',status='fitting',dataset=dataset,model_id=node['model_id'],
                        candidate_index=index,completed_candidates=len(all_candidates),total_candidates=234,
                        completed=len(choices),total=39,run_id=run.run_id)
                    p['_training_started']=True
                    result=fit_run(run,data,inputs,assets,root/'runs'/run.run_id)
                    record={'candidate_index':index,'hparams':candidate,'run_id':run.run_id,
                        'validation':result['validation'],'selected_epoch':result['selected_epoch'],
                        'run_status_sha256':hash_file(root/'runs'/run.run_id/'status.json'),
                        'fit_seconds':result['fit_seconds'],'feature_seconds':result['feature_seconds'],
                        'total_seconds':result['total_seconds']}
                    records.append(record);all_candidates.append({**record,'node_id':node['node_id']})
                    atomic_write_json(root/'training/anchor-search-progress.json',{'complete':False,
                        'completed_candidates':len(all_candidates),'total_candidates':234,
                        'candidates':all_candidates,'choices':choices,'official_test_predictions_generated':False},root)
                selected=choose_candidate(records,data.task)
                choices[node['node_id']]={'node_id':node['node_id'],'dataset_id':dataset,'model_id':node['model_id'],
                    'status':'complete','hparams':selected['hparams'],'selected_run_id':selected['run_id'],
                    'selected_epoch':selected['selected_epoch'],'validation':selected['validation'],
                    'evidence_sha256':selected['run_status_sha256'],'candidates':records}
                atomic_write_json(root/'selections.json',{'status':'anchors_complete_pilots_pending' if len(choices)==39 else 'anchors_running',
                    'design_hash':design['design_hash'],'learner_code_hash':code_hash,'choices':choices,
                    'discovery_pilots':{node['pilot_id']:{'status':'not_run'} for node in design['discovery_pilot_nodes']},
                    'selection_lock_created':False,'official_test_predictions_generated':False},root)
            del data,inputs;gc.collect()
    _progress(root,'anchors',status='complete_pilots_pending',completed=len(choices),total=39,
              completed_candidates=len(all_candidates),total_candidates=234)
    return {'phase':'anchors','complete':True,'selection_nodes':len(choices),'candidates':len(all_candidates),
            'selection_lock_created':False,'official_test_predictions_generated':False}


def run_training_phase(p,root,phase,authorization):
    if phase=='engineering':return run_engineering(p,root,authorization)
    if phase=='anchors':return run_anchors(p,root,authorization)
    authorize_phase(p,root,authorization,phase)
    raise NotImplementedError('Main/mechanism training dispatch requires the resolved selection manifest')
