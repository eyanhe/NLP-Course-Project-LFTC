"""Pre-training experiment graph, concrete sample manifests, and a design lock.

Selections and D-pilots remain unresolved until they are actually performed.
This module never trains or produces official-test predictions.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import gzip
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .config import PROJECT_ROOT, canonical_hash
from .preparation import load_prepared_dataset
from .sampling import (_condition, _support, sample_scale, sample_kshot,
                       sample_long_tail, sample_multilabel, shared_tail_budget)
from .storage import atomic_write_json, bounded_path, hash_file, read_json
from .types import Condition, PreparedDataset, RunSpec

NEURAL={'bilstm','distilbert','modernbert','m2_distilbert','frozen_generic_distilbert_head'}


def candidate_grid(model: str,p: dict[str,Any]) -> list[dict[str,Any]]:
    t=p['training']
    if model in {'tfidf_lr','tfidf_svm'}:
        return [{'ngram_range':ng,'C':c} for ng in t['tfidf']['candidate_ngrams'] for c in t['tfidf']['candidate_C']]
    if model=='w2v_svm':
        return [{'C':c} for c in t['w2v']['svm_candidate_C']]
    if model=='bilstm':
        return [{'lr':lr,'dropout':d} for lr in t['bilstm']['candidate_lr'] for d in t['bilstm']['candidate_dropout']]
    return [{'lr':lr,'weight_decay':wd} for lr in t['distilbert']['candidate_lr']
            for wd in t['distilbert']['candidate_weight_decay']]


def strict_hparams(model: str,p: dict[str,Any]) -> dict[str,Any]:
    d=p['training']['strict_defaults']
    if model in {'tfidf_lr','tfidf_svm'}:
        return {'C':d['tfidf_C'],'ngram_range':d['tfidf_ngram'],'max_features':30000,'min_df':1}
    if model=='w2v_svm':
        return {'C':d['w2v_svm_C'],**{k:v for k,v in p['training']['w2v'].items() if k!='svm_candidate_C'}}
    if model=='bilstm':
        return {'lr':d['bilstm_lr'],'dropout':d['bilstm_dropout'],'epochs':10,
                'vocabulary_limit':30000,'effective_batch':64,'min_count':1}
    return {'lr':d['distilbert_lr'],'weight_decay':d['distilbert_weight_decay'],
            'epochs':5,'effective_batch':16,'warmup_ratio':.1}


def training_policy(model: str,p: dict[str,Any]) -> dict[str,Any]:
    """Fixed effective settings carried alongside the selected grid values."""
    t=p['training']
    tfidf={k:v for k,v in t['tfidf'].items() if not k.startswith('candidate_')}
    w2v={k:v for k,v in t['w2v'].items() if k!='svm_candidate_C'}
    if model in {'tfidf_lr','tfidf_svm','fixed_tfidf_svm'}:
        return tfidf
    if model=='w2v_svm':
        return w2v
    if model=='bilstm':
        return {**{k:v for k,v in t['bilstm'].items() if not k.startswith('candidate_')},
                'vocabulary_limit':w2v['vocabulary_limit'],'embedding_initialization':w2v}
    if model in {'distilbert','modernbert'}:
        return {k:v for k,v in t['distilbert'].items() if not k.startswith('candidate_')}
    return {}


def build_design_matrix(data: dict[str,PreparedDataset],p: dict[str,Any],
                        condition_sink: Callable[[Condition],dict]|None=None) -> dict[str,Any]:
    expected=set(p['scope']['single_text_datasets'])
    if set(data)!=expected:
        raise ValueError('Design requires all seven registered datasets')
    conditions={};physical={};logical=[];selection_nodes=[];infeasible=[]
    # Recompute from public configuration rather than trusting an earlier
    # cached _protocol_hash after an in-memory edit.
    protocol_hash=canonical_hash(p)
    seeds=p['seeds']['data'];training_seeds=p['seeds']['training']
    anchor=p['experiments']['E1']['anchor']
    main_models=p['scope']['main_models']
    def keep(condition):
        if condition.condition_id not in conditions:
            record={'condition_id':condition.condition_id,'experiment_id':condition.experiment_id,
                'dataset_id':condition.dataset_id,'data_seed':condition.data_seed,
                'metadata':condition.metadata,
                'training_set_hash':hashlib.sha256('\n'.join(sorted(condition.train_ids)).encode()).hexdigest()}
            if condition_sink:
                record.update(condition_sink(condition))
            else:
                record['train_ids']=list(condition.train_ids)
            conditions[condition.condition_id]=record
        return conditions[condition.condition_id]
    def runs(condition,experiment,models=None,variant=None,binding=None,train_seeds=None,context=None):
        record=keep(condition)
        for model in models or main_models:
            strict=experiment=='E2'
            hp_binding=binding or ({'fixed':strict_hparams(model,p)} if strict else
                                  {'selection_node':f'select:{condition.dataset_id}:{model}'})
            for training_seed in ((train_seeds or training_seeds) if model in NEURAL else [None]):
                parameters={'dataset_id':condition.dataset_id,'training_set_hash':record['training_set_hash'],
                    'model_id':model,'training_seed':training_seed,'hparam_binding':hp_binding,
                    'protocol_hash':protocol_hash,
                    'training_policy':strict_hparams(model,p) if strict else training_policy(model,p),
                    'variant':variant or {},'strict_no_val':strict,
                    'input_hash':p.get('_input_hashes',{}).get(condition.dataset_id,'fixture'),
                    'asset_hash':p.get('_asset_hash','fixture'),
                    'training_input_order':'canonical_sample_id_before_seeded_epoch_shuffle'}
                run_id=canonical_hash(parameters)
                if run_id not in physical:
                    physical[run_id]={'run_id':run_id,**parameters,
                        'condition_id':condition.condition_id,'status':'planned_not_run',
                        'phase':'strict_fewshot' if strict else 'post_selection',
                        'requires':['selection.lock'],
                        'effective_batch':64 if model=='bilstm' else 16 if model in NEURAL else None}
                logical.append({'experiment_id':experiment,'dataset_id':condition.dataset_id,
                    'condition_id':condition.condition_id,'run_id':run_id,
                    'context':context or {},'model_id':model})
    def attempt(dataset,experiment,parameters,build):
        try:
            return build()
        except ValueError as exc:
            infeasible.append({'dataset_id':dataset,'experiment_id':experiment,
                               'parameters':parameters,'reason':str(exc)})
            return None
    for dataset,d in data.items():
        anchor_condition=attempt(dataset,'anchor',{'n':anchor},lambda:sample_scale(d,anchor,seeds[0]))
        if anchor_condition:
            keep(anchor_condition)
        for model in main_models+(['modernbert'] if dataset in p['experiments']['E5']['datasets'] else []):
            selection_nodes.append({'node_id':f'select:{dataset}:{model}','dataset_id':dataset,'model_id':model,
                'anchor_condition_id':anchor_condition.condition_id if anchor_condition else None,
                'data_seed':2026,'training_seed':42 if model in NEURAL else None,
                'candidates':candidate_grid(model,p),'status':'awaiting_anchor_training',
                'selection_partition':'select','test_access':False,'requires':['design.lock'],
                'metric':'macro_ap' if d.task in {'multilabel','multi_label'} else 'macro_f1',
                'tie_break':'declared_grid_order_then_earliest_epoch'})
        for requested in p['experiments']['E1']['requested_sizes']:
            for seed in ([seeds[0]] if requested=='full' or requested==len(d.partitions['train']) else seeds):
                c=attempt(dataset,'E1',{'n':requested,'seed':seed},lambda:sample_scale(d,requested,seed))
                if c:
                    runs(c,'E1')
        if dataset in p['experiments']['E2']['datasets']:
            for k,seed in itertools.product(p['experiments']['E2']['shots_per_class'],seeds):
                c=attempt(dataset,'E2',{'k':k,'seed':seed},lambda:sample_kshot(d,k,seed))
                if c:
                    runs(c,'E2')
    tail_plans={}
    for dataset in p['experiments']['E3']['datasets']:
        d=data[dataset];classes=len(d.label_names)
        orders=tuple(tuple(int(x) for x in np.random.default_rng(seed).permutation(classes))
                     for seed in p['seeds']['head_permutation'])
        n=attempt(dataset,'E3',{},lambda:shared_tail_budget(tuple(_support(d)),orders,
                  tuple(p['experiments']['E3']['ratios']),p['experiments']['E3']['budget_cap']))
        if n is None:
            continue
        tail_plans[dataset]={'common_n':n,'orders':[list(o) for o in orders],
                            'head_seeds':p['seeds']['head_permutation']}
        for rho,seed in itertools.product(p['experiments']['E3']['ratios'],seeds):
            for index,order in enumerate(orders[:1] if rho==1 else orders):
                c=sample_long_tail(d,n,rho,order,seed)
                runs(c,'E3',context={'head_seed':None if rho==1 else p['seeds']['head_permutation'][index],
                                    'head_order':list(order),'ratio':rho})
    go=data[p['experiments']['E4']['dataset']]
    for alpha,seed in itertools.product(p['experiments']['E4']['alpha'],seeds):
        c=attempt(go.dataset_id,'E4',{'alpha':alpha,'seed':seed},
                   lambda:sample_multilabel(go,p['experiments']['E4']['rows'],alpha,seed))
        if c:
            runs(c,'E4',context={'alpha':alpha})
    for dataset in p['experiments']['E5']['datasets']:
        d=data[dataset]
        for requested in p['experiments']['E5']['sizes']:
            n=min(100000,len(d.partitions['train'])) if requested=='min_100000_or_full' else int(requested)
            for seed in (seeds[:1] if n==len(d.partitions['train']) else seeds):
                c=attempt(dataset,'E5',{'n':n,'seed':seed},lambda:sample_scale(d,n,seed))
                if c:
                    runs(c,'E5',['modernbert','distilbert'])
    mechanisms=p['experiments']['E6']['mechanism_rules']
    m2=mechanisms['M2']
    for dataset,n,seed in itertools.product(m2['datasets'],m2['sizes'],seeds):
        c=attempt(dataset,'M2',{'n':n,'seed':seed},lambda:sample_scale(data[dataset],n,seed))
        if not c:
            continue
        for arm in m2['arms']:
            binding={'fixed':{'arm':arm,'head_lr':.001,'encoder_lr':.00002,'weight_decay':.01,
                 'epochs':10,'effective_batch':16,'same_head_state_across_arms':True,
                 'head_architecture':'DistilBertForSequenceClassification',
                 'select_checkpoint':'select_metric_but_train_all_epochs'}}
            runs(c,'M2',['m2_distilbert'],binding=binding,context={'arm':arm})
        for lr in [.00001,.00005,.0005]:
            hp={**binding['fixed'],'arm':'random_unfrozen','encoder_lr':lr}
            runs(c,'M2',['m2_distilbert'],binding={'fixed':hp},context={'arm':'random_unfrozen_lr_sensitivity'})
    m3=mechanisms['M3']
    for dataset in m3['datasets']:
        if dataset not in tail_plans:
            continue
        d=data[dataset];n=tail_plans[dataset]['common_n'];classes=len(d.label_names)
        for seed,order,rho in itertools.product(seeds,tail_plans[dataset]['orders'],m3['ratios']):
            order=tuple(order)
            reference=sample_long_tail(d,n,1,order,seed)
            lookup={r.sample_id:r for r in d.partitions['train']}
            grouped=[[] for _ in range(classes)]
            for sid in reference.train_ids:
                row=lookup[sid];grouped[row.labels[0]].append(row)
            tail=set(order[-max(1,classes//3):])
            for support,risk in itertools.product(['full','tail_reduced'],['uniform','skew']):
                rows=tuple(row for label,group in enumerate(grouped)
                    for row in (group[:max(1,len(group)//rho)] if support=='tail_reduced' and label in tail else group))
                q=np.ones(classes,dtype=float)
                if risk=='skew':
                    for rank,label in enumerate(order):
                        q[label]=rho**(-rank/(classes-1))
                q/=q.sum()
                c=_condition(d,rows,'M3',seed,{'reference_n':n,'support':support,'risk':risk,
                    'head_order':list(order),'rho':rho,'current_tail_labels':sorted(tail),
                    'target_q':q.tolist(),'representation_unlabeled_pool_hash':d.manifest_hash})
                for model in m3['models']:
                    hp={'representation':'T_unlabeled_tfidf' if model=='fixed_tfidf_svm' else 'generic_frozen_distilbert',
                        'target_q':q.tolist(),'reference_n':n,'C':1,'tol':.0001,'max_iter':10000,
                        'head_lr':.001,'updates':int(np.ceil(n/16))*5,'effective_batch':16,
                        'per_row_weight':'N0*q_c/n_c' if model=='fixed_tfidf_svm' else 'q_c/p_emp_c',
                        'unlabeled_pool_manifest':d.manifest_hash}
                    runs(c,'M3',[model],binding={'fixed':hp},context={'support':support,'risk':risk,'rho':rho,'order':list(order)})
    m5=mechanisms['M5']
    for dataset,requested in itertools.product(m5['datasets'],m5['sizes']):
        d=data[dataset]
        for seed in seeds[:1] if requested=='full' else seeds:
            c=attempt(dataset,'M5',{'n':requested,'seed':seed},lambda:sample_scale(d,requested,seed))
            if not c:
                continue
            for model in m5['models']:
                if model in {'tfidf_lr','tfidf_svm','w2v_svm'}:
                    variants=[{} , {'max_features':100000} if model!='w2v_svm' else {'vocabulary_limit':100000}]
                elif model=='bilstm':
                    variants=[{**({'vocabulary_limit':cap} if cap!=30000 else {}),
                               **({'max_epochs':ep} if ep!=10 else {})} for cap,ep in itertools.product([30000,100000],[10,20])]
                else:
                    variants=[{}, {'max_epochs':10}]
                for variant in variants:
                    runs(c,'M5',[model],variant=variant)
    pilots=[{'pilot_id':'Bank_M1_SVM','dataset_id':'banking77','mechanism':'M1'},
            {'pilot_id':'Yelp_M4_DistilBERT','dataset_id':'yelp','mechanism':'M4'},
            {'pilot_id':'Yahoo_M4_DistilBERT','dataset_id':'yahoo_answers_topics','mechanism':'M4'}]
    for node in pilots:
        node.update({'status':'not_run','partition':'discovery','pairs':50,'human_review_required':True,
                     'requires':['anchor_checkpoints','design.lock']})
    e0=[{'dataset_id':dataset,'model_id':model,'training_rows':256,'validation_rows':128,
         'evaluation_rows':128,'evaluation_source':'discovery','status':'not_run',
         'engineering_only':True,'test_access':False} for dataset in p['scope']['new_datasets'] for model in main_models]
    result={'schema_version':1,'status':'design_locked_selection_pending','protocol_hash':protocol_hash,
        'conditions':list(conditions.values()),'logical_runs':logical,'physical_runs':list(physical.values()),
        'selection_nodes':selection_nodes,'engineering_smoke_nodes':e0,'discovery_pilot_nodes':pilots,
        'mechanism_nodes':{name:{'rule':rule,'status':'configured_not_run'} for name,rule in mechanisms.items()},
        'tail_plans':tail_plans,'infeasible_conditions':infeasible,
        'dependency_graph':{'design.lock':['pinned_assets','prepared_partitions','shared_inputs','structural_matrix'],
            'selection.lock':['main_anchor_selections','modern_anchor_selections','three_discovery_pilots',
                              'resolved_manifest','all_primary_contrasts'],
            'evaluation.freeze':['all_E1_to_E5_and_M2_M3_M5_checkpoints_complete','thresholds','contrasts']},
        'budget':{'post_selection_unique_fit_nodes':len(physical),'logical_run_nodes':len(logical),
            'anchor_candidate_fit_nodes':sum(len(n['candidates']) for n in selection_nodes),
            'engineering_fit_nodes':len(e0),'post_selection_by_model':dict(Counter(r['model_id'] for r in physical.values())),
            'exact_fit_count_after_selection_pending':True,
            'raw_float32_main_test_score_bytes':sum(
                p.get('_test_rows',{}).get(r['dataset_id'],0)*len(data[r['dataset_id']].label_names)*4
                for r in physical.values()),'no_wall_clock_estimate':True},
        'actual_model_training_run':False,'model_predictions_generated':False,
        'selection_lock_created':False,'evaluation_freeze_created':False}
    result['protocol_clarifications']={
        'document':'docs/superpowers/specs/2026-11-07-preparation-clarifications.md',
        'current_tail_size':'max(1,floor(C/3))','natural_tail':'boundary_ties_included; NA_if_all_equal',
        'balanced_tail':'NA; contrast_uses_the_paired_skew_head_order',
        'task_aliases':{'multilabel':'multi_label'},
        'future_statistics_use_explicit_condition_label_ids':True}
    result['design_hash']=canonical_hash(result)
    return result


def resolve_runs(design: dict[str,Any],selections: dict[str,Any],assets: dict[str,Any],code_hash: str) -> tuple[RunSpec,...]:
    if not all(node['node_id'] in selections for node in design['selection_nodes']):
        raise ValueError('Unresolved anchor selections; selection.lock cannot be created')
    if not all(selections.get('discovery_pilots',{}).get(node['pilot_id'],{}).get('status')=='complete'
               for node in design['discovery_pilot_nodes']):
        raise ValueError('Unresolved discovery pilots')
    if not selections.get('all_primary_contrasts'):
        raise ValueError('Unresolved primary contrasts')
    resolved=[]
    for run in design['physical_runs']:
        binding=run['hparam_binding']
        hp=binding.get('fixed')
        if hp is None:
            node=next(n for n in design['selection_nodes'] if n['node_id']==binding['selection_node'])
            selected=selections[node['node_id']]
            if selected.get('hparams') not in node['candidates'] or not selected.get('evidence_sha256'):
                raise ValueError('Selection has no declared candidate or verified evidence')
            hp=selected['hparams']
        hp={**run['training_policy'],**hp,**run['variant']}
        identity=canonical_hash({**run,'hparams':hp,'code_hash':code_hash,'asset_hash':assets.get('asset_hash')})
        resolved.append(RunSpec(identity,run['condition_id'],run['model_id'],run['training_seed'],hp,
                      run['input_hash'],assets['asset_hash'],code_hash,run['strict_no_val']))
    return tuple(resolved)


def write_design_lock(root: Path,resources: dict[str,str],bindings: dict[str,Any]) -> dict[str,Any]:
    files={name:{'path':path,'sha256':hash_file(bounded_path(Path(root)/path,root))} for name,path in resources.items()}
    lock={'schema_version':1,'stage':'design','files':files,'bindings':bindings,
          'allows':'anchor_selection_and_D_pilots_only','allows_long_batches':False,'allows_test_predictions':False}
    lock['lock_hash']=canonical_hash(lock)
    atomic_write_json(Path(root)/'design.lock.json',lock,root)
    return lock


def verify_design_lock(root: Path) -> dict[str,Any]:
    lock=read_json(Path(root)/'design.lock.json')
    if canonical_hash({k:v for k,v in lock.items() if k!='lock_hash'})!=lock['lock_hash']:
        raise ValueError('Design lock hash changed')
    for record in lock['files'].values():
        path=bounded_path(Path(root)/record['path'],root)
        if not path.is_file() or hash_file(path)!=record['sha256']:
            raise ValueError(f'Design resource hash changed: {path}')
    for relative,expected in lock['bindings'].get('project_files',{}).items():
        path=bounded_path(PROJECT_ROOT/relative,PROJECT_ROOT)
        if not path.is_file() or hash_file(path)!=expected:
            raise ValueError(f'Design source/config hash changed: {relative}')
    return lock


def build_saved_matrix(p: dict[str,Any],root: Path,stage: str) -> dict[str,Any]:
    if stage!='design':
        raise ValueError('Only structural design is allowed before actual anchor selections and D pilots')
    prepared=read_json(root/'data/prepared_manifest.json')
    if not prepared.get('complete') or prepared['identity']['protocol_hash']!=p['_protocol_hash']:
        raise ValueError('Prepared data are incomplete or use a different protocol')
    data={name:load_prepared_dataset(root,meta,train_only=True) for name,meta in prepared['datasets'].items()}
    runtime={**p,'_input_hashes':{name:meta['inputs']['input_hash'] for name,meta in prepared['datasets'].items()},
             '_test_rows':{name:meta['partitions']['test']['rows'] for name,meta in prepared['datasets'].items()},
             '_asset_hash':prepared['identity']['asset_hash']}
    resources={'prepared_manifest':'data/prepared_manifest.json','asset_lock':'assets/assets.lock.json',
               'audit_schema':'audit/annotation.schema.json','human_review_protocol':'audit/review-protocol.md'}
    def sink(condition):
        relative=f'matrix/conditions/{condition.condition_id}.ids.gz'
        path=root/relative;path.parent.mkdir(parents=True,exist_ok=True)
        temporary=path.with_suffix('.gz.part')
        with temporary.open('wb') as raw:
            with gzip.GzipFile(filename='',fileobj=raw,mode='wb',mtime=0) as stream:
                for sid in condition.train_ids:
                    stream.write((sid+'\n').encode())
        temporary.replace(path)
        resources[f'condition:{condition.condition_id}']=relative
        return {'train_ids_file':relative,'train_ids_file_sha256':hash_file(path)}
    design=build_design_matrix(data,runtime,sink)
    for name,meta in prepared['datasets'].items():
        resources[f'input:{name}']=meta['inputs']['file']
        for purpose,record in meta['partitions'].items():
            resources[f'partition:{name}:{purpose}']=record['file']
    atomic_write_json(root/'matrix/design_manifest.json',design,root)
    resources['design_manifest']='matrix/design_manifest.json'
    project_files={path.relative_to(PROJECT_ROOT).as_posix():hash_file(path)
                   for path in sorted((PROJECT_ROOT/'research_v3').glob('*.py'))}
    for path in [Path(p['_config_path']),PROJECT_ROOT/'requirements-research-lock.txt',
                 PROJECT_ROOT/'docs/superpowers/specs/2026-11-07-text-classification-research-design.md',
                 PROJECT_ROOT/'docs/superpowers/specs/2026-11-07-preparation-clarifications.md']:
        project_files[path.relative_to(PROJECT_ROOT).as_posix()]=hash_file(path)
    for record in read_json(root/'assets/assets.lock.json')['files']:
        project_files[record['path']]=record['sha256']
    lock=write_design_lock(root,resources,{'protocol_hash':p['_protocol_hash'],
        'prepared_hash':prepared['prepared_hash'],'asset_hash':prepared['identity']['asset_hash'],
        'design_hash':design['design_hash'],'project_files':project_files})
    return {'stage':'G2a','design_hash':design['design_hash'],'lock_hash':lock['lock_hash'],
            'conditions':len(design['conditions']),'budget':design['budget'],
            'selection_lock_created':False,'actual_model_training_run':False}
