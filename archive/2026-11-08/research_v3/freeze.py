"""Whole-study selection/evaluation locks; no rolling checkpoint exemptions."""
from __future__ import annotations

from datetime import datetime,timezone
import json
import hashlib
import math
from pathlib import Path
from typing import Any
import numpy as np

from .config import PROJECT_ROOT,canonical_hash
from .interventions import plan_pair_count,validate_pair
from .matrix import verify_design_lock
from .storage import atomic_write_json,bounded_path,hash_file,read_json


def _path(root,stage):
    if stage not in {'selection','evaluation'}:raise ValueError('Unknown global freeze stage')
    return Path(root)/('selection.lock.json' if stage=='selection' else 'evaluation.freeze.json')


def _sources():
    return {path.relative_to(PROJECT_ROOT).as_posix():hash_file(path)
            for path in sorted((PROJECT_ROOT/'research_v3').glob('*.py'))}


def _pin(root,files,name,path):
    path=bounded_path(path,root)
    if not path.is_file():raise ValueError(f'Required freeze artifact missing: {path}')
    files[name]={'path':path.relative_to(root).as_posix(),'sha256':hash_file(path)}


def _checkpoint(root,artifact_id,files,expected_request=None,expected_model_request=None):
    if not isinstance(artifact_id,str) or Path(artifact_id).name!=artifact_id or artifact_id in {'.','..'}:
        raise ValueError('Invalid checkpoint identity')
    path=Path(root)/'runs'/artifact_id/'status.json'
    if not path.exists():raise PermissionError(f'Required checkpoint is incomplete/pending: {artifact_id}')
    status=read_json(path)
    if status['status']!='complete' or not status.get('reload_verified'):
        raise PermissionError(f'Required checkpoint reload is incomplete: {artifact_id}')
    if expected_request is not None and status.get('request_hash')!=expected_request:
        raise ValueError(f'Checkpoint request identity changed: {artifact_id}')
    if expected_model_request is not None and canonical_hash(status.get('run'))!=expected_model_request:
        raise ValueError(f'Checkpoint model request identity changed: {artifact_id}')
    if not status.get('files'):raise ValueError('Checkpoint has no saved artifact files')
    _pin(root,files,f'checkpoint:{artifact_id}:status',path)
    for relative,expected in status['files'].items():
        target=bounded_path(path.parent/relative,path.parent)
        if not target.is_file() or hash_file(target)!=expected:
            raise ValueError(f'Checkpoint artifact hash changed: {artifact_id}/{relative}')
        _pin(root,files,f'checkpoint:{artifact_id}:{relative}',target)
    return status


def assert_resolved_coverage(design,manifest):
    if manifest.get('status')!='resolved' or manifest.get('unresolved_nodes'):
        raise PermissionError('Resolved manifest still has incomplete selections/pilots')
    runs=manifest.get('runs',[])
    planned={row['run_id'] for row in design['physical_runs']}
    actual=[row.get('planned_run_id',row['run_id']) for row in runs]
    if len(actual)!=len(set(actual)):raise ValueError('Duplicate planned run identities in global manifest')
    if set(actual)!=planned:raise PermissionError('Global resolved run coverage is incomplete, including mechanism runs')
    return runs


def verify_token_budget(root,entry,files):
    artifact=entry.get('artifact_id',entry['run_id'])
    path=bounded_path(Path(root)/'training/token-budgets'/f'{artifact}.json',root)
    if not path.is_file():raise PermissionError('ModernBERT own-token budget is pending')
    budget=read_json(path);run=entry['run']
    if (budget.get('run_id')!=artifact or budget.get('run_request_hash')!=canonical_hash(run)
            or budget.get('input_hash')!=run['input_hash'] or budget.get('asset_hash')!=run['asset_hash']
            or budget.get('record_hash')!=canonical_hash({k:v for k,v in budget.items() if k!='record_hash'})):
        raise ValueError('ModernBERT token budget request identity/hash changed')
    if (budget.get('model_id')!='modernbert' or not isinstance(budget.get('roles'),dict)
            or set(budget['roles'])!={'train','select'} or not isinstance(budget.get('role_input_identity'),dict)
            or set(budget['role_input_identity'])!={'train','select'} or budget.get('includes_padding') is not False
            or budget.get('training_timing_changed') is not False or budget.get('official_test_predictions_generated') is not False):
        raise ValueError('ModernBERT token budget schema/counts are missing or invalid')
    elapsed=budget.get('auxiliary_tokenization_seconds')
    if isinstance(elapsed,bool) or not isinstance(elapsed,(int,float)) or not math.isfinite(elapsed) or elapsed<0:
        raise ValueError('ModernBERT token budget timing is invalid')
    try:
        recorded=datetime.fromisoformat(budget['recorded_at_utc'])
        if recorded.tzinfo is None:raise ValueError('Timezone missing')
    except (KeyError,TypeError,ValueError) as exc:raise ValueError('ModernBERT token budget timestamp is invalid') from exc
    role_ids={}
    for role,counts in budget['roles'].items():
        identities=budget['role_input_identity'][role]
        if (not isinstance(identities,list) or any(not isinstance(row,list) or len(row)!=2
                or not all(isinstance(value,str) and value for value in row) or len(row[1])!=64
                or any(c not in '0123456789abcdef' for c in row[1]) for row in identities)):
            raise ValueError('ModernBERT token budget row identities are invalid')
        ids=[row[0] for row in identities]
        if len(set(ids))!=len(ids) or identities!=sorted(identities):raise ValueError('Token budget row IDs are duplicated or unordered')
        role_ids[role]=set(ids)
        if hashlib.sha256('\n'.join(sorted(ids)).encode()).hexdigest()!=run.get('hparams',{}).get('role_ids_hash',{}).get(role):
            raise ValueError('ModernBERT token budget role IDs differ from the model request')
        fields=['rows','content_tokens','tokens_with_specials','max_content_tokens','max_tokens_with_specials']
        if (not isinstance(counts,dict) or any(isinstance(counts.get(k),bool) or not isinstance(counts.get(k),int)
                or counts[k]<0 for k in fields) or counts['rows']!=len(ids)
                or counts.get('includes_padding') is not False
                or counts.get('scope')!='unique_visible_rows_not_optimizer_exposures'):
            raise ValueError('ModernBERT token budget numeric counts are invalid')
        if (counts['tokens_with_specials']<counts['content_tokens'] or counts['max_tokens_with_specials']<counts['max_content_tokens']
                or counts['max_content_tokens']>counts['content_tokens'] or counts['max_tokens_with_specials']>counts['tokens_with_specials']
                or (not ids and any(counts[k] for k in fields))):
            raise ValueError('ModernBERT token budget counts are inconsistent')
    if not role_ids['train'] or role_ids['train']&role_ids['select'] or run.get('strict_no_val') and role_ids['select']:
        raise ValueError('ModernBERT token budget role coverage is invalid')
    assets=read_json(Path(root)/'assets/assets.lock.json');source=assets['models']['modernbert']
    if (assets['asset_hash']!=run['asset_hash'] or len(source.get('revision',''))!=40
            or budget.get('tokenizer_revision')!=source['revision'] or budget.get('request_hash')!=canonical_hash({
                'run':run,'visible_roles':budget['role_input_identity'],'tokenizer':source})):
        raise ValueError('ModernBERT token budget tokenizer identity changed')
    _pin(root,files,f'token_budget:{artifact}',path)
    return budget


def verify_anchor_choice(root,node,choice,files):
    """Authenticate the complete fixed search and its V-only deterministic winner."""
    if choice.get('selection_policy')=='fixed_defaults':
        from .fixed_policy import verify_fixed_choice
        return verify_fixed_choice(root,node,choice,files)
    records=choice.get('candidates',[]);grid=node['candidates']
    if len(records)!=len(grid) or len(grid)!=6:
        raise PermissionError('All six declared anchor candidates must be complete')
    indexes=[record['candidate_index'] for record in records]
    if sorted(indexes)!=list(range(6)):
        raise ValueError('Anchor candidate identities are missing or duplicated')
    metric=node['metric'];verified=[]
    for record in records:
        index=record['candidate_index'];candidate=grid[index]
        if record['hparams']!=candidate:raise ValueError('Anchor candidate hyperparameters changed')
        status=_checkpoint(root,record['run_id'],files)
        path=Path(root)/'runs'/record['run_id']/'status.json'
        if hash_file(path)!=record['run_status_sha256']:raise ValueError('Anchor candidate status hash changed')
        run=status.get('run',{})
        seed=node['training_seed'] if node['model_id'] in {'bilstm','distilbert','modernbert'} else None
        if (status.get('run_id')!=record['run_id'] or run.get('run_id')!=record['run_id']
                or run.get('model_id')!=node['model_id'] or run.get('condition_id')!=node['anchor_condition_id']
                or run.get('training_seed')!=seed or run.get('strict_no_val') is not False
                or any(run.get('hparams',{}).get(k)!=v for k,v in candidate.items())):
            raise ValueError('Anchor checkpoint model request identity changed')
        validation=status.get('validation',{})
        value=validation.get(metric)
        if validation!=record['validation'] or value is None or not np.isfinite(value):
            raise ValueError('Anchor validation evidence is invalid or changed')
        verified.append((float(value),index,record))
    best=min(verified,key=lambda item:(-item[0],item[1]))[2]
    if (choice.get('status')!='complete' or choice.get('hparams')!=best['hparams']
            or choice.get('selected_run_id')!=best['run_id'] or choice.get('validation')!=best['validation']
            or choice.get('evidence_sha256')!=best['run_status_sha256']):
        raise ValueError('Declared anchor selection is not the fixed validation winner')
    return best


def _verify_pilot(root,node,pilot,choices,files):
    if pilot.get('status')!='complete' or pilot.get('source_partition')!='discovery':
        raise PermissionError(f'Real human D pilot is pending: {node["pilot_id"]}')
    path=bounded_path(Path(root)/pilot['results_file'],root)
    if hash_file(path)!=pilot['results_sha256']:raise ValueError('Pilot result bytes changed')
    payload=read_json(path);records=payload['pairs']
    if len(records)!=50 or len({r['pair']['group_id'] for r in records})!=50:
        raise PermissionError('Each real pilot requires 50 independent valid D base groups')
    dataset=node['dataset_id'];model='tfidf_svm' if node['mechanism']=='M1' else 'distilbert'
    checkpoint=choices[f'select:{dataset}:{model}']['selected_run_id']
    prepared=read_json(Path(root)/'data/prepared_manifest.json');meta=prepared['datasets'][dataset]
    import pyarrow.parquet as pq
    source={row['sample_id']:row for row in pq.read_table(Path(root)/meta['partitions']['discovery']['file'],
        columns=['sample_id','labels','group_id']).to_pylist()}
    discordant=0
    for record in records:
        pair=record['pair'];validate_pair(pair)
        if pair['mechanism']!=node['mechanism'] or pair['source_partition']!='discovery':
            raise ValueError('Pilot mechanism/source identity changed')
        original=source.get(pair['sample_id'])
        if original is None or tuple(original['labels'])!=tuple(pair['gold_labels']) or original['group_id']!=pair['group_id']:
            raise ValueError('Pilot sample/gold/group does not match the actual D source')
        if record.get('checkpoint_run_id')!=checkpoint:
            raise ValueError('Pilot did not use the declared fixed anchor checkpoint')
        if record.get('edits_sha256')!=canonical_hash(pair):raise ValueError('Pilot edits changed after freezing')
        if datetime.fromisoformat(record['predicted_at_utc'])<=datetime.fromisoformat(record['edits_frozen_at_utc']):
            raise ValueError('Pilot predictions preceded edit/review freezing')
        a=np.asarray(record['intervention_scores'],dtype=float);b=np.asarray(record['control_scores'],dtype=float)
        if a.shape!=(len(meta['label_names']),) or b.shape!=a.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError('Pilot score vectors are invalid')
        if node['mechanism']=='M1':
            original_scores=np.asarray(record.get('original_scores'),dtype=float)
            if original_scores.shape!=a.shape or not np.isfinite(original_scores).all():
                raise ValueError('M1 original-input score vector is missing or invalid')
        gold=int(pair['gold_labels'][0]);discordant+=int((int(a.argmax())==gold)!=(int(b.argmax())==gold))
    planned=plan_pair_count(discordant,50)
    if pilot.get('n_planned')!=planned or pilot.get('discordant')!=discordant:
        raise ValueError('Pilot sample-size planning does not match actual paired responses')
    from .pilots import _verified_edits,_verify_result_edits,_selected_checkpoint
    frozen_edits=_verified_edits(root,node['pilot_id'])
    _verify_result_edits(root,pilot,frozen_edits)
    _,verified_checkpoint,output,_=_selected_checkpoint(root,frozen_edits['context'])
    if verified_checkpoint!=checkpoint or payload.get('checkpoint_status_sha256')!=hash_file(output/'status.json'):
        raise ValueError('Pilot predictions are not bound to the fixed checkpoint evidence')
    _pin(root,files,f'pilot:{node["pilot_id"]}',path)
    return {'pilot_n':50,'discordant':discordant,'n_planned':planned,'checkpoint_run_id':checkpoint,
            'human_reviews_present':True,'source_partition':'discovery'}


def freeze_stage(stage: str,root: Path,inputs: dict[str,Any]) -> dict[str,Any]:
    root=Path(root).resolve()
    if stage=='design':return verify_design_lock(root)
    destination=_path(root,stage);design_lock=verify_design_lock(root)
    design=read_json(root/'matrix/design_manifest.json');files={}
    _pin(root,files,'design_lock',root/'design.lock.json')
    _pin(root,files,'design_manifest',root/'matrix/design_manifest.json')
    bindings=design_lock['bindings'];pilots_checked={};human_verified=False
    if stage=='selection':
        selections=inputs or read_json(root/'selections.json')
        choices=selections.get('choices',{});pilots=selections.get('discovery_pilots',{})
        # Human work is a real prerequisite, never a synthetically filled flag.
        for node in design['discovery_pilot_nodes']:
            if pilots.get(node['pilot_id'],{}).get('status')!='complete':
                raise PermissionError(f'Real human discovery pilot pending: {node["pilot_id"]}')
        for node in design['selection_nodes']:
            choice=choices.get(node['node_id'])
            if not choice or choice.get('status')!='complete' or choice.get('hparams') not in node['candidates']:
                raise PermissionError(f'Anchor selection is unresolved: {node["node_id"]}')
            verify_anchor_choice(root,node,choice,files)
        for node in design['discovery_pilot_nodes']:
            pilots_checked[node['pilot_id']]=_verify_pilot(root,node,pilots[node['pilot_id']],choices,files)
        human_verified=len(pilots_checked)==len(design['discovery_pilot_nodes'])==3
        if not human_verified:raise PermissionError('All three actual human pilots must be verified')
        if selections!=read_json(root/'selections.json'):
            raise ValueError('Selection input differs from the persisted selection evidence')
        _pin(root,files,'M1_review_clarification',root/'training/m1-review-clarification.md')
        _pin(root,files,'audit_sampling_clarification',root/'training/audit-sampling-clarification.md')
        for name in ['selections.json','run_manifest.json','contrasts.json']:
            _pin(root,files,name,root/name)
        manifest=read_json(root/'run_manifest.json')
        assert_resolved_coverage(design,manifest)
    else:
        selection=verify_freeze('selection',root)
        human_verified=selection['actual_human_pilots_verified']
        _pin(root,files,'selection_lock',root/'selection.lock.json')
        manifest=read_json(root/'run_manifest.json')
        runs=assert_resolved_coverage(design,manifest)
        if not runs:raise PermissionError('No actual required checkpoints are enumerated')
        for run in runs:
            _checkpoint(root,run.get('artifact_id',run['run_id']),files,run.get('expected_request_hash'),
                        run.get('expected_model_request_hash'))
            if run.get('run',{}).get('model_id')=='modernbert':verify_token_budget(root,run,files)
        for name in ['run_manifest.json','contrasts.json','thresholds.json']:
            _pin(root,files,name,root/name)
        contrasts=read_json(root/'contrasts.json');thresholds=read_json(root/'thresholds.json')
        if contrasts.get('design_hash')!=design['design_hash'] or not contrasts.get('items'):
            raise ValueError('Primary contrast manifest is missing or uses a different design')
        if thresholds.get('design_hash')!=design['design_hash'] or not {r.get('artifact_id',r['run_id']) for r in runs}<=set(thresholds.get('runs',{})):
            raise ValueError('Threshold manifest does not cover every required physical checkpoint')
    source=_sources()
    lock={'schema_version':1,'stage':stage,'protocol_hash':bindings['protocol_hash'],
        'design_hash':design['design_hash'],'recorded_at_utc':datetime.now(timezone.utc).isoformat(),
        'files':files,'source_files':source,'source_inventory':sorted(source),
        'actual_human_pilots_verified':human_verified,
        'pilots':pilots_checked,'official_test_predictions_allowed':stage=='evaluation',
        'global_required_training_scope':True}
    lock['lock_hash']=canonical_hash(lock)
    atomic_write_json(destination,lock,root)
    return lock


def verify_freeze(stage: str,root: Path) -> dict[str,Any]:
    root=Path(root).resolve();path=_path(root,stage)
    if not path.is_file():raise PermissionError(f'A verified global {stage} freeze is required')
    lock=read_json(path)
    if lock.get('stage')!=stage or canonical_hash({k:v for k,v in lock.items() if k!='lock_hash'})!=lock['lock_hash']:
        raise ValueError('Global freeze content/hash changed')
    if lock.get('actual_human_pilots_verified') is not True:
        raise PermissionError('Actual human pilots have not been verified for this freeze')
    design=verify_design_lock(root)
    if lock['protocol_hash']!=design['bindings']['protocol_hash'] or lock['design_hash']!=design['bindings']['design_hash']:
        raise ValueError('Global freeze protocol/design identity changed')
    current=_sources()
    if sorted(current)!=lock['source_inventory'] or current!=lock['source_files']:
        raise ValueError('Research code inventory/hash changed after global freeze')
    for record in lock['files'].values():
        target=bounded_path(root/record['path'],root)
        if not target.is_file() or hash_file(target)!=record['sha256']:
            raise ValueError(f'Global frozen artifact hash changed: {target}')
    if stage=='evaluation':verify_freeze('selection',root)
    return lock
