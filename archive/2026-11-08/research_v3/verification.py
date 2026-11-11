"""Independent byte, support, input-span, access and design-graph verification."""
from __future__ import annotations

from collections import Counter
from datetime import datetime,timezone
import gzip
import hashlib
from pathlib import Path
from typing import Any

from .config import PROJECT_ROOT,canonical_hash
from .matrix import verify_design_lock,strict_hparams
from .partitions import clean_text,text_digest
from .storage import atomic_write_json,bounded_path,hash_file,read_json


def verified_file(root: Path,relative: str,expected: str) -> Path:
    path=bounded_path(Path(root)/relative,root)
    if not path.is_file() or hash_file(path)!=expected:
        raise ValueError(f'File hash changed or missing: {path}')
    return path


def validate_forward_evidence(evidence: dict[str,Any]) -> None:
    checks=evidence.get('checks',[])
    if not evidence.get('all_passed') or not checks or evidence.get('actual_model_training_run') or evidence.get('official_test_predictions_generated'):
        raise ValueError('Incomplete or invalid forward evidence')
    for check in checks:
        if check.get('gradients_created') or not check.get('weight_versions_unchanged') or not check.get('finite_outputs'):
            raise ValueError('Invalid forward/gradient evidence')
    expected={(family,task) for family in ['distilbert','modernbert','bilstm']
              for task in ['single_label','multi_label']}
    if {(r.get('family'),r.get('task')) for r in checks}!=expected:
        raise ValueError('Missing single-label/multilabel forward combinations')


def _rows(path: Path):
    import pyarrow.parquet as pq
    for batch in pq.ParquetFile(path).iter_batches(batch_size=4096):
        yield from batch.to_pylist()


def audit_prepared_artifacts(root: Path,prepared: dict[str,Any]) -> dict[str,Any]:
    groups={};exact_purposes={};total=0;datasets={}
    for dataset,meta in prepared['datasets'].items():
        input_meta=meta['inputs']
        input_path=verified_file(root,input_meta['file'],input_meta['file_sha256'])
        visible=iter(_rows(input_path));input_n=0
        partitions={}
        for purpose,record in meta['partitions'].items():
            path=verified_file(root,record['file'],record['sha256'])
            counts=[0]*len(meta['label_names']);n=0
            for row in _rows(path):
                n+=1
                for label in row['labels']:
                    if label<0 or label>=len(counts):
                        raise ValueError('Out-of-range prepared label')
                    counts[label]+=1
                if purpose=='quarantine':
                    continue
                normalized=clean_text(row['text'])
                if not normalized:
                    raise ValueError('Empty source retained in a legal partition')
                group=row['group_id']
                if group in groups and groups[group]!=purpose:
                    raise ValueError('Global duplicate group partition overlap')
                groups[group]=purpose
                exact=text_digest(row['text'])
                if exact in exact_purposes and exact_purposes[exact]!=purpose:
                    raise ValueError('Exact text crosses legal partitions independently of group IDs')
                exact_purposes[exact]=purpose
                entry=next(visible,None)
                if entry is None or entry['sample_id']!=row['sample_id'] or entry['partition']!=purpose:
                    raise ValueError('Visible-input IDs do not match the legal source order')
                input_n+=1
                if entry['start_char']!=0 or normalized[:entry['end_char']]!=entry['text']:
                    raise ValueError('Visible-input character span mismatch')
                if entry['normalized_source_hash']!=hashlib.sha256(normalized.encode()).hexdigest():
                    raise ValueError('Visible-input full source hash mismatch')
                if entry['text_hash']!=hashlib.sha256(entry['text'].encode()).hexdigest():
                    raise ValueError('Visible-input shared text hash mismatch')
                if not (0<=entry['content_tokens']<=254) or entry['content_tokens']!=min(254,entry['full_tokens']):
                    raise ValueError('Visible-input subword budget mismatch')
            if n!=record['rows'] or counts!=record['label_support']:
                raise ValueError('Prepared source row/support counts changed')
            mask=[label for label,count in enumerate(counts) if 0<count<n]
            if mask!=record['ap_mask']:
                raise ValueError('Frozen AP support mask is not supported by actual labels')
            partitions[purpose]={'rows':n,'label_support':counts,'ap_mask':mask}
            total+=n
        if next(visible,None) is not None or input_n!=input_meta['rows']:
            raise ValueError('Visible inputs contain extra or missing rows')
        if meta['task'] in {'multilabel','multi_label'} and not partitions['select']['ap_mask']:
            raise ValueError('Multilabel selection AP mask is empty')
        datasets[dataset]={'partitions':partitions,'visible_inputs_checked':input_n}
    return {'total_source_rows_checked':total,'global_group_overlap_count':0,
            'independent_exact_overlap_count':0,'visible_character_spans_checked_for_every_row':True,
            'datasets':datasets}


def audit_conditions(root: Path,design: dict[str,Any],prepared: dict[str,Any]) -> dict[str,Any]:
    checked=0;datasets={}
    for dataset,meta in prepared['datasets'].items():
        record=meta['partitions']['train']
        lookup={r['sample_id']:tuple(r['labels']) for r in _rows(root/record['file'])}
        conditions=[c for c in design['conditions'] if c['dataset_id']==dataset]
        scale_prefixes={}
        for condition in conditions:
            path=verified_file(root,condition['train_ids_file'],condition['train_ids_file_sha256'])
            with gzip.open(path,'rt',encoding='utf-8') as stream:
                ids=tuple(line.rstrip('\n') for line in stream)
            if len(ids)!=len(set(ids)) or not set(ids).issubset(lookup):
                raise ValueError('Condition has duplicate IDs or reads outside the legal T pool')
            cm=condition['metadata']
            if cm['pool_manifest_hash']!=meta['manifest_hash']:
                raise ValueError('Condition is bound to a different training pool')
            if hashlib.sha256('\n'.join(ids).encode()).hexdigest()!=cm['sample_hash']:
                raise ValueError('Condition sample hash changed')
            if hashlib.sha256('\n'.join(sorted(ids)).encode()).hexdigest()!=condition['training_set_hash']:
                raise ValueError('Condition canonical training set hash changed')
            counts=[0]*len(meta['label_names'])
            for sid in ids:
                for label in lookup[sid]:
                    counts[label]+=1
            if len(ids)!=cm['rows'] or counts!=cm['label_support']:
                raise ValueError('Condition actual support differs from declared support')
            if condition['experiment_id']=='E2':
                if any(n!=cm['shots_per_class'] for n in counts) or not cm['strict_no_val']:
                    raise ValueError('Strict K-shot budget violated')
            if condition['experiment_id']=='E1' and not cm['full_pool']:
                scale_prefixes.setdefault(condition['data_seed'],[]).append(ids)
            checked+=1
        for prefixes in scale_prefixes.values():
            prefixes.sort(key=len)
            for smaller,larger in zip(prefixes,prefixes[1:]):
                if smaller!=larger[:len(smaller)]:
                    raise ValueError('Scale subsets are not nested')
        datasets[dataset]={'legal_training_pool':len(lookup),'conditions_checked':len(conditions)}
    return {'conditions_checked':checked,'all_ids_unique_and_within_T':True,
            'nested_scale_prefixes_verified':True,'datasets':datasets}


def verify_pretraining(p: dict[str,Any],root: Path) -> dict[str,Any]:
    root=Path(root);failures=[];checks={}
    def check(name,action):
        try:
            checks[name]=action()
        except Exception as exc:
            failures.append({'check':name,'error_type':type(exc).__name__,'error':str(exc)})
    def gates():
        execution=p['execution']
        if any(execution.get(key,False) for key in ['training_enabled','test_predictions_enabled','actual_fitting_permitted']):
            raise ValueError('Pretraining access gates must remain disabled')
        for name in ['selection.lock.json','evaluation.freeze.json']:
            if (root/name).exists():
                raise ValueError(f'A training-dependent lock exists before training: {name}')
        return {'training_disabled':True,'official_test_predictions_disabled':True}
    check('access_gates',gates)
    def assets():
        lock=read_json(root/'assets/assets.lock.json')
        for record in lock['files']:
            verified_file(PROJECT_ROOT,record['path'],record['sha256'])
        return {'verified_files':len(lock['files']),'asset_hash':lock['asset_hash']}
    check('assets',assets)
    def environment():
        e=read_json(root/'environment.json')
        if not e['runtime_ready'] or e.get('actual_model_training_run') or not e['gpu'].get('arithmetic_verified'):
            raise ValueError('Actual environment evidence is not ready')
        return {'python':e['python_version'],'gpu':e['gpu']['name'],'actual_gpu_arithmetic':True}
    check('environment',environment)
    def forward():
        e=read_json(root/'forward_checks.json');validate_forward_evidence(e)
        if e['protocol_hash']!=p['_protocol_hash']:
            raise ValueError('Forward evidence uses a different protocol')
        return {'checks':len(e['checks']),'device':e['device'],'dtype':e['dtype'],'no_gradients_or_weight_changes':True}
    check('forward_only',forward)
    def prepared_check():
        prepared=read_json(root/'data/prepared_manifest.json')
        if not prepared.get('complete') or prepared['identity']['protocol_hash']!=p['_protocol_hash']:
            raise ValueError('Prepared data are incomplete or bound to a different protocol')
        if set(prepared['datasets'])!=set(p['scope']['single_text_datasets']):
            raise ValueError('Incomplete dataset coverage')
        if canonical_hash({k:v for k,v in prepared.items() if k!='prepared_hash'})!=prepared['prepared_hash']:
            raise ValueError('Prepared manifest hash changed')
        return audit_prepared_artifacts(root,prepared)
    check('data_and_inputs',prepared_check)
    def matrix_check():
        lock=verify_design_lock(root)
        if lock['bindings']['protocol_hash']!=p['_protocol_hash']:
            raise ValueError('Design lock uses a different protocol')
        design=read_json(root/'matrix/design_manifest.json')
        if design['selection_lock_created'] or design['evaluation_freeze_created']:
            raise ValueError('Matrix falsely claims training-dependent freeze completion')
        if canonical_hash({k:v for k,v in design.items() if k!='design_hash'})!=design['design_hash']:
            raise ValueError('Design manifest hash changed')
        run_ids={r['run_id'] for r in design['physical_runs']}
        if len(run_ids)!=len(design['physical_runs']) or any(r['run_id'] not in run_ids for r in design['logical_runs']):
            raise ValueError('Invalid physical-run deduplication or logical aliases')
        for run in design['physical_runs']:
            if run['strict_no_val'] and run['hparam_binding']!={'fixed':strict_hparams(run['model_id'],p)}:
                raise ValueError('E2 has an anchor dependency or changed fixed defaults')
            if run['status']!='planned_not_run':
                raise ValueError('Pretraining matrix contains an executed training state')
        result=audit_conditions(root,design,read_json(root/'data/prepared_manifest.json'))
        return {**result,'physical_fit_nodes':len(run_ids),'logical_run_nodes':len(design['logical_runs']),
                'anchor_candidate_fit_nodes':design['budget']['anchor_candidate_fit_nodes'],
                'engineering_fit_nodes':len(design['engineering_smoke_nodes']),
                'design_lock_hash':lock['lock_hash'],'training_dependent_selections':'not_run'}
    check('design_and_conditions',matrix_check)
    def tests():
        evidence=read_json(root/'test-evidence.json')
        for suite in evidence['suites']:
            if suite['exit_code']!=0 or suite['tests_run']<1:
                raise ValueError('A required test suite failed or did not run')
            verified_file(PROJECT_ROOT,suite['log'],suite['log_sha256'])
        if {r['name'] for r in evidence['suites']}!={'research','legacy'}:
            raise ValueError('Both research and original regression suites are required')
        return {'suites':[{k:r[k] for k in ['name','tests_run','exit_code']} for r in evidence['suites']]}
    check('tests',tests)
    complete=not failures
    result={'schema_version':1,'scope':'all_pre_model_training_preparation',
        'recorded_at_utc':datetime.now(timezone.utc).isoformat(),'protocol_hash':p['_protocol_hash'],
        'complete':complete,'checks':checks,'failures':failures,
        'gates':{'G0a_runtime_and_forward':all(k in checks for k in ['environment','forward_only']),
                 'G1_full_data_and_shared_input':'data_and_inputs' in checks,
                 'G2a_structural_design':'design_and_conditions' in checks,
                 'G0_full_training_smoke':'not_run','G2b_selection':'not_run','G3_evaluation_freeze':'not_run'},
        'deliberately_pending':['E0 engineering fits','anchor hyperparameter selections',
            'three genuine D pilots and human review','selection.lock','all actual fits and checkpoints','evaluation.freeze','Q predictions'],
        'actual_model_training_run':False,'official_test_predictions_generated':False}
    atomic_write_json(root/'pretraining-verification.json',result,root)
    return result
