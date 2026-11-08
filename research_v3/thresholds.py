"""Fixed operative thresholds and the auxiliary one-global-V-threshold rule.

The primary probability cutoff remains .5; SVM margins/logits remain 0. The
validation optimum is auxiliary metadata only. Strict E2 and single-label runs
never load V, D or Q. No labelwise cutoff optimization is performed.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .config import canonical_hash
from .storage import atomic_write_json, bounded_path, hash_file, read_json
from .types import Example, VisibleInput


def _operative(score_kind):
    if score_kind not in {'probability','svm_margin','logit'}:
        raise ValueError(f'Unknown score kind: {score_kind}')
    return .5 if score_kind=='probability' else 0.


def _ids_hash(ids):
    return hashlib.sha256('\n'.join(sorted(ids)).encode()).hexdigest()


def choose_global_threshold(labels: Any, scores: Any, score_kind: str,
                            label_mask: Any) -> dict[str, Any]:
    """Choose one auxiliary cutoff on V using the frozen supported-label mask.

    Candidates are NumPy linear percentiles 5,10,...,95 of all full-target V
    scores, plus the operative cutoff. Score each with masked Macro-F1,
    zero_division=0. Ties prefer nearer the operative cutoff, then larger.
    The returned operative cutoff is always the unchanged primary rule.
    """
    operative = _operative(score_kind)
    y = np.asarray(labels); s = np.asarray(scores,dtype=np.float64)
    if s.ndim!=2 or not s.size or y.shape!=s.shape or not np.isfinite(s).all():
        raise ValueError('Threshold selection needs finite nonempty full-target N by C scores/labels')
    if not np.isin(y,[0,1]).all():
        raise ValueError('Threshold selection requires binary multilabel targets')
    if score_kind=='probability' and ((s<0).any() or (s>1).any()):
        raise ValueError('Probability scores must lie in [0,1]')
    try:
        raw = list(label_mask)
    except TypeError as error:
        raise ValueError('A frozen nonempty label mask is required') from error
    if (not raw or any(isinstance(i,(bool,np.bool_)) or not isinstance(i,(int,np.integer)) for i in raw)
            or len(set(raw))!=len(raw) or not set(raw)<=set(range(s.shape[1]))):
        raise ValueError('Frozen label mask must contain unique valid label indices')
    mask = [int(i) for i in raw]
    support = y[:,mask].sum(axis=0)
    if np.any(support<=0) or np.any(support>=len(y)):
        raise ValueError('Frozen V label mask requires positive and negative examples')
    quantiles = list(range(5,100,5))
    candidates = sorted(set([operative,*np.percentile(s,quantiles,method='linear').tolist()]))
    metrics = []
    target = y[:,mask].astype(bool)
    for threshold in candidates:
        prediction = s[:,mask]>=threshold
        tp = (prediction & target).sum(axis=0)
        fp = (prediction & ~target).sum(axis=0)
        fn = (~prediction & target).sum(axis=0)
        denominator = 2*tp+fp+fn
        f1 = np.divide(2*tp,denominator,out=np.zeros(len(mask),dtype=float),where=denominator!=0)
        metrics.append({'threshold':float(threshold),'masked_macro_f1':float(f1.mean())})
    chosen = min(metrics,key=lambda row:(-row['masked_macro_f1'],abs(row['threshold']-operative),-row['threshold']))
    return {'score_kind':score_kind,'operative':operative,'chosen':chosen['threshold'],
            'chosen_masked_macro_f1':chosen['masked_macro_f1'],'candidates':candidates,'metrics':metrics,
            'label_mask':mask,'candidate_percentiles':quantiles,'percentile_method':'linear',
            'percentile_population':'all_full_target_V_scores','selection_metric':'masked_macro_f1',
            'tie_break':'nearest_operative_then_larger','optimization':'one_global_threshold',
            'auxiliary_only':True}


def _checkpoint(root,entry):
    artifact = entry.get('artifact_id',entry['run_id'])
    if not isinstance(artifact,str) or artifact in {'.','..'} or Path(artifact).name!=artifact:
        raise ValueError('Invalid checkpoint artifact identity')
    output = bounded_path(root/'runs'/artifact,root/'runs')
    path = output/'status.json'
    if not path.is_file():
        raise PermissionError(f'Checkpoint is pending: {artifact}')
    status = read_json(path)
    if status.get('status')!='complete' or not status.get('reload_verified'):
        raise PermissionError(f'Checkpoint is not complete/reload verified: {artifact}')
    declared = entry.get('run')
    expected = entry.get('expected_model_request_hash')
    if (not declared or not expected or canonical_hash(declared)!=expected
            or canonical_hash(status.get('run',{}))!=expected
            or declared.get('run_id')!=artifact or status.get('run_id')!=artifact):
        raise ValueError(f'Checkpoint model request identity changed: {artifact}')
    files = status.get('files',{})
    if not files or 'bundle.joblib' not in files:
        raise ValueError(f'Checkpoint has no authenticated saved model files: {artifact}')
    for relative,expected_file in files.items():
        saved = bounded_path(output/relative,output)
        if not saved.is_file() or hash_file(saved)!=expected_file:
            raise ValueError(f'Checkpoint artifact hash changed: {artifact}/{relative}')
    _operative(status['score_kind'])
    return output,status,hash_file(path)


def _source_files(root,metadata):
    selection = metadata['partitions']['select']; inputs = metadata['inputs']
    records = {}
    for name,spec,key in [('selection',selection,'sha256'),('inputs',inputs,'file_sha256')]:
        path = bounded_path(root/spec['file'],root)
        if not path.is_file() or hash_file(path)!=spec[key]:
            raise ValueError(f'Prepared V {name} source bytes/hash changed')
        records[name] = {'path':spec['file'],'sha256':spec[key]}
    return records


def _load_selection(root,metadata):
    """Read only actual V targets and V common-visible rows; no T/D/Q targets."""
    import pyarrow.parquet as pq
    source = pq.read_table(bounded_path(root/metadata['partitions']['select']['file'],root),
                           columns=['sample_id','labels','official_split','group_id']).to_pylist()
    rows = tuple(Example(metadata['dataset_id'],r['sample_id'],'',tuple(r['labels']),r['official_split'],r['group_id'])
                 for r in source)
    visible = pq.read_table(bounded_path(root/metadata['inputs']['file'],root),
                            columns=['sample_id','text','start_char','end_char','content_tokens','full_tokens','text_hash'],
                            filters=[('partition','=','select')]).to_pylist()
    inputs = {r['sample_id']:VisibleInput(**r) for r in visible}
    if len(inputs)!=len(visible):
        raise ValueError('Duplicate frozen V input IDs')
    return rows,inputs


def _selection_context(root,metadata,run,loader):
    files = _source_files(root,metadata)
    rows,inputs = loader(root,metadata)
    ordered = tuple(sorted(rows,key=lambda row:row.sample_id))
    ids = [row.sample_id for row in ordered]
    if not ids or len(ids)!=len(set(ids)) or len(ids)!=metadata['partitions']['select']['rows']:
        raise ValueError('Actual V IDs/row-count identity changed')
    if set(inputs)!=set(ids) or _ids_hash(ids)!=run['hparams']['role_ids_hash']['select']:
        raise ValueError('Actual V IDs hash differs from model selection binding')
    visible = []
    y = np.zeros((len(ordered),len(metadata['label_names'])),dtype=np.int8)
    for i,row in enumerate(ordered):
        if row.dataset_id!=metadata['dataset_id'] or not row.group_id or not row.labels:
            raise ValueError('Invalid actual V source/gold/group identity')
        if len(row.labels)!=len(set(row.labels)) or not set(row.labels)<=set(range(y.shape[1])):
            raise ValueError('Invalid actual V target label identity')
        value = inputs[row.sample_id]
        if value.sample_id!=row.sample_id or value.text_hash!=hashlib.sha256(value.text.encode()).hexdigest():
            raise ValueError('Frozen V common-visible input text hash changed')
        visible.append(value); y[i,list(row.labels)] = 1
    return ordered,tuple(visible),y,files


def freeze_thresholds(root: Path, run_manifest: dict[str, Any], prepared_metadata: dict[str, Any],
                      *, loader: Callable | None=None, scorer: Callable | None=None) -> dict[str, Any]:
    """Authenticate every physical checkpoint, then atomically save thresholds.

    Defaults read actual prepared V parquet files and call train.score_saved,
    whose bundle format also supports mechanism checkpoints. ``loader(root,
    metadata)->(Example rows, VisibleInput dict)`` and ``scorer(output, visible)``
    injection are only for isolated fixtures, never replacement research labels.
    Source file hashes, selection ID hashes and model identities are checked
    with either loader. Writes only root/thresholds.json, after complete success.
    """
    root = Path(root).resolve()
    if run_manifest.get('status')!='resolved' or run_manifest.get('unresolved_nodes'):
        raise PermissionError('Threshold freeze requires a resolved run manifest')
    metadata = prepared_metadata.get('datasets',prepared_metadata)
    unique = {}; aliases = defaultdict(list)
    for entry in run_manifest.get('runs',[]):
        artifact = entry.get('artifact_id',entry['run_id'])
        if artifact in unique and (entry['dataset_id']!=unique[artifact]['dataset_id']
                                   or entry['run']!=unique[artifact]['run']
                                   or entry.get('expected_model_request_hash')!=unique[artifact].get('expected_model_request_hash')):
            raise ValueError('Conflicting model request aliases for one physical artifact')
        unique[artifact] = entry; aliases[artifact].append(entry.get('planned_run_id',entry['run_id']))
    if not unique or not run_manifest.get('design_hash'):
        raise ValueError('Threshold freeze needs all declared physical runs and design identity')
    # Validate the entire checkpoint set before any validation-label read.
    checked = {artifact:_checkpoint(root,entry) for artifact,entry in unique.items()}
    for artifact,entry in unique.items():
        if entry['run']['input_hash']!=metadata[entry['dataset_id']]['inputs']['input_hash']:
            raise ValueError('Prepared common-input identity differs from the model request')
    runs = {}; selection_cache = {}
    loader = loader or _load_selection
    for artifact,entry in unique.items():
        output,status,status_hash = checked[artifact]
        run = entry['run']; meta = metadata[entry['dataset_id']]
        if meta['task'] not in {'single_label','multilabel','multi_label'}:
            raise ValueError('Unknown target task')
        multi = meta['task'] in {'multilabel','multi_label'}
        record = {'run_id':artifact,'expected_model_request_hash':entry['expected_model_request_hash'],
                  'status_sha256':status_hash,'score_kind':status['score_kind'],
                  'operative':_operative(status['score_kind']) if multi else None,'auxiliary':None,
                  'planned_run_ids':aliases[artifact],'selection_labels_accessed':0,
                  'strict_no_val':run['strict_no_val'],'task':meta['task']}
        if multi and not run['strict_no_val']:
            cache_key = (entry['dataset_id'],run['hparams']['role_ids_hash']['select'])
            if cache_key not in selection_cache:
                selection_cache[cache_key] = _selection_context(root,meta,run,loader)
            ordered,visible,y,files = selection_cache[cache_key]
            if scorer is None:
                from .train import score_saved
                actual_scorer = score_saved
            else:
                actual_scorer = scorer
            scores = np.asarray(actual_scorer(output,visible),dtype=float)
            if scores.shape!=y.shape:
                raise ValueError('Saved model lost the full V score-column shape')
            mask = meta['partitions']['select']['ap_mask']
            if run['hparams'].get('select_ap_mask',mask)!=mask:
                raise ValueError('Frozen model/V selection mask identity changed')
            record['auxiliary'] = choose_global_threshold(y,scores,status['score_kind'],mask)
            record['selection_labels_accessed'] = len(ordered)
            record['selection'] = {'partition':'select','rows':len(ordered),
                'ids_hash':_ids_hash([row.sample_id for row in ordered]),'input_hash':run['input_hash'],
                'prepared_manifest_hash':meta['manifest_hash'],'files':files,
                'targets_and_input_hash':canonical_hash([
                    (row.sample_id,row.group_id,list(row.labels),value.text_hash)
                    for row,value in zip(ordered,visible)])}
        else:
            record['auxiliary_reason'] = 'strict_no_validation' if run['strict_no_val'] else 'single_label_argmax'
        runs[artifact] = record
    # Refuse checkpoint/source replacement during a potentially long scoring pass.
    for artifact,entry in unique.items():
        if _checkpoint(root,entry)[2]!=checked[artifact][2]:
            raise ValueError('Checkpoint status bytes changed during threshold freeze')
    for ds,_ in selection_cache:
        _source_files(root,metadata[ds])
    result = {'schema_version':1,'status':'complete','design_hash':run_manifest['design_hash'],
              'run_manifest_hash':run_manifest.get('manifest_hash'),'runs':runs,
              'unique_physical_runs':len(runs),'operative_rule':'probability_0.5_other_continuous_0; single_label_argmax',
              'auxiliary_thresholds_change_primary_predictions':False,
              'official_test_predictions_generated':False,'test_labels_accessed':False,
              'discovery_labels_accessed':False}
    result['thresholds_hash'] = canonical_hash(result)
    atomic_write_json(root/'thresholds.json',result,root)
    return result
